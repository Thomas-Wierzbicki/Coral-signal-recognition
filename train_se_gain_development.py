"""Paired larger development comparison; test split is deliberately not opened."""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from build_modulation_dataset import sha
from check_modulation_learning import state_digest
from diagnose_training_collapse import write_json
from diagnose_training_lr import score
from se_gain_experiment import make_gain_model, set_gain, restore_gain_checkpoint
from train_modulation12 import load_metadata, load_split, as_tensor
from torchsig_probe import CLASSES


LEVELS=[-5.,0.,5.,10.,15.,20.,25.]


def select_condition_groups(labels, conditions, snr, per_snr=20, seed=2080):
    if per_snr<1: raise ValueError('Positive group count required')
    labels=np.asarray(labels);conditions=np.asarray(conditions);snr=np.asarray(snr)
    if conditions.shape!=labels.shape or snr.shape!=labels.shape:
        raise ValueError('Condition arrays have wrong shape')
    groups={}
    for index,group in enumerate(conditions.tolist()): groups.setdefault(group,[]).append(index)
    by_snr={level:[] for level in LEVELS}
    for group,indices in groups.items():
        if sorted(labels[indices].tolist())!=list(range(12)):
            raise ValueError('Condition must contain each class exactly once')
        signal=[i for i in indices if labels[i]!=11]
        values=snr[signal]
        if not np.isfinite(values).all() or not np.all(values==values[0]) or float(values[0]) not in by_snr:
            raise ValueError('Inconsistent condition SNR')
        noise=[i for i in indices if labels[i]==11][0]
        if not np.isnan(snr[noise]): raise ValueError('Noise must have undefined SNR')
        by_snr[float(values[0])].append(group)
    rng=np.random.default_rng(seed); selected=[]
    for level in LEVELS:
        candidates=sorted(by_snr[level])
        if len(candidates)<per_snr: raise ValueError('Too few conditions at SNR '+str(level))
        selected.extend(rng.choice(candidates,per_snr,replace=False).tolist())
    wanted=set(selected)
    indices=np.flatnonzero(np.asarray([group in wanted for group in conditions.tolist()]))
    return indices,selected


def atomic_checkpoint(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');torch.save(value,temporary);temporary.replace(path)


def snr_evaluation(model,x,y,snr,device):
    result=score(model,x,y,device)
    result['snr_accuracy_signal_classes_only']={}
    for level in LEVELS:
        indices=np.flatnonzero(snr==level)
        if len(indices):
            with torch.inference_mode():
                predictions=[]
                for batch in torch.from_numpy(indices).split(32):
                    predictions.extend(model(x[batch].to(device)).argmax(1).cpu().tolist())
            result['snr_accuracy_signal_classes_only'][str(level)]=dict(
                count=len(indices),accuracy=float(np.mean(np.asarray(predictions)==y[indices].numpy())))
    return result


def comparison(runs,seeds):
    pairs={}
    for seed,cases in runs.items():
        if set(cases)!={'gain1','gain2'}: continue
        a,b=(cases[name]['selected_validation'] for name in ['gain1','gain2'])
        pairs[seed]=dict(loss_gain2_minus_gain1=b['loss']-a['loss'],
                        macro_f1_gain2_minus_gain1=b['macro_f1']-a['macro_f1'],
                        accuracy_gain2_minus_gain1=b['accuracy']-a['accuracy'],
                        both_primary_metrics_improve=b['loss']<a['loss'] and b['macro_f1']>a['macro_f1'])
    complete=set(pairs)==set(map(str,seeds))
    return dict(pairs=pairs,complete=complete,
                directional_development_advantage=all(p['both_primary_metrics_improve'] for p in pairs.values()) if complete else None,
                interpretation='reused development validation, checkpoint selected by validation loss; not independent final accuracy')


def run(args):
    if args.epochs<1 or args.per_snr<1: raise ValueError('Positive epochs/groups required')
    if args.device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    if args.output is None:
        args.output=Path('models')/('se-gain-development-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    if args.output.exists():raise ValueError('Output exists; choose a new folder')
    plan_path=Path(__file__).with_name('SE_GAIN_DEVELOPMENT_PLAN_V1.json')
    plan=json.loads(plan_path.read_text());meta=load_metadata(args.data)
    if sha(args.data/'metadata.json')!=plan['controls']['metadata_sha256']:
        raise ValueError('Dataset metadata differs from plan')
    if sha(args.data/'train.npz')!=plan['controls']['train_sha256']:
        raise ValueError('Training data differs from plan')
    seen=set()
    train_iq,train_y,train_ids,train_snr=load_split(args.data,'train',meta,seen)
    val_iq,val_y,val_ids,val_snr=load_split(args.data,'validation',meta,seen)
    with np.load(args.data/'train.npz',allow_pickle=False) as d: train_conditions=d['condition_id']
    with np.load(args.data/'validation.npz',allow_pickle=False) as d: val_conditions=d['condition_id']
    if train_conditions.shape!=train_y.shape or val_conditions.shape!=val_y.shape:
        raise ValueError('Invalid condition IDs')
    if set(train_ids.tolist())&set(val_ids.tolist()) or set(train_conditions.tolist())&set(val_conditions.tolist()):
        raise ValueError('Train/validation overlap')
    indices,groups=select_condition_groups(train_y,train_conditions,train_snr,args.per_snr)
    train_iq,train_y,train_ids,train_snr=(a[indices] for a in [train_iq,train_y,train_ids,train_snr])
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'training_selection.npz',source_id=train_ids,
                        condition_id=train_conditions[indices],y=train_y,snr_db=train_snr)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    protocol=dict(plan=plan,plan_sha256=sha(plan_path),classes=CLASSES,epochs=args.epochs,per_snr=args.per_snr,
        default_design=(args.epochs==100 and args.per_snr==20),train_examples=len(train_y),validation_examples=len(val_y),
        device=args.device,device_name=torch.cuda.get_device_name(0) if args.device=='cuda' else 'CPU',
        torch_version=torch.__version__,numpy_version=np.__version__,cuda=torch.version.cuda,
        dataset_metadata_sha256=sha(args.data/'metadata.json'),train_sha256=sha(args.data/'train.npz'),
        validation_sha256=sha(args.data/'validation.npz'),selection_sha256=sha(args.output/'training_selection.npz'),
        selected_condition_ids=groups,test_opened=False,validation_role='reused development selection set',
        sources={name:sha(Path(__file__).with_name(name)) for name in [Path(__file__).name,
            'se_gain_experiment.py','modulation_reference.py','train_modulation12.py',
            'diagnose_training_lr.py','check_modulation_learning.py','diagnose_training_collapse.py',
            'build_modulation_dataset.py','torchsig_probe.py']})
    write_json(args.output/'protocol.json',protocol)
    x,y=as_tensor(train_iq),torch.from_numpy(train_y)
    vx,vy=as_tensor(val_iq),torch.from_numpy(val_y)
    del train_iq,val_iq
    runs={}
    for seed in plan['model_seeds']:
        torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        model=make_gain_model(1.);initial=copy.deepcopy(model.state_dict());initial_sha=state_digest(initial)
        if initial_sha!=plan['controls']['initial_sha256_by_seed'][str(seed)]:
            raise ValueError('Initialization differs from plan')
        parent=args.output/('seed'+str(seed));parent.mkdir()
        torch.save(initial,parent/'initial_state.pt');runs[str(seed)]={}
        for gain in [1.,2.]:
            name='gain'+str(int(gain));folder=parent/name;folder.mkdir()
            set_gain(model,gain);model.load_state_dict(initial);model.to(args.device)
            if state_digest(model.state_dict())!=initial_sha:raise ValueError('Unpaired initial weights')
            optimizer=torch.optim.Adam(model.parameters(),lr=.0001,eps=1e-8)
            generator=torch.Generator().manual_seed(2080)
            history=[];best_loss=float('inf');best_epoch=None;steps=0
            for epoch in range(1,args.epochs+1):
                start=time.perf_counter();model.train();loss_sum=0.
                for batch in torch.randperm(len(y),generator=generator).split(32):
                    optimizer.zero_grad(set_to_none=True)
                    logits=model(x[batch].to(args.device))
                    loss=nn.functional.cross_entropy(logits,y[batch].to(args.device))
                    if not torch.isfinite(loss):raise ValueError('Nonfinite optimization loss')
                    loss.backward()
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise ValueError('Missing/nonfinite gradient')
                    optimizer.step();steps+=1;loss_sum+=float(loss.detach())*len(batch)
                validation=score(model,vx,vy,args.device)
                if not np.isfinite(validation['loss']):raise ValueError('Nonfinite validation loss')
                row=dict(epoch=epoch,steps=steps,examples_processed=epoch*len(y),
                         optimization_loss=loss_sum/len(y),validation=validation,seconds=time.perf_counter()-start)
                history.append(row)
                checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,steps=steps,
                    generator_state=generator.get_state(),se_gain=gain,activation='tf-compatible',
                    architecture='Harper2023-experimental-se-gain',model_seed=seed,
                    protocol_sha256=sha(args.output/'protocol.json'),selection_sha256=protocol['selection_sha256'])
                if validation['loss']<best_loss:
                    best_loss=validation['loss'];best_epoch=epoch
                    atomic_checkpoint(folder/'best_checkpoint.pt',checkpoint)
                atomic_checkpoint(folder/'last_checkpoint.pt',checkpoint)
                write_json(folder/'history.json',history)
                print(f'Seed {seed}, {name}, Epoche {epoch}/{args.epochs}: '
                      f'Validierungsverlust {validation["loss"]:.4f}, Zuordnung {validation["accuracy"]:.1%}',flush=True)
            best=torch.load(folder/'best_checkpoint.pt',map_location='cpu',weights_only=True)
            selected=restore_gain_checkpoint(best).to(args.device)
            selected_train=score(selected,x,y,args.device)
            selected_val=snr_evaluation(selected,vx,vy,val_snr,args.device)
            result=dict(model_seed=seed,se_gain=gain,initial_sha256=initial_sha,best_epoch=best_epoch,
                        selection_rule='strict minimum mean validation cross entropy; first tie',
                        selected_training=selected_train,selected_validation=selected_val,
                        selected_state_sha256=state_digest(selected.state_dict()),
                        final_validation=history[-1]['validation'],epochs=args.epochs,steps=steps,
                        best_checkpoint_sha256=sha(folder/'best_checkpoint.pt'),
                        last_checkpoint_sha256=sha(folder/'last_checkpoint.pt'),history_sha256=sha(folder/'history.json'))
            if not np.isclose(selected_val['loss'],best_loss,atol=1e-7,rtol=1e-6):
                raise ValueError('Selected checkpoint fails validation loss reproduction')
            runs[str(seed)][name]=result
            write_json(folder/'report.json',result)
            write_json(args.output/'report.json',dict(protocol=protocol,runs=runs,
                         summary=comparison(runs,plan['model_seeds'])))
            del selected;model.cpu()
            (folder/'CASE_COMPLETE').write_text('OK\n')
    (args.output/'DEVELOPMENT_COMPLETE').write_text('OK\n')
    print('Ergebnisse:',args.output.resolve())
    print('Entwicklungsvalidierung, keine unabhaengige Testgenauigkeit. Testdatei nicht geoeffnet.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/modulation12-v1'))
    p.add_argument('--output',type=Path)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--per-snr',type=int,default=20)
    try:run(p.parse_args())
    except (OSError,ValueError,RuntimeError,KeyError,AssertionError) as error:
        p.exit(1,'Entwicklungsvergleich fehlgeschlagen: '+str(error)+'\n')
