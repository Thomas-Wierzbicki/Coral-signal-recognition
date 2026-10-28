"""Paired preregistered feature comparison on reused development data."""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from torch import nn
from build_modulation_dataset import sha
from check_modulation_learning import state_digest
from diagnose_training_lr import score,write_json
from iq_feature_experiment import (ARMS,ARCHITECTURE,PREPROCESSING,feature_tensor,
    make_feature_model,restore_feature_checkpoint)
from train_modulation12 import load_metadata,load_split,metrics
from train_se_gain_development import select_condition_groups,atomic_checkpoint,LEVELS
from torchsig_probe import CLASSES

PLAN=Path(__file__).with_name('IQ_FEATURE_COMPARISON_PLAN_V1.json')


def compare(runs,seeds):
    pairs={}
    for seed in map(str,seeds):
        if set(runs.get(seed,{}))!=set(ARMS):continue
        a,b=(runs[seed][arm]['selected_validation'] for arm in ARMS)
        pairs[seed]=dict(ce_candidate_minus_baseline=b['loss']-a['loss'],
            macro_f1_candidate_minus_baseline=b['macro_f1']-a['macro_f1'],
            accuracy_candidate_minus_baseline=b['accuracy']-a['accuracy'],
            both_primary_metrics_improve=b['loss']<a['loss'] and b['macro_f1']>a['macro_f1'])
    complete=set(pairs)==set(map(str,seeds))
    return dict(complete=complete,pairs=pairs,
        directional_development_advantage=all(v['both_primary_metrics_improve'] for v in pairs.values()) if complete else None,
        interpretation='Reused development validation; no independent final accuracy or isolated feature cause')


def detailed_score(model,x,y,snr,device):
    result=score(model,x,y,device)
    fsk=['2fsk','2gfsk','2gmsk']
    result['fsk_family_mean_recall']=float(np.mean([result['recall'][label] for label in fsk]))
    high=np.flatnonzero(np.isfinite(snr)&(snr>=15))
    result['snr_ge15']=signal_subset_score(model,x[high],y[high],device)
    result['snr_accuracy_signal_classes_only']={}
    for level in LEVELS:
        indices=np.flatnonzero(snr==level)
        values=signal_subset_score(model,x[indices],y[indices],device)
        result['snr_accuracy_signal_classes_only'][str(level)]=dict(count=len(indices),accuracy=values['accuracy'])
    return result


def signal_subset_score(model,x,y,device):
    if not len(y):return dict(count=0,accuracy=None)
    model.eval();predictions=[];losses=[]
    with torch.inference_mode():
        for start in range(0,len(y),32):
            logits=model(x[start:start+32].to(device))
            losses.extend(nn.functional.cross_entropy(logits,y[start:start+32].to(device),reduction='none').cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    result=metrics(y.numpy(),np.asarray(predictions));result.pop('macro_f1')
    counts=np.bincount(y.numpy(),minlength=len(CLASSES))
    for index,label in enumerate(CLASSES):
        if not counts[index]:result['recall'][label]=None
    result['loss']=float(np.mean(losses))
    return result


def train_case(model,x,y,vx,vy,folder,device,plan,context,evaluate=score):
    """Fixed budget, fresh optimizer and permutation generator; save full state each epoch."""
    optimizer=torch.optim.Adam(model.parameters(),**{k:plan['optimizer'][k] for k in ('lr','eps')})
    generator=torch.Generator().manual_seed(plan['batch_order_seed'])
    order_digest=hashlib.sha256();steps=0;history=[];best_loss=float('inf')
    for epoch in range(1,plan['epochs']+1):
        start=time.perf_counter();model.train();loss_sum=0.
        permutation=torch.randperm(len(y),generator=generator)
        order_digest.update(permutation.numpy().tobytes())
        for batch in permutation.split(plan['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(model(x[batch].to(device)),y[batch].to(device))
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Missing/nonfinite gradient')
            optimizer.step();steps+=1;loss_sum+=float(loss.detach())*len(batch)
        validation=evaluate(model,vx,vy,device)
        if not np.isfinite(validation['loss']):raise ValueError('Nonfinite validation loss')
        row=dict(epoch=epoch,steps=steps,optimization_loss=loss_sum/len(y),validation=validation,
            seconds=time.perf_counter()-start,batch_order_sha256=order_digest.hexdigest())
        history.append(row)
        saved=dict(context,model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,steps=steps,
            generator_state=generator.get_state(),torch_rng_state=torch.get_rng_state(),
            cuda_rng_states=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        if validation['loss']<best_loss:
            best_loss=validation['loss'];atomic_checkpoint(folder/'best_checkpoint.pt',saved)
        atomic_checkpoint(folder/'last_checkpoint.pt',saved)
        write_json(folder/'history.json',history)
        print(f'Seed {context["model_seed"]}, {context["arm"]}, {epoch}/{plan["epochs"]}: '
              f'CE {validation["loss"]:.4f}, Zuordnung {validation["accuracy"]:.1%}',flush=True)
    return history


def prepare(data,source,plan,output):
    meta=load_metadata(data)
    for name,key in [('metadata.json','metadata_sha256'),('train.npz','train_sha256'),('validation.npz','validation_sha256')]:
        if sha(data/name)!=plan['data'][key]:raise ValueError('Dataset differs from registered plan: '+name)
    if not (source/'DEVELOPMENT_COMPLETE').is_file():raise ValueError('Source development incomplete')
    selection=source/'training_selection.npz'
    if sha(selection)!=plan['data']['original_selection_sha256']:
        raise ValueError('Original selection hash differs')
    seen=set()
    train,ty,tids,tsnr=load_split(data,'train',meta,seen)
    val,vy,vids,vsnr=load_split(data,'validation',meta,seen)
    with np.load(data/'train.npz',allow_pickle=False) as d:tc=d['condition_id']
    with np.load(data/'validation.npz',allow_pickle=False) as d:vc=d['condition_id']
    if set(tc.tolist())&set(vc.tolist()):raise ValueError('Train/validation condition overlap')
    indices,groups=select_condition_groups(ty,tc,tsnr,plan['data']['condition_groups_per_snr'],
        plan['data']['condition_selection_seed'])
    selected=dict(source_id=tids[indices],condition_id=tc[indices],y=ty[indices],snr_db=tsnr[indices])
    with np.load(selection,allow_pickle=False) as saved:
        if set(saved.files)!=set(selected):raise ValueError('Selection fields differ')
        for name,values in selected.items():
            if values.dtype.kind=='f':equal=np.array_equal(values,saved[name],equal_nan=True)
            else:equal=np.array_equal(values,saved[name])
            if not equal:raise ValueError('Original selected rows differ: '+name)
    if len(indices)!=plan['data']['train_examples'] or len(vy)!=plan['data']['validation_examples']:
        raise ValueError('Registered example count differs')
    shutil.copyfile(selection,output/'training_selection.npz')
    initial={}
    for seed in plan['model_seeds']:
        path=source/f'seed{seed}'/'initial_state.pt'
        state=torch.load(path,map_location='cpu',weights_only=True)
        if state_digest(state)!=plan['source_initial_state_digest'][str(seed)]:
            raise ValueError('Saved initial tensors differ')
        initial[str(seed)]=state
    return (train[indices],torch.from_numpy(ty[indices]),tsnr[indices],
            val,torch.from_numpy(vy),vsnr,initial,groups)


def run(args):
    plan=json.loads(PLAN.read_text())
    if args.device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    if args.source is None:
        sources=[p.parent for p in Path('models').glob('se-gain-development-*/DEVELOPMENT_COMPLETE')]
        if len(sources)!=1:raise ValueError('Expected one development source; specify --source')
        args.source=sources[0]
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(complete=False,runs={},plan_id=plan['id'],test_opened=False)
    try:
        tiq,y,tsnr,viq,vy,vsnr,initial,groups=prepare(args.data,args.source,plan,args.output)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        protocol=dict(plan=plan,plan_sha256=sha(PLAN),source=str(args.source.resolve()),
            source_selection_sha256=sha(args.source/'training_selection.npz'),selected_condition_ids=groups,
            classes=CLASSES,torch_version=torch.__version__,numpy_version=np.__version__,cuda=torch.version.cuda,
            device=args.device,device_name=torch.cuda.get_device_name(0) if args.device=='cuda' else 'CPU',
            deterministic_algorithms=True,tf32=False,test_opened=False,
            cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
            source_initial_file_sha256={str(s):sha(args.source/f'seed{s}'/'initial_state.pt') for s in plan['model_seeds']},
            sources={n:sha(Path(__file__).with_name(n)) for n in [Path(__file__).name,'iq_feature_experiment.py',
                'modulation_reference.py','se_gain_experiment.py','train_se_gain_development.py',
                'train_modulation12.py','diagnose_training_lr.py','check_modulation_learning.py',
                'build_modulation_dataset.py','torchsig_probe.py']})
        write_json(args.output/'protocol.json',protocol);report['protocol']=protocol
        for seed in plan['model_seeds']:
            parent=args.output/f'seed{seed}';parent.mkdir();runs={};report['runs'][str(seed)]=runs
            model=make_feature_model(initial[str(seed)]);paired=copy.deepcopy(model.state_dict())
            paired_sha=state_digest(paired);torch.save(paired,parent/'initial_four_channel_state.pt')
            # Same four-channel graph/weights: new feature kernels are exactly zero initially.
            with torch.inference_mode():
                model=model.to(args.device).eval()
                a=model(feature_tensor(tiq[:32],ARMS[0]).to(args.device))
                b=model(feature_tensor(tiq[:32],ARMS[1]).to(args.device))
                if not torch.equal(a,b):raise ValueError('Paired initial outputs differ')
            expected_orders=None
            for arm in ARMS:
                folder=parent/arm;folder.mkdir()
                torch.manual_seed(seed)
                if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
                model=make_feature_model(initial[str(seed)]);model.load_state_dict(paired);model.to(args.device)
                if state_digest(model.state_dict())!=paired_sha:raise ValueError('Paired initial state differs')
                start=time.perf_counter();x=feature_tensor(tiq,arm);vx=feature_tensor(viq,arm)
                feature_seconds=time.perf_counter()-start
                context=dict(architecture=ARCHITECTURE,preprocessing=PREPROCESSING,arm=arm,
                    se_gain=2.,activation='tf-compatible',model_seed=seed,protocol_sha256=sha(args.output/'protocol.json'),
                    selection_sha256=sha(args.output/'training_selection.npz'))
                history=train_case(model,x,y,vx,vy,folder,args.device,plan,context)
                orders=[row['batch_order_sha256'] for row in history]
                if expected_orders is not None and orders!=expected_orders:raise ValueError('Paired minibatch orders differ')
                expected_orders=orders
                if history[-1]['steps']!=plan['epochs']*plan['steps_per_epoch']:
                    raise ValueError('Update budget differs')
                saved=torch.load(folder/'best_checkpoint.pt',map_location='cpu',weights_only=True)
                selected=restore_feature_checkpoint(saved).to(args.device)
                validation=detailed_score(selected,vx,vy,vsnr,args.device)
                best_loss=min(row['validation']['loss'] for row in history)
                if not np.isclose(validation['loss'],best_loss,atol=1e-7,rtol=1e-6):
                    raise ValueError('Selected loss reproduction failed')
                if arm=='iq_zero' and torch.count_nonzero(selected.features[0].weight[:,2:]).item()!=0:
                    raise ValueError('Baseline inactive kernels changed')
                result=dict(initial_four_channel_sha256=paired_sha,paired_initial_outputs_exact=True,
                    paired_batch_order_verified=True,best_epoch=saved['epoch'],epochs=plan['epochs'],steps=history[-1]['steps'],
                    feature_preprocessing_seconds=feature_seconds,
                    parameter_count=sum(p.numel() for p in selected.parameters()),
                    extra_kernel_norm=float(selected.features[0].weight[:,2:].detach().norm()),
                    selected_training=detailed_score(selected,x,y,tsnr,args.device),selected_validation=validation,
                    final_validation=history[-1]['validation'],
                    best_checkpoint_sha256=sha(folder/'best_checkpoint.pt'),last_checkpoint_sha256=sha(folder/'last_checkpoint.pt'),
                    history_sha256=sha(folder/'history.json'),selected_state_sha256=state_digest(selected.state_dict()))
                runs[arm]=result;write_json(folder/'report.json',result)
                (folder/'CASE_COMPLETE').write_text('OK\n')
                report['summary']=compare(report['runs'],plan['model_seeds'])
                write_json(args.output/'report.json',report)
                del selected,model,x,vx
        report['complete']=True
    except Exception as error:
        report['error']=str(error)
        raise
    finally:
        write_json(args.output/'report.json',report)
        print('Bericht:',args.output/'report.json',flush=True)
    (args.output/'FEATURE_COMPARISON_COMPLETE').write_text('OK\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/modulation12-v1'))
    p.add_argument('--source',type=Path)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--output',type=Path,default=Path('models')/('iq-feature-comparison-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%fZ')))
    args=p.parse_args()
    try:run(args)
    except (OSError,ValueError,KeyError,RuntimeError,AssertionError) as error:
        p.exit(1,'Merkmalsvergleich fehlgeschlagen: '+str(error)+'\n')
