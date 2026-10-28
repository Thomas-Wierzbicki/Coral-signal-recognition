"""Bounded paired continuation: restore weights, Adam and batch RNG; change only LR."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
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
from se_gain_experiment import restore_gain_checkpoint
import train_se_gain_development as development


BRANCHES=[('keep_lr',.0001),('lower_lr',.00001)]
SEEDS=[2081,2082]


def tree_digest(value):
    digest=hashlib.sha256()
    def append(raw):
        digest.update(str(len(raw)).encode()+b':'+raw)
    def walk(item):
        if torch.is_tensor(item):
            append(b'tensor');append(str(item.dtype).encode());append(str(tuple(item.shape)).encode())
            append(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item,dict):
            append(b'dict')
            for key in sorted(item,key=lambda k:(str(type(k)),repr(k))):walk(key);walk(item[key])
        elif isinstance(item,(tuple,list)):
            append(str(type(item)).encode())
            for member in item:walk(member)
        else:append((str(type(item))+':'+repr(item)).encode())
    walk(value);return digest.hexdigest()


def adam_digest(state,ignore_lr=False):
    if ignore_lr:
        state=dict(state,param_groups=[{k:v for k,v in group.items() if k!='lr'} for group in state['param_groups']])
    return tree_digest(state)


def validate_snapshot(saved,epoch=100,steps=5300):
    if (saved.get('se_gain')!=2. or saved.get('activation')!='tf-compatible'
        or saved.get('architecture')!='Harper2023-experimental-se-gain'
        or saved.get('epoch')!=epoch or saved.get('steps')!=steps):
        raise ValueError('Expected last gain2 checkpoint at epoch100/step5300')
    optimizer=saved['optimizer']
    if len(optimizer['param_groups'])!=1 or len(optimizer['state'])!=32:
        raise ValueError('Incomplete Adam state')
    group=optimizer['param_groups'][0]
    if (group.get('lr')!=.0001 or group.get('eps')!=1e-8 or tuple(group.get('betas',()))!=(.9,.999)
        or group.get('weight_decay')!=0 or group.get('amsgrad') or group.get('maximize')):
        raise ValueError('Unexpected source Adam settings')
    for item in optimizer['state'].values():
        if not {'step','exp_avg','exp_avg_sq'}<=set(item) or float(item['step'])!=steps:
            raise ValueError('Missing or reset Adam step/moments')
        if any(not torch.isfinite(item[name]).all() for name in ['exp_avg','exp_avg_sq']):
            raise ValueError('Nonfinite source Adam state')
    generator=torch.Generator();generator.set_state(saved['generator_state'])


def restore_training(saved,device,lr):
    model=restore_gain_checkpoint(saved).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.0001,eps=1e-8)
    optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
    if adam_digest(optimizer.state_dict())!=adam_digest(saved['optimizer']):
        raise ValueError('Adam state did not restore exactly')
    for group in optimizer.param_groups:group['lr']=lr
    if adam_digest(optimizer.state_dict(),True)!=adam_digest(saved['optimizer'],True):
        raise ValueError('Restoration changed more than the learning rate')
    generator=torch.Generator();generator.set_state(saved['generator_state'])
    return model,optimizer,generator


def choose_source(path):
    if path is not None:return Path(path)
    matches=[p for p in Path('models').glob('se-gain-development-*')
             if (p/'DEVELOPMENT_COMPLETE').is_file() and (p/'report.json').is_file()]
    if len(matches)!=1:raise ValueError('Specify --source models/se-gain-development-TIMESTAMP (not exactly one completed run)')
    return matches[0]


def registered_source_plan():
    path=Path(__file__).with_name('SE_GAIN_DEVELOPMENT_PLAN_V1.json')
    return json.loads(path.read_text()),sha(path)


def prepare_source(source,data):
    if not (source/'DEVELOPMENT_COMPLETE').is_file():raise ValueError('Source run is incomplete')
    report=json.loads((source/'report.json').read_text());protocol=report['protocol']
    if (not protocol.get('default_design') or protocol.get('epochs')!=100 or protocol.get('train_examples')!=1680
        or protocol.get('validation_examples')!=1080 or set(report['runs'])!=set(map(str,SEEDS))
        or protocol['plan'].get('id')!='se-gain-development-v1'):
        raise ValueError('Source run does not match preregistered development design')
    registered_plan,registered_hash=registered_source_plan()
    if protocol.get('plan_sha256')!=registered_hash or protocol['plan']!=registered_plan:
        raise ValueError('Source plan differs from registered design')
    if (protocol['dataset_metadata_sha256']!=protocol['plan']['controls']['metadata_sha256']
        or protocol['train_sha256']!=protocol['plan']['controls']['train_sha256']
        or protocol.get('classes')!=development.CLASSES):raise ValueError('Source data/class configuration differs')
    if json.loads((source/'protocol.json').read_text())!=protocol:raise ValueError('Source protocol/report mismatch')
    for name,expected in protocol['sources'].items():
        if sha(Path(__file__).with_name(name))!=expected:raise ValueError('Source code differs: '+name)
    if sha(source/'training_selection.npz')!=protocol['selection_sha256']:raise ValueError('Selection hash mismatch')
    meta=development.load_metadata(data)
    for filename,key in [('metadata.json','dataset_metadata_sha256'),('train.npz','train_sha256'),('validation.npz','validation_sha256')]:
        if sha(data/filename)!=protocol[key]:raise ValueError('Data hash mismatch: '+filename)
    seen=set()
    iq,labels,ids,snr=development.load_split(data,'train',meta,seen)
    viq,vy,vids,vsnr=development.load_split(data,'validation',meta,seen)
    with np.load(data/'train.npz',allow_pickle=False) as d:conditions=d['condition_id']
    with np.load(data/'validation.npz',allow_pickle=False) as d:vconditions=d['condition_id']
    if conditions.shape!=labels.shape or vconditions.shape!=vy.shape:raise ValueError('Invalid condition IDs')
    if set(ids.tolist())&set(vids.tolist()) or set(conditions.tolist())&set(vconditions.tolist()):
        raise ValueError('Train/validation overlap')
    indices,groups=development.select_condition_groups(labels,conditions,snr,20,2080)
    if groups!=protocol['selected_condition_ids'] or len(indices)!=1680 or len(vy)!=1080:
        raise ValueError('Condition selection or input counts differ')
    with np.load(source/'training_selection.npz',allow_pickle=False) as selected:
        for key,actual in [('source_id',ids[indices]),('condition_id',conditions[indices]),('y',labels[indices])]:
            np.testing.assert_array_equal(selected[key],actual)
        np.testing.assert_array_equal(selected['snr_db'],snr[indices])
    snapshots={};fingerprints={}
    expected_rng=torch.Generator().manual_seed(2080)
    for _ in range(100):torch.randperm(1680,generator=expected_rng)
    for seed in SEEDS:
        case=report['runs'][str(seed)]['gain2'];folder=source/('seed'+str(seed))/'gain2'
        if case.get('epochs')!=100 or case.get('steps')!=5300 or case.get('se_gain')!=2.:
            raise ValueError('Source case was not completed as planned')
        initial=torch.load(source/('seed'+str(seed))/'initial_state.pt',map_location='cpu',weights_only=True)
        if (state_digest(initial)!=protocol['plan']['controls']['initial_sha256_by_seed'][str(seed)]
            or state_digest(initial)!=case['initial_sha256']):raise ValueError('Source initialization differs')
        path=folder/'last_checkpoint.pt'
        if sha(path)!=case['last_checkpoint_sha256'] or sha(folder/'history.json')!=case['history_sha256']:
            raise ValueError('Source checkpoint/history hash mismatch')
        history=json.loads((folder/'history.json').read_text())
        if (len(history)!=100 or history[-1]['validation']!=case['final_validation']
            or any(row['epoch']!=i or row['steps']!=i*53 for i,row in enumerate(history,1))):
            raise ValueError('Source history/report mismatch')
        saved=torch.load(path,map_location='cpu',weights_only=True);validate_snapshot(saved)
        if (saved['model_seed']!=seed or saved['selection_sha256']!=protocol['selection_sha256']
            or saved['protocol_sha256']!=sha(source/'protocol.json')):raise ValueError('Checkpoint provenance mismatch')
        if not torch.equal(saved['generator_state'],expected_rng.get_state()):raise ValueError('Batch RNG state differs')
        snapshots[str(seed)]=saved;fingerprints[str(seed)]=sha(path)
    return dict(report=report,source=source,source_report_sha256=sha(source/'report.json'),
                checkpoint_sha256=fingerprints,snapshots=snapshots,
                x=development.as_tensor(iq[indices]),y=torch.from_numpy(labels[indices]),
                vx=development.as_tensor(viq),vy=torch.from_numpy(vy),validation_snr=vsnr)


def trajectory(history):
    transitions=[]
    for previous,current in zip(history,history[1:]):
        transitions.append(float(np.mean([abs(current['validation']['recall'][key]-previous['validation']['recall'][key])
                                           for key in current['validation']['recall']])))
    jumps=[b['validation']['loss']-a['validation']['loss'] for a,b in zip(history,history[1:])]
    return dict(largest_loss_increase=max([0.]+jumps),
        mean_absolute_class_recall_change_last20=float(np.mean(transitions[-20:])) if transitions else None,
        recall_range_last20_by_class={key:dict(minimum=min(r['validation']['recall'][key] for r in history[-20:]),
                                             maximum=max(r['validation']['recall'][key] for r in history[-20:]))
                                     for key in history[-1]['validation']['recall']},
        epoch_blocks=[dict(start=block[0]['epoch'],end=block[-1]['epoch'],
                     mean_loss=float(np.mean([r['validation']['loss'] for r in block])),
                     mean_accuracy=float(np.mean([r['validation']['accuracy'] for r in block])))
                     for block in [history[1:][-20:-10],history[1:][-10:]] if block])


def require_baseline(reproduced,baseline):
    if (reproduced['confusion_matrix']!=baseline['confusion_matrix']
        or not np.isclose(reproduced['loss'],baseline['loss'],atol=1e-7,rtol=1e-6)):
        raise ValueError('Starting checkpoint validation does not reproduce source')


def fit_branch(saved,inputs,device,lr,epochs,folder,baseline,protocol_hash):
    folder.mkdir(parents=True)
    model,optimizer,generator=restore_training(saved,device,lr)
    start_epoch=saved['epoch'];steps=saved['steps'];x,y,vx,vy=inputs
    reproduced=score(model,vx,vy,device)
    require_baseline(reproduced,baseline)
    checks=dict(model_sha256=state_digest(model.state_dict()),adam_sha256_except_lr=adam_digest(optimizer.state_dict(),True),
                generator_sha256=tree_digest(generator.get_state()),source_validation_reproduced=True)
    history=[dict(epoch=start_epoch,steps=steps,validation=reproduced)]
    best_loss=reproduced['loss'];best_epoch=start_epoch
    def checkpoint(epoch):
        return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),generator_state=generator.get_state(),
                    architecture='Harper2023-experimental-se-gain',activation='tf-compatible',se_gain=2.,
                    epoch=epoch,steps=steps,model_seed=saved['model_seed'],protocol_sha256=protocol_hash,
                    selection_sha256=saved['selection_sha256'],source_model_sha256=checks['model_sha256'])
    development.atomic_checkpoint(folder/'best_checkpoint.pt',checkpoint(start_epoch))
    for epoch in range(start_epoch+1,start_epoch+epochs+1):
        start=time.perf_counter();model.train();loss_sum=0.
        for batch in torch.randperm(len(y),generator=generator).split(32):
            optimizer.zero_grad(set_to_none=True)
            logits=model(x[batch].to(device));loss=nn.functional.cross_entropy(logits,y[batch].to(device))
            if not torch.isfinite(loss):raise ValueError('Nonfinite optimization loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Missing/nonfinite gradient')
            optimizer.step();steps+=1;loss_sum+=float(loss.detach())*len(batch)
        validation=score(model,vx,vy,device)
        if not np.isfinite(validation['loss']):raise ValueError('Nonfinite validation loss')
        history.append(dict(epoch=epoch,steps=steps,optimization_loss=loss_sum/len(y),validation=validation,
                            seconds=time.perf_counter()-start))
        if validation['loss']<best_loss:
            best_loss=validation['loss'];best_epoch=epoch
            development.atomic_checkpoint(folder/'best_checkpoint.pt',checkpoint(epoch))
        development.atomic_checkpoint(folder/'last_checkpoint.pt',checkpoint(epoch))
        write_json(folder/'history.json',history)
        print(f'Seed {saved["model_seed"]}, LR {lr:g}, Epoche {epoch}: Validierungsverlust '
              f'{validation["loss"]:.4f}, Zuordnung {validation["accuracy"]:.1%}',flush=True)
    best=torch.load(folder/'best_checkpoint.pt',map_location='cpu',weights_only=True)
    selected=restore_gain_checkpoint(best).to(device)
    selected_validation=score(selected,vx,vy,device)
    if not np.isclose(selected_validation['loss'],best_loss,atol=1e-7,rtol=1e-6):raise ValueError('Best checkpoint fails reproduction')
    result=dict(learning_rate=lr,start_epoch=start_epoch,end_epoch=history[-1]['epoch'],steps=steps,
        additional_steps=steps-saved['steps'],best_epoch=best_epoch,starting_checks=checks,
        source_validation=baseline,selected_validation=selected_validation,final_validation=history[-1]['validation'],
        selected_training=score(selected,x,y,device),trajectory=trajectory(history),
        best_checkpoint_sha256=sha(folder/'best_checkpoint.pt'),last_checkpoint_sha256=sha(folder/'last_checkpoint.pt'),
        history_sha256=sha(folder/'history.json'))
    del selected,model,optimizer
    return result,best


def summarize(runs):
    pairs={}
    for seed,cases in runs.items():
        if set(cases)!={'keep_lr','lower_lr'}:continue
        a,b=(cases[name]['selected_validation'] for name in ['keep_lr','lower_lr'])
        pairs[seed]=dict(loss_lower_minus_keep=b['loss']-a['loss'],macro_f1_lower_minus_keep=b['macro_f1']-a['macro_f1'],
                        accuracy_lower_minus_keep=b['accuracy']-a['accuracy'],
                        both_primary_metrics_improve=b['loss']<a['loss'] and b['macro_f1']>a['macro_f1'])
    complete=set(pairs)==set(map(str,SEEDS))
    return dict(pairs=pairs,complete=complete,
        lower_lr_directional_advantage=all(v['both_primary_metrics_improve'] for v in pairs.values()) if complete else None,
        interpretation='reused development validation; lower volatility alone is not recognition improvement')


def run(args):
    if args.epochs<1:raise ValueError('Positive continuation length required')
    if args.device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    source=choose_source(args.source)
    if args.output is None:
        args.output=source.parent/('se-gain-continuation-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%fZ'))
    if args.output.exists():raise ValueError('Output exists; choose a new folder')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    prepared=prepare_source(source,args.data)
    for seed in SEEDS:
        check_model=restore_gain_checkpoint(prepared['snapshots'][str(seed)]).to(args.device)
        reproduced=score(check_model,prepared['vx'],prepared['vy'],args.device)
        require_baseline(reproduced,prepared['report']['runs'][str(seed)]['gain2']['final_validation'])
        del check_model
    args.output.mkdir(parents=True)
    plan=Path(__file__).with_name('SE_GAIN_CONTINUATION_PLAN_V1.json')
    protocol=dict(plan=json.loads(plan.read_text()),plan_sha256=sha(plan),source=str(source.resolve()),
        source_report_sha256=prepared['source_report_sha256'],source_checkpoints=prepared['checkpoint_sha256'],
        data=str(args.data.resolve()),epochs=args.epochs,default_design=args.epochs==50,device=args.device,
        torch_version=torch.__version__,numpy_version=np.__version__,cuda=torch.version.cuda,
        device_name=torch.cuda.get_device_name(0) if args.device=='cuda' else 'CPU',test_opened=False,
        all_source_validation_checks_passed_before_updates=True,
        sources={name:sha(Path(__file__).with_name(name)) for name in [Path(__file__).name,
            'train_se_gain_development.py','diagnose_training_lr.py','diagnose_training_collapse.py',
            'se_gain_experiment.py','modulation_reference.py','train_modulation12.py','check_modulation_learning.py',
            'build_modulation_dataset.py','torchsig_probe.py']})
    write_json(args.output/'protocol.json',protocol)
    inputs=tuple(prepared[key] for key in ['x','y','vx','vy']);runs={}
    for seed in SEEDS:
        key=str(seed);runs[key]={}
        saved=prepared['snapshots'][key]
        baseline=prepared['report']['runs'][key]['gain2']['final_validation']
        for name,lr in BRANCHES:
            folder=args.output/('seed'+key)/name
            result,best=fit_branch(saved,inputs,args.device,lr,args.epochs,folder,baseline,sha(args.output/'protocol.json'))
            selected=restore_gain_checkpoint(best).to(args.device)
            result['selected_validation']=development.snr_evaluation(selected,prepared['vx'],prepared['vy'],
                                                                     prepared['validation_snr'],args.device)
            result['source_checkpoint_sha256']=prepared['checkpoint_sha256'][key]
            del selected,best
            if runs[key] and result['starting_checks']!=runs[key]['keep_lr']['starting_checks']:
                raise ValueError('Branches did not start from identical model, Adam and RNG state')
            runs[key][name]=result
            write_json(folder/'report.json',result)
            write_json(args.output/'report.json',dict(protocol=protocol,runs=runs,summary=summarize(runs)))
            (folder/'CASE_COMPLETE').write_text('OK\n')
    for seed in SEEDS:
        if sha(source/('seed'+str(seed))/'gain2/last_checkpoint.pt')!=prepared['checkpoint_sha256'][str(seed)]:
            raise ValueError('Source checkpoint changed during continuation')
    (args.output/'CONTINUATION_COMPLETE').write_text('OK\n')
    print('Ergebnisse:',args.output.resolve())
    print('Bitte report.json aus dem Hauptordner hochladen. Entwicklungswerte, kein unabhaengiger Test.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path)
    p.add_argument('--data',type=Path,default=Path('data/modulation12-v1'))
    p.add_argument('--output',type=Path)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--epochs',type=int,default=50)
    try:run(p.parse_args())
    except (OSError,ValueError,RuntimeError,KeyError,AssertionError) as error:
        p.exit(1,'Fortsetzung fehlgeschlagen: '+str(error)+'\n')
