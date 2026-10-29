"""Inference-only paired comparison of all four frozen CNNs with classical holdout."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
import classical_modulation_baseline as classical
import evaluate_classical_holdout as holdout_module
from check_modulation_learning import state_digest
from diagnose_training_lr import score
from iq_feature_experiment import ARMS,feature_tensor,restore_feature_checkpoint
from train_modulation12 import load_metadata,load_split

PLAN=Path(__file__).with_name('CNN_CLASSICAL_HOLDOUT_PLAN_V1.json')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def checked_file(folder,name,digest):
    path=folder/name
    if path.resolve().parent!=folder.resolve() or classical.sha(path)!=digest:
        raise ValueError('Artifact path/hash differs: '+str(path))
    return path


def paired_counts(y,classic,cnn):
    if y.shape!=classic.shape or y.shape!=cnn.shape:raise ValueError('Unpaired prediction shapes')
    a=classic==y;b=cnn==y
    return dict(count=len(y),both_correct=int(np.sum(a&b)),classical_only_correct=int(np.sum(a&~b)),
        cnn_only_correct=int(np.sum(~a&b)),both_wrong=int(np.sum(~a&~b)),
        prediction_disagreement=int(np.sum(classic!=cnn)))


def load_holdout(folder,data,classical_source=None):
    report=read(folder/'report.json')
    if not (folder/'CLASSICAL_HOLDOUT_COMPLETE').is_file() or report['complete'] is not True or report['classifier_unchanged'] is not True:
        raise ValueError('Holdout not complete')
    if report['plan']!=read(holdout_module.PLAN) or report['plan_sha256']!=classical.sha(holdout_module.PLAN):
        raise ValueError('Holdout plan differs')
    protocol=read(checked_file(folder,'protocol.json',report['artifact_sha256']['protocol.json']))
    if protocol!=report['protocol']:raise ValueError('Holdout protocol/report differs')
    for name,digest in protocol['sources'].items():
        if classical.sha(Path(__file__).with_name(name))!=digest:raise ValueError('Holdout source code differs: '+name)
    source=classical_source or Path(protocol['source'])
    if classical.sha(source/'report.json')!=protocol['source_report_sha256']:
        raise ValueError('Classical source report differs')
    _,_,model,_,_=holdout_module.freeze_source(data,source)
    if classical.sha(source/'classifier.npz')!=protocol['classifier_sha256'] or holdout_module.array_digest(model)!=protocol['fitted_arrays_sha256']:
        raise ValueError('Frozen classical model differs')
    with np.load(checked_file(folder,'holdout.npz',report['artifact_sha256']['holdout.npz']),allow_pickle=False) as saved:
        arrays={key:saved[key] for key in ('iq','y','source_id','condition_id','snr_db')}
    n=report['plan']['total_examples'];a=arrays
    if a['iq'].shape!=(n,4096) or not np.isfinite(a['iq']).all() or a['iq'].dtype!=np.complex64:
        raise ValueError('Invalid holdout IQ shape/type')
    if any(a[name].shape!=(n,) for name in ('y','source_id','condition_id','snr_db')) or len(set(a['source_id'].tolist()))!=n:
        raise ValueError('Invalid holdout IDs/shapes')
    if not np.array_equal(np.bincount(a['y'],minlength=12),np.full(12,140)):raise ValueError('Holdout class counts differ')
    parameters=checked_file(folder,'parameters.jsonl',report['artifact_sha256']['parameters.jsonl'])
    rows=[json.loads(line) for line in parameters.read_text().splitlines()]
    if len(rows)!=n:raise ValueError('Holdout record count differs')
    for i,row in enumerate(rows):
        label=classical.CLASSES[int(a['y'][i])]
        if (row['source_id']!=a['source_id'][i] or row['condition_id']!=a['condition_id'][i]
                or row['label']!=label or row['iq_sha256']!=hashlib.sha256(a['iq'][i].tobytes()).hexdigest()):
            raise ValueError('Holdout IQ/label/ID join differs')
        if label=='noise':
            if row['parameters'] is not None or not np.isnan(a['snr_db'][i]):raise ValueError('Noise SNR differs')
        elif float(a['snr_db'][i])!=row['parameters']['snr_db']:raise ValueError('Holdout SNR differs')
    with np.load(checked_file(folder,'features_predictions.npz',report['artifact_sha256']['features_predictions.npz']),allow_pickle=False) as saved:
        if not np.array_equal(saved['source_id'],a['source_id']):raise ValueError('Classical prediction IDs/order differ')
        features=np.stack([classical.features(z) for z in a['iq']])
        if not np.array_equal(features,saved['features']):raise ValueError('Classical features no longer reproduce')
        predictions,_=classical.neighbors(model['train_scaled'],model['train_labels'],(features-model['mean'])/model['std'])
        if not np.array_equal(predictions,saved['predicted']):raise ValueError('Classical predictions do not reproduce')
    result=classical.evaluation(a['y'],predictions,a['snr_db'])
    if result!=report['holdout']:raise ValueError('Classical holdout metrics differ')
    return arrays,predictions,result


def load_cnn_source(folder,data):
    report=read(folder/'report.json');protocol=read(folder/'protocol.json')
    if not (folder/'FEATURE_COMPARISON_COMPLETE').is_file() or report['complete'] is not True:
        raise ValueError('CNN comparison source incomplete')
    expected=read(Path(__file__).with_name('IQ_FEATURE_COMPARISON_PLAN_V1.json'))
    if protocol!=report['protocol'] or protocol['plan']!=expected:
        raise ValueError('CNN source protocol/plan differs')
    if protocol['plan_sha256']!=classical.sha(Path(__file__).with_name('IQ_FEATURE_COMPARISON_PLAN_V1.json')):
        raise ValueError('CNN source plan hash differs')
    if set(report['runs'])!={'2081','2082'} or any(set(cases)!=set(ARMS) for cases in report['runs'].values()):
        raise ValueError('Expected all four CNN cases')
    for name,digest in protocol['sources'].items():
        if classical.sha(Path(__file__).with_name(name))!=digest:raise ValueError('CNN implementation differs: '+name)
    if classical.sha(folder/'training_selection.npz')!=expected['data']['original_selection_sha256']:
        raise ValueError('CNN training selection differs')
    meta=load_metadata(data)
    if classical.sha(data/'metadata.json')!=expected['data']['metadata_sha256'] or classical.sha(data/'validation.npz')!=expected['data']['validation_sha256']:
        raise ValueError('Original CNN validation data differs')
    iq,y,ids,snr=load_split(data,'validation',meta,set())
    return report,iq,torch.from_numpy(y)


def load_checkpoint(source,seed,arm,expected):
    folder=source/f'seed{seed}'/arm;path=folder/'best_checkpoint.pt'
    if not (folder/'CASE_COMPLETE').is_file() or classical.sha(path)!=expected['best_checkpoint_sha256']:
        raise ValueError('CNN checkpoint incomplete/hash differs')
    if classical.sha(folder/'history.json')!=expected['history_sha256']:raise ValueError('History hash differs')
    history=read(folder/'history.json')
    if len(history)!=100 or any(r['epoch']!=i+1 or r['steps']!=(i+1)*53 for i,r in enumerate(history)):
        raise ValueError('CNN training budget/history differs')
    best=min(history,key=lambda row:row['validation']['loss'])
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if (saved['model_seed']!=int(seed) or saved['arm']!=arm or saved['epoch']!=expected['best_epoch']
            or saved['epoch']!=best['epoch'] or saved['steps']!=saved['epoch']*53
            or saved['protocol_sha256']!=classical.sha(source/'protocol.json')
            or saved['selection_sha256']!=classical.sha(source/'training_selection.npz')):
        raise ValueError('CNN selected checkpoint provenance differs')
    model=restore_feature_checkpoint(saved)
    if state_digest(model.state_dict())!=expected['selected_state_sha256']:raise ValueError('CNN state digest differs')
    return model,path


def infer(model,x,device):
    model.eval();predictions=[]
    with torch.inference_mode():
        for start in range(0,len(x),32):
            logits=model(x[start:start+32].to(device))
            if logits.shape!=(min(32,len(x)-start),12) or not torch.isfinite(logits).all():raise ValueError('Invalid CNN logits')
            predictions.extend(logits.argmax(1).cpu().tolist())
    return np.asarray(predictions)


def run(args):
    if args.device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(complete=False,training_performed=False,old_test_opened=False,plan=read(PLAN),cases={})
    try:
        a,classic,classic_metrics=load_holdout(args.holdout,args.data,args.classical_source)
        source,viq,vy=load_cnn_source(args.source,args.data)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        protocol=dict(plan=report['plan'],plan_sha256=classical.sha(PLAN),device=args.device,
            torch_version=torch.__version__,numpy_version=np.__version__,cuda=torch.version.cuda,
            cnn_report_sha256=classical.sha(args.source/'report.json'),holdout_report_sha256=classical.sha(args.holdout/'report.json'),
            holdout_sha256=classical.sha(args.holdout/'holdout.npz'),
            sources={name:classical.sha(Path(__file__).with_name(name)) for name in [Path(__file__).name,
                'iq_feature_experiment.py','modulation_reference.py','se_gain_experiment.py','classical_modulation_baseline.py']})
        classical.write(args.output/'protocol.json',protocol);report['protocol']=protocol;report['classical']=classic_metrics
        for seed in ('2081','2082'):
            for arm in ARMS:
                expected=source['runs'][seed][arm];model,path=load_checkpoint(args.source,seed,arm,expected)
                before=state_digest(model.state_dict());model.to(args.device)
                original=score(model,feature_tensor(viq,arm),vy,args.device)
                prior=expected['selected_validation']
                if (original['confusion_matrix']!=prior['confusion_matrix'] or
                        not np.isclose(original['loss'],prior['loss'],atol=1e-7,rtol=1e-6)):
                    raise ValueError('CNN original validation reproduction failed')
                predictions=infer(model,feature_tensor(a['iq'],arm),args.device)
                result=classical.evaluation(a['y'],predictions,a['snr_db'])
                if state_digest(model.state_dict())!=before or classical.sha(path)!=expected['best_checkpoint_sha256']:
                    raise ValueError('CNN checkpoint/state changed')
                if any(p.grad is not None for p in model.parameters()):raise ValueError('Unexpected CNN gradients')
                name=f'{seed}-{arm}';filename=name+'-predictions.npz'
                np.savez_compressed(args.output/filename,predicted=predictions,source_id=a['source_id'])
                report['cases'][name]=dict(holdout=result,paired=paired_counts(a['y'],classic,predictions),
                    delta_cnn_minus_classical=dict(accuracy=result['accuracy']-classic_metrics['accuracy'],
                        macro_f1=result['macro_f1']-classic_metrics['macro_f1']),
                    original_validation_reproduced=True,checkpoint_unchanged=True,
                    checkpoint_sha256=expected['best_checkpoint_sha256'],selected_epoch=expected['best_epoch'],
                    prediction_file=filename,prediction_sha256=classical.sha(args.output/filename))
                classical.write(args.output/'report.json',report)
                print(name,f'Zuordnung {result["accuracy"]:.1%}, Macro-F1 {result["macro_f1"]:.4f}',flush=True)
                del model
        if classical.sha(args.holdout/'holdout.npz')!=protocol['holdout_sha256']:raise ValueError('Source holdout changed')
        report['complete']=True
    except Exception as error:
        report['error']=str(error);raise
    finally:
        classical.write(args.output/'report.json',report);print('Bericht:',args.output/'report.json',flush=True)
    (args.output/'CNN_CLASSICAL_HOLDOUT_COMPLETE').write_text('OK\n')
    return report


def find_unique(pattern,marker,explicit):
    if explicit is not None:return explicit
    folders=[p.parent for p in Path('models').glob(pattern+'/'+marker)]
    if len(folders)!=1:raise ValueError('Expected one '+marker+' source; specify --source/--holdout')
    return folders[0]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/modulation12-v1'))
    p.add_argument('--source',type=Path);p.add_argument('--holdout',type=Path);p.add_argument('--classical-source',type=Path)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--output',type=Path,default=Path('models')/('cnn-classical-holdout-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%fZ')))
    args=p.parse_args()
    try:
        args.source=find_unique('iq-feature-comparison-*','FEATURE_COMPARISON_COMPLETE',args.source)
        args.holdout=find_unique('classical-holdout-*','CLASSICAL_HOLDOUT_COMPLETE',args.holdout)
        run(args)
    except (OSError,ValueError,KeyError,RuntimeError,AssertionError) as error:
        p.exit(1,'Gepaarter Vergleich fehlgeschlagen: '+str(error)+'\n')
