"""Read-only layer diagnostics of a completed paired learning-rate experiment."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from build_modulation_dataset import sha
from check_modulation_learning import state_digest
from diagnose_training_collapse import write_json
from modulation_reference import Harper2023, SqueezeExcitation
from train_modulation12 import as_tensor, metrics
from torchsig_probe import CLASSES


def descriptor(a):
    if a.ndim == 3:
        return torch.cat((a.mean(2),a.var(2,correction=0)),1)
    if a.ndim != 2:
        raise ValueError('Unexpected activation shape')
    return a


def separation(features, labels):
    z=np.asarray(features,dtype=np.float64); labels=np.asarray(labels)
    if z.ndim != 2 or len(z)!=len(labels) or not np.isfinite(z).all():
        raise ValueError('Invalid descriptors')
    classes=np.arange(len(CLASSES)); counts=np.bincount(labels,minlength=len(classes))
    if np.any(counts<2): raise ValueError('At least two examples per class required')
    centroids=np.stack([z[labels==cid].mean(0) for cid in classes])
    center=z.mean(0)
    between=float(sum(counts[cid]*np.square(centroids[cid]-center).sum() for cid in classes))
    within=float(np.square(z-centroids[labels]).sum())
    predictions=[]
    for row,cid in zip(z,labels):
        candidates=centroids.copy()
        candidates[cid]=(counts[cid]*centroids[cid]-row)/(counts[cid]-1)
        predictions.append(int(np.square(candidates-row).sum(1).argmin()))
    return dict(dimensions=z.shape[1],between_scatter=between,within_scatter=within,
                between_within_ratio=between/within if within>0 else None,
                loo_centroid=metrics(labels,np.asarray(predictions)))


def inspect(model, x, labels, device):
    descriptors={}; stats={}; gates={}; handles=[]; predictions=[]
    def capture(name):
        def hook(module,inputs,output):
            a=output.detach()
            if not torch.isfinite(a).all(): raise ValueError('Nonfinite activation: '+name)
            descriptors.setdefault(name,[]).append(descriptor(a).cpu().numpy())
            item=stats.setdefault(name,dict(squared_sum=0.,zero_count=0,elements=0,shape=list(a.shape[1:])))
            item['squared_sum']+=float(a.double().square().sum())
            item['zero_count']+=int((a==0).sum()); item['elements']+=a.numel()
            if isinstance(module,SqueezeExcitation):
                # Recompute the exact gate expression; does not alter module output.
                gate=torch.sigmoid(module.expand(torch.relu(module.reduce(inputs[0].mean(2)))))
                gates.setdefault(name,[]).append(gate.detach().cpu().numpy())
        return hook
    for name,module in model.named_modules():
        if (name.startswith('features.') and name.count('.')==1) or name=='pool' or (name.startswith('head.') and name.count('.')==1):
            handles.append(module.register_forward_hook(capture(name)))
    try:
        model.eval()
        with torch.inference_mode():
            for start in range(0,len(x),8):
                predictions.extend(model(x[start:start+8].to(device)).argmax(1).cpu().tolist())
    finally:
        for handle in handles: handle.remove()
    layers={}
    arrays={}
    for name,parts in descriptors.items():
        z=np.concatenate(parts); arrays[name]=z
        item=stats[name]
        layers[name]=dict(shape=item['shape'],rms=(item['squared_sum']/item['elements'])**.5,
                          zero_fraction=item['zero_count']/item['elements'],**separation(z,labels))
        if name in gates:
            values=np.concatenate(gates[name]); arrays[name+'.gates']=values
            layers[name]['gate']=dict(minimum=float(values.min()),maximum=float(values.max()),
                p10=float(np.quantile(values,.1)),median=float(np.median(values)),p90=float(np.quantile(values,.9)),
                fraction_below_005=float((values<.05).mean()),fraction_above_095=float((values>.95).mean()))
    return dict(model_predictions=metrics(labels,np.asarray(predictions)),layers=layers),arrays


def choose_run(path):
    if path is not None: return path
    candidates=[p for p in Path('models').glob('training-lr-*')
                if (p/'LR_DIAGNOSTIC_COMPLETE').is_file() and (p/'report.json').is_file()]
    if len(candidates)!=1:
        raise ValueError('Bitte genau einen Lauf mit --run models/training-lr-ZEITSTEMPEL angeben')
    return candidates[0]


def run(args):
    source=choose_run(args.run)
    if not (source/'LR_DIAGNOSTIC_COMPLETE').is_file(): raise ValueError('Completed learning-rate run required')
    report=json.loads((source/'report.json').read_text()); protocol=report['protocol']
    if not protocol.get('default_design') or protocol['activation']!='tf-compatible':
        raise ValueError('Expected unchanged default learning-rate design')
    for name in ['modulation_reference.py','train_modulation12.py']:
        if sha(Path(__file__).with_name(name))!=protocol['sources'][name]:
            raise ValueError('Source mismatch: '+name)
    if sha(source/'selection.npz')!=protocol['selection_sha256']: raise ValueError('Selection hash mismatch')
    with np.load(source/'selection.npz',allow_pickle=False) as data:
        iq,labels,ids=(data[k] for k in ['iq','y','source_id'])
    if (iq.shape!=(120,4096) or labels.shape!=(120,) or not np.isfinite(iq).all()
        or not np.iscomplexobj(iq) or not np.issubdtype(labels.dtype,np.integer)
        or not np.array_equal(np.bincount(labels,minlength=12),np.full(12,10))
        or len(np.unique(ids))!=120):
        raise ValueError('Invalid 120-example selection')
    if args.device=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
    if args.output is None:
        args.output=source/('feature-diagnosis-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    if args.output.exists(): raise ValueError('Output exists')
    args.output.mkdir(parents=True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    plan=Path(__file__).with_name('FEATURE_DIAGNOSTIC_PLAN_V1.json')
    provenance=dict(source=str(source.resolve()),report_sha256=sha(source/'report.json'),
        selection_sha256=sha(source/'selection.npz'),plan=json.loads(plan.read_text()),plan_sha256=sha(plan),
        device=args.device,torch_version=torch.__version__,numpy_version=np.__version__,cuda=torch.version.cuda,
        script_sha256=sha(Path(__file__)),model_source_sha256=sha(Path(__file__).with_name('modulation_reference.py')),
        training_score_helper_sha256=sha(Path(__file__).with_name('train_modulation12.py')),
        helper_sources={name:sha(Path(__file__).with_name(name)) for name in
                        ['train_modulation12.py','check_modulation_learning.py',
                         'diagnose_training_collapse.py','build_modulation_dataset.py','torchsig_probe.py']},
        device_name=torch.cuda.get_device_name(0) if args.device=='cuda' else 'CPU')
    write_json(args.output/'protocol.json',provenance)
    x=as_tensor(iq); cases={}
    for case in ['initial','lr0.001','lr0.0001']:
        path=source/'initial_state.pt' if case=='initial' else source/case/'last_checkpoint.pt'
        if case!='initial' and sha(path)!=report['cases'][case]['checkpoint_sha256']:
            raise ValueError('Checkpoint hash mismatch: '+case)
        saved=torch.load(path,map_location='cpu',weights_only=True)
        state=saved if case=='initial' else saved['model']
        expected=protocol['initial_sha256'] if case=='initial' else report['cases'][case]['final_sha256']
        if state_digest(state)!=expected: raise ValueError('Model state mismatch: '+case)
        model=Harper2023(tensorflow_selu=True); model.load_state_dict(state); model.to(args.device)
        before=state_digest(model.state_dict())
        result,arrays=inspect(model,x,labels,args.device)
        if state_digest(model.state_dict())!=before: raise ValueError('Model was modified')
        result['checkpoint_sha256']=sha(path); result['state_sha256']=before
        if case!='initial':
            expected_matrix=report['cases'][case]['final']['training']['confusion_matrix']
            result['prediction_matrix_matches_training_report']=(result['model_predictions']['confusion_matrix']==expected_matrix)
        np.savez_compressed(args.output/(case+'-descriptors.npz'),**arrays)
        cases[case]=result
        write_json(args.output/'report.json',dict(protocol=provenance,cases=cases))
        print(case,': Modell-Wiedererkennung {:.1%}; {} Schichten'.format(result['model_predictions']['accuracy'],len(result['layers'])),flush=True)
        del model
    (args.output/'FEATURE_DIAGNOSTIC_COMPLETE').write_text('OK\n')
    print('Ergebnisse:',args.output.resolve())
    print('Deskriptive Trainingsdiagnose; kein Beweis fuer Informationsverlust oder Testgenauigkeit.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    try: run(parser.parse_args())
    except (ValueError,RuntimeError,OSError,KeyError,AssertionError) as error:
        parser.exit(1,'Merkmalsdiagnose fehlgeschlagen: '+str(error)+'\n')
