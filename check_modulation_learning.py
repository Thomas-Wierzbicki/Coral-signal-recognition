"""Training-only memorization check of all 12 classes; not a recognition benchmark."""
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from train_modulation12 import as_tensor, load_metadata, load_split, make_model, metrics
from torchsig_probe import CLASSES
from build_modulation_dataset import sha


def select_examples(iq,y,ids,snr,per_class=2,seed=2070):
    rng=np.random.default_rng(seed); selected=[]
    for cid,label in enumerate(CLASSES):
        candidates=np.flatnonzero((y==cid)&(np.isnan(snr) if label=='noise' else snr>=15))
        if len(candidates)<per_class: raise ValueError('Zu wenige Trainingsbeispiele >=15 dB: '+label)
        selected.extend(rng.choice(candidates,per_class,replace=False).tolist())
    selected=np.array(selected)
    return iq[selected],y[selected],ids[selected],snr[selected]


def state_digest(state):
    digest=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        digest.update(name.encode()); digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def evaluate(model,x,y):
    model.eval()
    with torch.inference_mode():
        logits=model(x)
        return dict(loss=float(nn.functional.cross_entropy(logits,y)),
                    accuracy=float((logits.argmax(1)==y).float().mean()),
                    predicted_counts=torch.bincount(logits.argmax(1),minlength=len(CLASSES)).cpu().tolist(),
                    logit_std_between_examples=float(logits.std(dim=0,correction=0).mean()))


def fit_case(model,x,y,steps,lr,log_every,output):
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,eps=1e-8)
    initial=evaluate(model,x,y); rows=[]; latest={}; handles=[]
    def capture(name):
        def hook(module,inputs,result):
            a=result.detach()
            latest[name]=dict(rms=float(a.square().mean().sqrt()),
                std=float(a.std(correction=0)),zero_fraction=float((a==0).float().mean()))
        return hook
    for name,module in model.named_modules():
        if isinstance(module,(nn.Conv1d,nn.Conv2d,nn.Linear)) or name=='pool':
            handles.append(module.register_forward_hook(capture(name)))
    start=time.perf_counter()
    try:
        for step in range(1,steps+1):
            recording=(step==1 or step%log_every==0 or step==steps)
            # Hooks only sample the designated optimization steps.
            if not recording:
                for h in handles: h.remove()
                handles=[]
            elif not handles:
                for name,module in model.named_modules():
                    if isinstance(module,(nn.Conv1d,nn.Conv2d,nn.Linear)) or name=='pool':
                        handles.append(module.register_forward_hook(capture(name)))
            latest.clear(); model.train(); optimizer.zero_grad(set_to_none=True)
            logits=model(x); loss=nn.functional.cross_entropy(logits,y)
            if not torch.isfinite(loss): raise ValueError('Nichtendlicher Verlust')
            loss.backward()
            before={}; parameters={}
            for name,p in model.named_parameters():
                if p.grad is None or not torch.isfinite(p.grad).all(): raise ValueError('Fehlender/nichtendlicher Gradient: '+name)
                if recording:
                    before[name]=p.detach().clone()
                    parameters[name]=dict(gradient_l2=float(p.grad.norm()),weight_l2=float(p.detach().norm()))
            activations=copy.deepcopy(latest)
            optimizer.step()
            if recording:
                for name,p in model.named_parameters():
                    change=float((p.detach()-before[name]).norm())
                    parameters[name]['update_l2']=change
                    norm=parameters[name]['weight_l2']
                    parameters[name]['relative_update']=change/norm if norm>0 else None
                score=evaluate(model,x,y)
                row=dict(step=step,optimization_loss=float(loss.detach()),memorization=score,
                         activations_before_update=activations,parameters=parameters)
                rows.append(row)
                (output/'history.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
                print('Schritt {}: Verlust {:.4f}; Wiedererkennung {:.1%}'.format(step,score['loss'],score['accuracy']),flush=True)
                if score['accuracy']>=.99 and score['loss']<=.1: break
        score=evaluate(model,x,y)
        with torch.inference_mode(): pred=model(x).argmax(1).cpu().numpy()
        return dict(initial=initial,final=score,steps=step,seconds=time.perf_counter()-start,
                    memorization_pass=score['accuracy']>=.99 and score['loss']<=.1,
                    matrix=metrics(y.cpu().numpy(),pred)['confusion_matrix'])
    finally:
        for handle in handles: handle.remove()


def run(data,output=None,steps=400,per_class=2,device='cuda',seed=2070,log_every=25):
    if min(steps,per_class,log_every)<1 or seed<0: raise ValueError('Ungueltige Parameter')
    if device=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA nicht verfuegbar')
    data=Path(data); metadata=load_metadata(data)
    # No validation/test arrays are opened by this tool.
    iq,y,ids,snr=load_split(data,'train',metadata,set())
    iq,y,ids,snr=select_examples(iq,y,ids,snr,per_class,seed)
    if output is None: output=Path('models')/('modulation12-learning-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    output=Path(output)
    if output.exists(): raise ValueError('Ausgabe existiert; neuen --output waehlen')
    output.mkdir(parents=True)
    np.savez_compressed(output/'training_selection.npz',iq=iq,y=y,source_id=ids,snr_db=snr)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    initial_reference=copy.deepcopy(make_model('harper2023').state_dict())
    protocol=dict(classes=CLASSES,source_split='train only; validation/test not opened',per_class=per_class,
        examples=len(y),selection_seed=seed,signal_snr_min_db=15,noise_snr=None,steps=steps,
        batch='one fixed balanced full batch',log_every=log_every,device=device,
        torch_version=torch.__version__,numpy_version=np.__version__,cuda_version=torch.version.cuda,
        device_name=torch.cuda.get_device_name(0) if device=='cuda' else 'CPU',
        optimizer='PyTorch Adam eps=1e-8; no clipping/scheduler/augmentation',
        pass_rule='same selected training examples: accuracy >=99% AND loss <=0.1; not generalization',
        dataset_metadata_sha256=sha(data/'metadata.json'),selection_sha256=sha(output/'training_selection.npz'),
        reference_initial_sha256=state_digest(initial_reference),
        cases=['legacy_lr0.001','harper2023_lr0.001','harper2023_lr0.0001'],
        script_sha256=sha(Path(__file__)),model_source_sha256=sha(Path(__file__).with_name('modulation_reference.py')),
        training_helpers_sha256=sha(Path(__file__).with_name('train_modulation12.py')))
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    x=as_tensor(iq).to(device); target=torch.from_numpy(y).to(device); results={}
    for architecture,lr in [('legacy',.001),('harper2023',.001),('harper2023',.0001)]:
        name=architecture+'_lr'+str(lr); folder=output/name; folder.mkdir()
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        model=make_model(architecture)
        if architecture=='harper2023': model.load_state_dict(initial_reference)
        initial_hash=state_digest(model.state_dict())
        if architecture=='harper2023' and initial_hash!=protocol['reference_initial_sha256']:
            raise ValueError('Anfangsgewichte stimmen nicht ueberein')
        model=model.to(device)
        print('\nLerntest:',name,'|',len(y),'feste Trainingsbeispiele, alle 12 Klassen',flush=True)
        result=fit_case(model,x,target,steps,lr,log_every,folder)
        result.update(architecture=architecture,learning_rate=lr,initial_sha256=initial_hash,
                      final_sha256=state_digest(model.state_dict()))
        results[name]=result
        (output/'report.json').write_text(json.dumps(dict(protocol=protocol,cases=results),indent=2),encoding='utf-8')
        del model
    (output/'LEARNING_CHECK_COMPLETE').write_text('OK\n')
    for name,result in results.items():
        print('{}: {:.1%}, Verlust {:.4f}, Lerntest {}'.format(name,result['final']['accuracy'],result['final']['loss'],
            'bestanden' if result['memorization_pass'] else 'nicht bestanden'))
    print('Ergebnisse:',output.resolve())
    print('Wiedererkennung derselben Trainingsbeispiele; KEINE Testgenauigkeit.')
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/modulation12-v1')); p.add_argument('--output',type=Path)
    p.add_argument('--steps',type=int,default=400); p.add_argument('--per-class',type=int,default=2)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda'); p.add_argument('--seed',type=int,default=2070)
    p.add_argument('--log-every',type=int,default=25); args=p.parse_args()
    try: run(args.data,args.output,args.steps,args.per_class,args.device,args.seed,args.log_every)
    except (OSError,ValueError,KeyError,RuntimeError,AssertionError) as error:
        p.exit(1,'Lerntest fehlgeschlagen: '+str(error)+'\n')
