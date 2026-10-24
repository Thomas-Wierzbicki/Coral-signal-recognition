"""Controlled TorchSig 2.2 FSK diagnosis; preserves existing data and models."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import importlib.metadata
import numpy as np
from torchsig.signals.builders import fsk

LABELS=['2fsk','2gfsk','2gmsk']
RATE=32000
N=4096

def features(iq):
    step=np.angle(iq[1:]*iq[:-1].conj())
    step=step-step.mean()
    scale=max(float(np.std(step)),1e-9)
    z=step/scale
    return np.r_[scale,np.quantile(z,[.01,.1,.25,.5,.75,.9,.99]),
        [np.mean(z[k:]*z[:-k]) for k in (1,2,4,8,16,32,64)],np.std(abs(iq))/np.mean(abs(iq))]

def matrix(y,p):
    a=np.zeros((3,3),dtype=int); np.add.at(a,(y,p),1); return a.tolist()

def run(output,per_class=120,reference_model=None):
    if importlib.metadata.version('torchsig')!='2.2.0':
        raise ValueError('Dieser versionsgebundene Test erwartet TorchSig 2.2.0')
    output=Path(output)
    if output.exists(): raise ValueError('Ausgabe existiert; neuen --output verwenden')
    if per_class<12: raise ValueError('Mindestens 12 Beispiele je Klasse erforderlich')
    output.mkdir(parents=True)
    rows=[]; arrays=[]; ys=[]
    for cid,label in enumerate(LABELS):
        for k in range(per_class):
            seed=[2060,cid,k]
            # Preview the parameter draw with a separate identically seeded RNG.
            params=fsk.fsk_random_params(label[1:],np.random.default_rng(seed))
            clean=fsk.fsk_modulator(2,label[1:],6000,RATE,16384,np.random.default_rng(seed))[6144:10240]
            clean=clean/np.sqrt(np.mean(abs(clean)**2))
            rng=np.random.default_rng([2061,cid,k])
            noise=(rng.normal(size=N)+1j*rng.normal(size=N))*np.sqrt(.001/2)
            iq=clean+noise; iq=(.5*iq/np.sqrt(np.mean(abs(iq)**2))).astype(np.complex64)
            arrays.append(iq); ys.append(cid)
            rows.append(dict(label=label,index=k,seed=seed,h=float(params[0]),bt=params[1],gaussian_span=params[2]))
        print(label,'fertig',flush=True)
    iq=np.array(arrays); y=np.array(ys); x=np.array([features(a) for a in iq])
    train=np.tile(np.arange(per_class)<(2*per_class//3),3); test=~train
    mean=x[train].mean(0); std=np.maximum(x[train].std(0),1e-6); z=(x-mean)/std
    distances=((z[test,None]-z[None,train])**2).sum(-1)
    neighbors=np.argsort(distances,axis=1)[:,:5]
    pred=np.array([np.bincount(y[train][a],minlength=3).argmax() for a in neighbors])
    # Same h/BT/span, same random symbols: labels alone do not alter the waveform.
    a=fsk.fsk_modulator_baseband(2,'gfsk',16384,4,np.random.default_rng(9),params=(.5,.3,3))
    b=fsk.fsk_modulator_baseband(2,'gmsk',16384,4,np.random.default_rng(9),params=(.5,.3,3))
    identical=bool(np.array_equal(a,b))
    report=dict(classes=LABELS,conditions=dict(sample_rate_hz=RATE,samples=N,requested_bandwidth_hz=6000,
        snr_db=30,frequency_offset_hz=0,phase_rad=0),
        train_per_class=int(train.sum()//3),test_per_class=int(test.sum()//3),
        feature_knn=dict(accuracy=float(np.mean(y[test]==pred)),matrix=matrix(y[test],pred),neighbors=5,
            features='centered phase-step scale, quantiles, normalized autocorrelation; envelope CV'),
        equal_parameter_boundary=dict(h=.5,bt=.3,span=3,bit_identical=identical,
            note='GFSK h draw is [0.1,0.5), so this exact boundary is not an ordinary sampled GFSK example.'),
        generator_source_sha256=hashlib.sha256(inspect.getsource(fsk).encode()).hexdigest(),
        limitation='Synthetic generator diagnosis. Fixed requested bandwidth does not fix symbol rate or deviation. Feature comparison is not an isolated architecture test or real-world accuracy.')
    if reference_model:
        import torch
        from train_modulation12 import as_tensor, predict
        model=torch.jit.load(str(reference_model),map_location='cpu').eval()
        full=predict(model,as_tensor(iq[test]),torch.device('cpu'),32)
        from torchsig_probe import CLASSES
        m=np.zeros((3,len(CLASSES)),dtype=int); np.add.at(m,(y[test],full),1)
        report['frozen_reference']=dict(classes=CLASSES,matrix=m.tolist(),
            accuracy=float(np.mean(full==np.array([CLASSES.index(LABELS[c]) for c in y[test]]))),
            model_sha256=hashlib.sha256(Path(reference_model).read_bytes()).hexdigest())
    np.savez_compressed(output/'examples.npz',iq=iq,y=y,train_mask=train,features=x)
    (output/'parameters.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,8),layout='constrained')
    for cid,label in enumerate(LABELS):
        signal=iq[cid*per_class]; step=np.angle(signal[1:]*signal[:-1].conj())*RATE/(2*np.pi)
        axes[cid,0].plot(np.arange(512)/RATE,step[:512]); axes[cid,0].set_ylabel(label+' Hz')
        axes[cid,1].hist(step,bins=60); axes[cid,1].set_xlabel('phase-step frequency, Hz')
    axes[-1,0].set_xlabel('seconds'); fig.savefig(output/'phase_steps.png',dpi=140); plt.close(fig)
    (output/'DIAGNOSIS_COMPLETE').write_text('OK\n')
    print('GFSK/GMSK bei identischen Grenzparametern identisch:',identical)
    print('Merkmalsvergleich, eigener synthetischer Pruefsatz: {:.1%}'.format(report['feature_knn']['accuracy']))
    print(np.array(report['feature_knn']['matrix']))
    if reference_model: print('Eingefrorenes 12-Klassen-Modell: {:.1%}'.format(report['frozen_reference']['accuracy']))
    print('Ergebnisse:',output.resolve())
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('data/fsk-separability-v1'))
    p.add_argument('--per-class',type=int,default=120)
    p.add_argument('--reference-model',type=Path)
    args=p.parse_args()
    run(args.output,args.per_class,args.reference_model)
