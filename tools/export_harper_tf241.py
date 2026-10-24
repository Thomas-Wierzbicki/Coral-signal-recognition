"""Export fixtures from the unchanged original TensorFlow 2.4.1 model (Python 3.8)."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import numpy as np
import tensorflow as tf

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'third_party/harper2023_original'


def key(name): return name.replace(':0','').replace('/','__')


def verify_source():
    manifest=json.loads((SOURCE/'SOURCE_MANIFEST.json').read_text())
    for f in manifest['files']:
        b=(SOURCE/f['path']).read_bytes()
        if hashlib.sha256(b).hexdigest()!=f['sha256']: raise ValueError('Changed original source: '+f['path'])
    return manifest['commit']


def fixture(seed,length):
    rng=np.random.default_rng(seed)
    n=np.arange(length,dtype=np.float64)
    phase=2*np.pi*(.073*n+.000002*n*n)
    sine=(1+.2*np.sin(2*np.pi*.003*n))*np.exp(1j*phase)
    noise=rng.normal(size=length)+1j*rng.normal(size=length)
    iq=np.stack([np.zeros(length,dtype=complex),sine,noise])
    return np.stack([iq.real,iq.imag],axis=-1).astype(np.float32)


def main(output):
    if tf.__version__!='2.4.1' or np.__version__!='1.19.2': raise ValueError('Requires TF 2.4.1 and NumPy 1.19.2')
    commit=verify_source()
    sys.path[:0]=[str(SOURCE/'experiments/radioml2018.01a'),str(SOURCE)]
    import models,run
    spec=importlib.util.spec_from_file_location('original_sweep',SOURCE/'experiments/radioml2018.01a/configs/sweep-models.py')
    sweep=importlib.util.module_from_spec(spec);spec.loader.exec_module(sweep)
    cfg=sweep.get_config()
    missing_version='version' not in cfg
    from ml_collections import ConfigDict
    expanded=cfg.to_dict()
    expanded.update(dict(use_se=True,use_act=True,residual=False,dilate=True,self_attention=False))
    cfg=ConfigDict(expanded)
    # Metadata-only launcher addition, not a modification of upstream model/training code.
    cfg.version='reference-validation-v1'
    run.get_name(cfg)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cases=[]
    for length,classes in [(1024,24),(4096,12)]:
        for seed in [901,902,903]:
            tf.keras.backend.clear_session();tf.random.set_seed(seed)
            model=models.create_model(input_shape=(length,2),n_classes=classes)
            expected=202880 if classes==24 else 201332
            if model.count_params()!=expected: raise ValueError('Parameter count mismatch')
            names=[]
            for i in range(1,8):
                names.append('conv_'+str(i))
                if i<7: names.append('se_block_'+str(i)+'_mult')
            names+=['gap','gvp','stat_pool','dense_variant_1','act_variant_1','dense_variant_2','act_variant_2','variant']
            probe=tf.keras.Model(model.inputs,[model.get_layer(name).output for name in names])
            x=fixture(seed,length);labels=np.array([0,classes//2,classes-1],dtype=np.int64)
            last=model.get_layer('variant')
            with tf.GradientTape() as tape:
                values=probe(x,training=False)
                logits=tf.matmul(values[names.index('act_variant_2')],last.kernel)+last.bias
                loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits))
            grads=tape.gradient(loss,model.trainable_variables)
            arrays={'input':x,'labels':labels,'logits':logits.numpy(),'loss':np.array(loss.numpy())}
            for name,value in zip(names,values): arrays['activation__'+name]=value.numpy()
            for var,grad in zip(model.trainable_variables,grads):
                if grad is None or not np.isfinite(grad.numpy()).all(): raise ValueError('Invalid gradient: '+var.name)
                arrays['weight__'+key(var.name)]=var.numpy()
                arrays['gradient__'+key(var.name)]=grad.numpy()
            optimizer=tf.keras.optimizers.Adam(lr=1e-4,clipnorm=1)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))
            for var in model.trainable_variables: arrays['after_step__'+key(var.name)]=var.numpy()
            name='samples{}-classes{}-seed{}'.format(length,classes,seed)
            np.savez_compressed(output/(name+'.npz'),**arrays)
            cases.append(dict(name=name,samples=length,classes=classes,seed=seed,parameters=model.count_params(),
                              sha256=hashlib.sha256((output/(name+'.npz')).read_bytes()).hexdigest()))
            print('TF original:',name,flush=True)
    report=dict(python=platform.python_version(),tensorflow=tf.__version__,numpy=np.__version__,device='CPU',
        upstream_commit=commit,cases=cases,fixtures=['zero','chirped amplitude-varying sinusoid','complex Gaussian noise'],
        loss='mean sparse softmax cross entropy of logits, labels [0,classes//2,classes-1]',
        optimizer=dict(name='original tf.keras.optimizers.Adam',lr=.0001,clipnorm=1,epsilon=1e-7,steps=1),
        original_runner_import='passed',config_version_missing_upstream=missing_version,
        explicit_metadata_addition="config.version='reference-validation-v1'; get_name checked; no full run started",
        limitations='CPU technical fixtures only, no RadioML training or classification performance reproduction')
    (output/'export.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
