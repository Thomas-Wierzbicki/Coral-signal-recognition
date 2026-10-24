"""Direct parity against fixtures from unchanged original TF 2.4.1 code; CPU only."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from modulation_reference import Harper2023

ATOL=1e-6
RTOL=1e-4


def parameter_map():
    mapping={}
    for i in range(7):
        prefix='features.'+str(3*i)
        mapping[prefix+'.weight']='conv_'+str(i+1)+'__kernel'
        mapping[prefix+'.bias']='conv_'+str(i+1)+'__bias'
        if i<6:
            prefix='features.'+str(3*i+2)
            mapping[prefix+'.reduce.weight']='se_block_'+str(i+1)+'_dense_1__kernel'
            mapping[prefix+'.expand.weight']='se_block_'+str(i+1)+'_dense_2__kernel'
    for index,name in [(0,'dense_variant_1'),(2,'dense_variant_2'),(4,'variant')]:
        mapping['head.'+str(index)+'.weight']=name+'__kernel'
        mapping['head.'+str(index)+'.bias']=name+'__bias'
    return mapping


def to_torch_layout(array):
    if array.ndim==3: return array.transpose(2,1,0).copy()
    if array.ndim==2: return array.T.copy()
    return array.copy()


def check(name,actual,expected):
    a=np.asarray(actual);b=np.asarray(expected)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return dict(name=name,passed=False,reason='shape or nonfinite',actual_shape=list(a.shape),expected_shape=list(b.shape))
    error=np.abs(a.astype(np.float64)-b.astype(np.float64))
    tolerance=ATOL+RTOL*np.abs(b.astype(np.float64))
    return dict(name=name,passed=bool(np.all(error<=tolerance)),max_abs_error=float(error.max()),
        max_relative_error=float((error/np.maximum(np.abs(b),1e-30)).max()),
        max_tolerance_ratio=float((error/tolerance).max()))


def matched_first_adam_step(weight,grad):
    # A single-step equation check, not a new production optimizer.
    g=grad/torch.maximum(grad.norm(),torch.tensor(1.,dtype=grad.dtype))
    b1=torch.tensor(.9,dtype=g.dtype);b2=torch.tensor(.999,dtype=g.dtype)
    moment=(1-b1)*g;variance=(1-b2)*g.square()
    alpha=torch.tensor(.0001,dtype=g.dtype)*torch.sqrt(1-b2)/(1-b1)
    return weight-alpha*moment/(variance.sqrt()+torch.tensor(1e-7,dtype=g.dtype))


def case_check(folder,spec):
    path=folder/(spec['name']+'.npz')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']: raise ValueError('Fixture hash changed')
    with np.load(path,allow_pickle=False) as saved: arrays={name:saved[name] for name in saved.files}
    model=Harper2023(spec['classes']).eval();mapping=parameter_map()
    state=model.state_dict()
    if set(state)!=set(mapping): raise ValueError('Weight mapping incomplete')
    for name in state: state[name]=torch.from_numpy(to_torch_layout(arrays['weight__'+mapping[name]]))
    model.load_state_dict(state)
    actual={};hooks=[]
    def capture(name,temporal=False):
        def hook(module,inputs,output):
            value=output.detach().numpy().copy()
            actual[name]=value.transpose(0,2,1) if temporal else value
        return hook
    for i in range(7):
        hooks.append(model.features[3*i+1].register_forward_hook(capture('conv_'+str(i+1),True)))
        if i<6: hooks.append(model.features[3*i+2].register_forward_hook(capture('se_block_'+str(i+1)+'_mult',True)))
    hooks.append(model.pool.register_forward_hook(capture('stat_pool')))
    for i,name in [(0,'dense_variant_1'),(1,'act_variant_1'),(2,'dense_variant_2'),(3,'act_variant_2')]:
        hooks.append(model.head[i].register_forward_hook(capture(name)))
    x=torch.from_numpy(arrays['input'].transpose(0,2,1).copy()).unsqueeze(2)
    logits=model(x);labels=torch.from_numpy(arrays['labels'].astype(np.int64))
    loss=torch.nn.functional.cross_entropy(logits,labels);loss.backward()
    for h in hooks: h.remove()
    actual['gap']=actual['stat_pool'][:,:108];actual['gvp']=actual['stat_pool'][:,108:]
    actual['variant']=logits.detach().softmax(1).numpy()
    checks=[]
    for name,value in actual.items():
        for k,label in enumerate(['zero','sinusoid','noise']):
            checks.append(check('activation/'+name+'/'+label,value[k],arrays['activation__'+name][k]))
    checks.append(check('logits',logits.detach().numpy(),arrays['logits']))
    checks.append(check('loss',loss.detach().numpy(),arrays['loss']))
    native={str(eps):[] for eps in (1e-7,1e-8)}
    for name,p in model.named_parameters():
        expected_grad=to_torch_layout(arrays['gradient__'+mapping[name]])
        checks.append(check('gradient/'+name,p.grad.numpy(),expected_grad))
        start=p.detach().clone();g=torch.from_numpy(expected_grad)
        expected_after=to_torch_layout(arrays['after_step__'+mapping[name]])
        checks.append(check('matched_first_adam_step/'+name,matched_first_adam_step(start,g).numpy(),expected_after))
        # Same exported gradient and per-tensor clipping isolate optimizer arithmetic.
        for eps in (1e-7,1e-8):
            param=torch.nn.Parameter(start.clone());param.grad=g.clone()/torch.maximum(g.norm(),torch.tensor(1.))
            optimizer=torch.optim.Adam([param],lr=.0001,eps=eps);optimizer.step()
            native[str(eps)].append(check(name,param.detach().numpy(),expected_after))
    failed=[c for c in checks if not c['passed']]
    report=dict(**spec,passed=not failed,checks=len(checks),failures=failed,
        max_abs_error=max(c.get('max_abs_error',float('inf')) for c in checks),
        max_tolerance_ratio=max(c.get('max_tolerance_ratio',float('inf')) for c in checks),
        native_adam={eps:dict(arrays=len(values),arrays_failing_same_tolerance=sum(not v['passed'] for v in values),
            max_abs_difference=max(v.get('max_abs_error',float('inf')) for v in values),
            note='Same learning rate and exported gradients; this is an optimizer difference check, not a required equality gate')
            for eps,values in native.items()})
    return report,checks


def main(tf_python,output,fixture_dir=None):
    output=Path(output)
    if output.exists(): raise ValueError('Output exists; choose another --output')
    output.mkdir(parents=True)
    protocol=json.loads((Path(__file__).parent/'EXPERIMENT_PROTOCOL_V1.json').read_text())
    if protocol['reference_commit']!='902cca45c3bf04b5da9f78030372b5a03f730f42': raise ValueError('Unexpected reference')
    if fixture_dir is None:
        fixture_dir=output/'tf-fixtures'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',TF_CPP_MIN_LOG_LEVEL='2')
        subprocess.run([str(tf_python),str(Path(__file__).parent/'tools/export_harper_tf241.py'),
                        '--output',str(fixture_dir)],check=True,env=env)
    fixture_dir=Path(fixture_dir)
    exported=json.loads((fixture_dir/'export.json').read_text())
    if exported['tensorflow']!='2.4.1' or exported['numpy']!='1.19.2': raise ValueError('Unexpected TF environment')
    if exported['upstream_commit']!=protocol['reference_commit']: raise ValueError('Unexpected original source')
    expected={(n,c,s) for n,c in [(1024,24),(4096,12)] for s in [901,902,903]}
    if {(x['samples'],x['classes'],x['seed']) for x in exported['cases']}!=expected or len(exported['cases'])!=6:
        raise ValueError('Incomplete or altered fixture plan')
    reports=[];details={}
    for spec in exported['cases']:
        report,checks=case_check(fixture_dir,spec);reports.append(report);details[spec['name']]=checks
        print(spec['name'], 'PASS' if report['passed'] else 'FAIL', 'max abs',report['max_abs_error'],flush=True)
    report=dict(schema=1,protocol_id=protocol['protocol_id'],tensorflow_export=exported,
        pytorch_version=torch.__version__,device='CPU',tolerance=dict(atol=ATOL,rtol=RTOL),cases=reports,
        all_passed=all(c['passed'] for c in reports),
        scope='Direct transferred-weight technical parity and single-step equation comparison; no classification accuracy, no full historical training',
        source_sha256={p:hashlib.sha256((Path(__file__).parent/p).read_bytes()).hexdigest()
            for p in ['modulation_reference.py','verify_reference_parity.py','tools/export_harper_tf241.py']})
    (output/'parity_report.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'details.json').write_text(json.dumps(details,indent=2)+'\n')
    if not report['all_passed']: raise ValueError('Parity gate failed; inspect details; tolerance was not relaxed')
    (output/'PARITY_COMPLETE').write_text('OK\n')
    print('Technical parity PASSED. No RF recognition performance established.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tf-python',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fixture-dir',type=Path)
    args=p.parse_args()
    try: main(args.tf_python,args.output,args.fixture_dir)
    except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as e:p.exit(1,'Parity check failed: '+str(e)+'\n')
