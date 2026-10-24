"""Post-hoc causal probe only: replay TF Conv5's backward mask, preserve PT forward."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from modulation_reference import Harper2023
from verify_reference_parity import parameter_map,to_torch_layout,check

class ReplayBackwardMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,mask):
        if x.shape!=mask.shape: raise ValueError('Mask shape mismatch')
        ctx.save_for_backward(mask)
        return torch.relu(x)
    @staticmethod
    def backward(ctx,gradient):
        return gradient*ctx.saved_tensors[0],None

class MaskProbe(torch.nn.Module):
    def __init__(self,mask):
        super().__init__();self.register_buffer('mask',mask)
    def forward(self,x):return ReplayBackwardMask.apply(x,self.mask)


def run(tf_export,output):
    folder=Path(tf_export);output=Path(output)
    if output.exists():raise ValueError('Output exists')
    export=json.loads((folder/'localization_export.json').read_text())
    spec=next(c for c in export['cases'] if c['name']=='samples4096-classes12-seed903' and c['dtype']=='float32')
    path=folder/(spec['diagnostic_name']+'.npz')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['diagnostic_sha256']:raise ValueError('Fixture changed')
    with np.load(path,allow_pickle=False) as saved:a={k:saved[k] for k in saved.files}
    torch.backends.mkldnn.enabled=True
    mapping=parameter_map();state={n:torch.from_numpy(to_torch_layout(a['weight__'+k])) for n,k in mapping.items()}
    x=torch.from_numpy(a['input'].transpose(0,2,1).copy()).unsqueeze(2);labels=torch.from_numpy(a['labels'])
    rows=[];baseline_logits=None
    for replay in (False,True):
        model=Harper2023(12,tensorflow_selu=True);model.load_state_dict(state);model.eval()
        if replay:
            mask=torch.from_numpy((a['activation__conv_5']>0).transpose(0,2,1).copy())
            model.features[13]=MaskProbe(mask)
        logits=model(x);loss=torch.nn.functional.cross_entropy(logits,labels);loss.backward()
        checks=[check(n,p.grad.numpy(),to_torch_layout(a['gfull__'+mapping[n]])) for n,p in model.named_parameters()]
        if baseline_logits is None:baseline_logits=logits.detach().clone()
        rows.append(dict(counterfactual_mask_replay=replay,
            logits_bit_identical_to_baseline=bool(torch.equal(logits.detach(),baseline_logits)),
            parameter_gradient_checks=len(checks),failures=[c for c in checks if not c['passed']],
            conv5_bias=next(c for c in checks if c['name']=='features.12.bias')))
    report=dict(scope='Post-hoc diagnostic intervention, not a new production activation or gate pass',cases=rows,
                original_float32_gate_overridden=False)
    output.mkdir(parents=True);(output/'mask_replay_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tf-export',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.tf_export,args.output)
