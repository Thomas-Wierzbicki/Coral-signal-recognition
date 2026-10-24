"""Mechanism diagnosis for CPU gradient parity; does not replace the FP32 gate."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from modulation_reference import Harper2023
from verify_reference_parity import parameter_map,to_torch_layout,check


def bias_vjp(cotangent,channels_first):
    q=torch.from_numpy(np.ascontiguousarray(cotangent))
    if channels_first: q=q.permute(0,2,1).contiguous()
    bias=torch.zeros(cotangent.shape[-1],dtype=q.dtype,requires_grad=True)
    shape=(1,-1,1) if channels_first else (1,1,-1)
    objective=((torch.zeros_like(q)+bias.reshape(shape))*q).sum()
    return torch.autograd.grad(objective,bias)[0].numpy()


def decompose_bias(tf_gradient,pt_gradient,tf_upstream,pt_upstream,tf_mask,pt_mask):
    t=tf_upstream.astype(np.float64);p=pt_upstream.astype(np.float64)
    mt=tf_mask.astype(np.float64);mp=pt_mask.astype(np.float64)
    sum_tf=(t*mt).sum(axis=(0,1));sum_pt=(p*mp).sum(axis=(0,1))
    mask=(t*(mp-mt)).sum(axis=(0,1))
    upstream=((p-t)*mp).sum(axis=(0,1))
    tf_reduction=sum_tf-tf_gradient
    pt_reduction=pt_gradient-sum_pt
    actual=pt_gradient.astype(np.float64)-tf_gradient.astype(np.float64)
    residual=actual-(mask+upstream+tf_reduction+pt_reduction)
    index=int(np.argmax(np.abs(actual)))
    def entry(a): return dict(max_abs=float(np.max(np.abs(a))),at_worst_channel=float(a[index]))
    return dict(worst_channel=index,actual_difference=entry(actual),mask_change_term=entry(mask),
        upstream_change_term=entry(upstream),tf_reduction_term=entry(tf_reduction),
        pt_reduction_term=entry(pt_reduction),decomposition_residual_max_abs=float(np.max(np.abs(residual))))


def analyze(folder,spec,mkldnn):
    path=folder/(spec['diagnostic_name']+'.npz')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['diagnostic_sha256']: raise ValueError('Diagnostic fixture changed')
    with np.load(path,allow_pickle=False) as saved:a={k:saved[k] for k in saved.files}
    torch.backends.mkldnn.enabled=mkldnn
    model=Harper2023(spec['classes'],tensorflow_selu=True)
    if spec['dtype']=='float64':model=model.double()
    mapping=parameter_map()
    model.load_state_dict({n:torch.from_numpy(to_torch_layout(a['weight__'+k])) for n,k in mapping.items()})
    model.eval();activations={};handles=[]
    for i in range(7):
        def hook(mod,inp,out,i=i): activations[i+1]=out
        handles.append(model.features[3*i+1].register_forward_hook(hook))
    x=torch.from_numpy(a['input'].transpose(0,2,1).copy()).unsqueeze(2)
    logits=model(x);labels=torch.from_numpy(a['labels'].astype(np.int64))
    per=torch.nn.functional.cross_entropy(logits,labels,reduction='none');loss=per.mean()
    parameters=list(model.parameters());names=list(dict(model.named_parameters()))
    full=torch.autograd.grad(loss,parameters,retain_graph=True)
    gradients=dict(zip(names,full));rows=[]
    for i in range(3):rows.append(dict(zip(names,torch.autograd.grad(per[i]/3,parameters,retain_graph=True))))
    checks=[check('logits',logits.detach().numpy(),a['logits']),check('loss',loss.detach().numpy(),a['loss'])]
    for name,g in gradients.items():
        checks.append(check('gradient/'+name,g.numpy(),to_torch_layout(a['gfull__'+mapping[name]])))
    row_checks=[]
    for row in range(3):
        for name,g in rows[row].items():
            row_checks.append(check('row'+str(row)+'/'+name,g.numpy(),to_torch_layout(a['grow'+str(row)+'__'+mapping[name]])))
    masks=[]
    for layer,tensor in activations.items():
        v=tensor.detach().numpy().transpose(0,2,1);expected=a['activation__conv_'+str(layer)]
        mask=(v>0)!=(expected>0)
        masks.append(dict(layer=layer,different_relu_decisions=int(mask.sum()),
            largest_tf_magnitude_at_difference=float(np.abs(expected[mask]).max()) if mask.any() else 0))
        checks.append(check('activation/conv_'+str(layer),v,expected))
    diagnostics=[]
    for layer in (5,7):
        name='conv_'+str(layer)
        tf_up=a['cotangent__'+name]
        pt_up=torch.autograd.grad(loss,activations[layer],retain_graph=True)[0].numpy().transpose(0,2,1)
        tf_mask=a['activation__'+name]>0;pt_mask=activations[layer].detach().numpy().transpose(0,2,1)>0
        tf_g=a['gfull__'+name+'__bias'];pt_g=gradients['features.'+str((layer-1)*3)+'.bias'].numpy()
        q=a['preactivation_cotangent__'+name]
        gold=q.astype(np.float64).sum(axis=(0,1))
        tf_vjp=a['bias_vjp__'+name];pt_vjp=bias_vjp(q,True);pt_last=bias_vjp(q,False)
        row_errors=[]
        for i,label in enumerate(['zero','sinusoid','noise']):
            tf_row=a['grow'+str(i)+'__'+name+'__bias']
            pt_row=rows[i]['features.'+str((layer-1)*3)+'.bias'].numpy()
            row_errors.append(dict(input=label,**check('bias',pt_row,tf_row)))
        diagnostics.append(dict(layer=layer,decomposition=decompose_bias(tf_g,pt_g,tf_up,pt_up,tf_mask,pt_mask),
            per_input_bias_checks=row_errors,
            isolated_reduction=dict(tf_vjp_vs_actual_model_max_abs=float(np.abs(tf_vjp-tf_g).max()),
                tf_vjp_vs_same_q_float64_sum_max_abs=float(np.abs(tf_vjp-gold).max()),
                pt_channels_first_vs_same_q_float64_sum_max_abs=float(np.abs(pt_vjp-gold).max()),
                pt_channels_last_vs_same_q_float64_sum_max_abs=float(np.abs(pt_last-gold).max()),
                tf_vjp_vs_pt_channels_first=check('isolated_bias_vjp',pt_vjp,tf_vjp)),
            per_input_gradient_sum=dict(
                tf_vs_full_max_abs=float(np.abs(sum(a['grow'+str(i)+'__'+name+'__bias'].astype(np.float64) for i in range(3))-tf_g).max()),
                pt_vs_full_max_abs=float(np.abs(sum(rows[i]['features.'+str((layer-1)*3)+'.bias'].numpy().astype(np.float64) for i in range(3))-pt_g).max()))))
    for h in handles:h.remove()
    result=dict(name=spec['name'],dtype=spec['dtype'],mkldnn_enabled=mkldnn,
        full_checks=len(checks),full_passed=all(c['passed'] for c in checks),
        full_failures=[c for c in checks if not c['passed']],
        per_input_gradient_failures=[c for c in row_checks if not c['passed']],
        relu_masks=masks,bias_layers=diagnostics)
    return result


def main(folder,output):
    folder=Path(folder);output=Path(output)
    if output.exists():raise ValueError('Output exists')
    exported=json.loads((folder/'localization_export.json').read_text())
    if exported['tensorflow']!='2.4.1':raise ValueError('Wrong reference')
    expected={(n,c,s,d) for n,c in [(1024,24),(4096,12)] for s in [901,902,903] for d in ['float32','float64']}
    actual={(c['samples'],c['classes'],c['seed'],c['dtype']) for c in exported['cases']}
    if actual!=expected or len(exported['cases'])!=12:raise ValueError('Changed or incomplete plan')
    output.mkdir(parents=True)
    results=[]
    for spec in exported['cases']:
        for backend in (True,False):
            result=analyze(folder,spec,backend);results.append(result)
            print(result['name'],spec['dtype'],'oneDNN-enabled' if backend else 'native',
                  'PASS' if result['full_passed'] else 'FAIL', 'row gradient failures',len(result['per_input_gradient_failures']),flush=True)
            report=dict(schema=1,plan='GRADIENT_LOCALIZATION_PLAN.json',cases=results,
                        scope='Mechanism diagnosis only; previous FP32 gates remain unchanged',
                        torch_version=torch.__version__,threads=torch.get_num_threads(),
                        source_sha256={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest()
                            for n in ['localize_reference_gradients.py','tools/export_harper_gradient_localization.py','modulation_reference.py']})
            (output/'localization_report.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'LOCALIZATION_COMPLETE').write_text('OK\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tf-export',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();main(args.tf_export,args.output)
