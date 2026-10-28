"""Explicit four-channel feature treatment; historical model code stays unchanged."""
import numpy as np
import torch
from torch import nn
from se_gain_experiment import make_gain_model

ARMS=('iq_zero','iq_envelope_phase')
ARCHITECTURE='Harper2023-se2-four-channel-feature-v1'
PREPROCESSING='iq-envelope-phase-f64-to-f32-v1'


def feature_tensor(iq,arm):
    if arm not in ARMS:
        raise ValueError('Unknown feature arm')
    iq=np.asarray(iq)
    if iq.ndim!=2 or iq.shape[1]<2 or not np.iscomplexobj(iq) or not np.isfinite(iq).all():
        raise ValueError('Invalid IQ array')
    result=np.zeros((len(iq),4,1,iq.shape[1]),dtype=np.float32)
    result[:,0,0]=iq.real;result[:,1,0]=iq.imag
    if arm=='iq_envelope_phase':
        z=iq.astype(np.complex128)
        result[:,2,0]=np.abs(z)
        product=z[:,1:]*z[:,:-1].conj()
        phase=np.angle(product)/np.pi
        phase[np.abs(product)==0]=0.
        result[:,3,0,1:]=phase
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite features')
    return torch.from_numpy(result)


def make_feature_model(iq_initial_state):
    model=make_gain_model(2.,tensorflow_selu=True)
    model.load_state_dict(iq_initial_state)
    old=model.features[0]
    expanded=nn.Conv1d(4,old.out_channels,old.kernel_size,stride=old.stride,
        padding=old.padding,dilation=old.dilation,bias=old.bias is not None)
    with torch.no_grad():
        expanded.weight.zero_();expanded.weight[:,:2].copy_(old.weight)
        if old.bias is not None:expanded.bias.copy_(old.bias)
    model.features[0]=expanded
    return model


def restore_feature_checkpoint(saved):
    if (saved.get('architecture')!=ARCHITECTURE or saved.get('preprocessing')!=PREPROCESSING
            or saved.get('arm') not in ARMS or saved.get('se_gain')!=2.
            or saved.get('activation')!='tf-compatible'):
        raise ValueError('Not a qualified feature experiment checkpoint')
    state=saved['model']
    base={key:value for key,value in state.items()}
    base['features.0.weight']=state['features.0.weight'][:,:2].clone()
    model=make_feature_model(base)
    model.load_state_dict(state)
    return model
