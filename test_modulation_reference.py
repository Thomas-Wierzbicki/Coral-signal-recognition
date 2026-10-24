"""Numerical check against the source's layer equations, independent NumPy path."""
import unittest
import numpy as np
import torch
from modulation_reference import Harper2023, SqueezeExcitation, StatisticsPooling


def numpy_forward(model, inputs):
    x=inputs.squeeze(2).numpy().astype(np.float64)
    for layer in model.features:
        if isinstance(layer,torch.nn.Conv1d):
            w=layer.weight.detach().numpy(); b=layer.bias.detach().numpy()
            dilation=layer.dilation[0]; kernel=layer.kernel_size[0]; pad=layer.padding[0]
            padded=np.pad(x,((0,0),(0,0),(pad,pad)))
            x=sum(np.einsum('bct,oc->bot',padded[:,:,j*dilation:j*dilation+x.shape[2]],w[:,:,j])
                  for j in range(kernel))+b[None,:,None]
        elif isinstance(layer,torch.nn.ReLU): x=np.maximum(x,0)
        elif isinstance(layer,SqueezeExcitation):
            s=x.mean(2)@layer.reduce.weight.detach().numpy().T
            s=np.maximum(s,0)@layer.expand.weight.detach().numpy().T
            x=x/(1+np.exp(-s))[:,:,None]
        else: raise AssertionError(type(layer))
    x=np.concatenate((x.mean(2),x.var(2)),axis=1)
    for layer in model.head:
        if isinstance(layer,torch.nn.Linear):
            x=x@layer.weight.detach().numpy().T+layer.bias.detach().numpy()
        elif isinstance(layer,torch.nn.SELU):
            x=1.0507009873554805*np.where(x>0,x,1.6732632423543772*np.expm1(x))
        else: raise AssertionError(type(layer))
    return x


class ReferenceTests(unittest.TestCase):
    def test_source_equations(self):
        torch.manual_seed(7); model=Harper2023().eval()
        inputs=torch.randn(2,2,1,43)
        with torch.inference_mode(): actual=model(inputs).numpy()
        np.testing.assert_allclose(actual,numpy_forward(model,inputs),atol=2e-6,rtol=2e-5)

    def test_population_variance(self):
        x=torch.tensor([[[0.,2.],[1.,5.]]])
        torch.testing.assert_close(StatisticsPooling()(x),torch.tensor([[1.,3.,1.,4.]]))

    def test_full_length_backward_and_portable_batch(self):
        model=Harper2023(); x=torch.randn(2,2,1,4096)
        output=model(x); self.assertEqual(tuple(output.shape),(2,12))
        torch.nn.functional.cross_entropy(output,torch.tensor([0,11])).backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        model.eval()
        with torch.inference_mode():
            traced=torch.jit.trace(model,x[:1])
            torch.testing.assert_close(model(x),traced(x))

if __name__=='__main__': unittest.main()
