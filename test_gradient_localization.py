import unittest
import numpy as np
import torch
from localize_reference_gradients import bias_vjp,decompose_bias
from tools.replay_reference_relu_mask import MaskProbe

class LocalizationTests(unittest.TestCase):
    def test_isolated_bias_vjp_in_both_layouts(self):
        q=np.arange(24,dtype=np.float64).reshape(2,3,4)
        expected=q.sum(axis=(0,1))
        for channels_first in (False,True):
            np.testing.assert_array_equal(bias_vjp(q,channels_first),expected)

    def test_decomposition_separates_mask_and_reduction_effects(self):
        u=np.ones((2,3,2),dtype=np.float64)
        mt=np.ones_like(u,dtype=bool);mt[0,0,0]=False
        mp=np.ones_like(mt)
        tf_g=(u*mt).sum((0,1))-np.array([.01,0])
        pt_g=(u*mp).sum((0,1))+np.array([0,.02])
        r=decompose_bias(tf_g,pt_g,u,u,mt,mp)
        self.assertEqual(r['worst_channel'],0)
        self.assertAlmostEqual(r['mask_change_term']['at_worst_channel'],1.)
        self.assertAlmostEqual(r['tf_reduction_term']['at_worst_channel'],.01)
        self.assertAlmostEqual(r['pt_reduction_term']['max_abs'],.02)
        self.assertLess(r['decomposition_residual_max_abs'],1e-12)

    def test_counterfactual_changes_only_backward_mask(self):
        x=torch.tensor([-1.,0.,1.],requires_grad=True)
        probe=MaskProbe(torch.tensor([True,False,False]))
        y=probe(x)
        torch.testing.assert_close(y,torch.relu(x),atol=0,rtol=0)
        y.sum().backward()
        torch.testing.assert_close(x.grad,torch.tensor([1.,0.,0.]),atol=0,rtol=0)

if __name__=='__main__':unittest.main()
