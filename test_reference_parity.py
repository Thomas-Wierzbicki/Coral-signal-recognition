import unittest
import numpy as np
import torch
from modulation_reference import Harper2023,TensorFlowSELU
from verify_reference_parity import parameter_map,to_torch_layout,check,matched_first_adam_step

class ParityTests(unittest.TestCase):
    def test_mapping_covers_all_actual_parameters(self):
        for classes in (12,24):
            self.assertEqual(set(Harper2023(classes).state_dict()),set(parameter_map()))
        self.assertEqual(len(set(parameter_map().values())),len(parameter_map()))

    def test_kernel_layout_preserves_known_element(self):
        a=np.arange(7*2*32).reshape(7,2,32)
        b=to_torch_layout(a)
        self.assertEqual(b.shape,(32,2,7));self.assertEqual(b[3,1,4],a[4,1,3])
        np.testing.assert_array_equal(to_torch_layout(np.eye(3)),np.eye(3))

    def test_gate_rejects_nonfinite_shape_and_numerical_mismatch(self):
        self.assertTrue(check('equal',np.ones(3),np.ones(3))['passed'])
        self.assertFalse(check('shape',np.ones(4),np.ones(3))['passed'])
        self.assertFalse(check('nan',np.array([np.nan]),np.zeros(1))['passed'])
        self.assertFalse(check('wrong',np.ones(3)*1.01,np.ones(3))['passed'])

    def test_zero_gradient_has_no_update(self):
        w=torch.tensor([1.,-2.]);g=torch.zeros_like(w)
        torch.testing.assert_close(matched_first_adam_step(w,g),w,atol=0,rtol=0)

    def test_tensorflow_selu_zero_derivative_and_large_positive_input(self):
        x=torch.tensor([-.1,0.,.1,100.],requires_grad=True)
        y=TensorFlowSELU()(x);y.sum().backward()
        torch.testing.assert_close(x.grad,torch.tensor([1.5907940865,1.0507010221,1.0507010221,1.0507010221]))
        self.assertTrue(torch.isfinite(y).all())

    def test_qualified_candidate_trace_and_backward(self):
        m=Harper2023(12,tensorflow_selu=True)
        x=torch.randn(2,2,1,128)
        m(x).sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
        with torch.inference_mode():
            traced=torch.jit.trace(m.eval(),x[:1])
            torch.testing.assert_close(traced(x),m(x))

if __name__=='__main__': unittest.main()
