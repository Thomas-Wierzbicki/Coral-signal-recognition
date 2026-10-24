import unittest
import numpy as np
import torch
from train_modulation12 import as_tensor, make_model, metrics, shifted
from torchsig_probe import CLASSES, SAMPLES

class TrainingTests(unittest.TestCase):
    def test_iq_layout_and_export(self):
        iq=np.full((2,SAMPLES),.3+.4j,dtype=np.complex64)
        inputs=as_tensor(iq)
        self.assertEqual(tuple(inputs.shape),(2,2,1,SAMPLES))
        np.testing.assert_allclose(inputs[:,0,0].numpy(),.3)
        np.testing.assert_allclose(inputs[:,1,0].numpy(),.4)
        model=make_model().eval()
        with torch.inference_mode():
            logits=model(inputs)
            self.assertEqual(tuple(logits.shape),(2,len(CLASSES)))
            traced=torch.jit.trace(model,inputs[:1])
            torch.testing.assert_close(logits,traced(inputs))

    def test_matrix_orientation_and_macro_f1(self):
        y=np.arange(len(CLASSES)); pred=y.copy(); pred[0]=1
        result=metrics(y,pred)
        self.assertEqual(result['confusion_matrix'][0][1],1)
        self.assertEqual(result['confusion_matrix'][1][0],0)
        self.assertAlmostEqual(result['accuracy'],11/12)
        self.assertAlmostEqual(result['macro_f1'],(10+2/3)/12)

    def test_frequency_shift_preserves_envelope_and_is_reversible(self):
        rng=np.random.default_rng(42)
        iq=(rng.normal(size=(3,SAMPLES))+1j*rng.normal(size=(3,SAMPLES))).astype(np.complex64)
        changed=shifted(iq,1600)
        np.testing.assert_allclose(abs(changed),abs(iq),atol=1e-6)
        np.testing.assert_allclose(shifted(changed,-1600),iq,atol=1e-6)
        self.assertFalse(np.allclose(changed,iq))

if __name__=='__main__': unittest.main()
