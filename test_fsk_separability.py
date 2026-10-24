import unittest
import numpy as np
from diagnose_fsk_separability import features, matrix, RATE

class DiagnosisTests(unittest.TestCase):
    def test_matrix_orientation(self):
        self.assertEqual(matrix(np.array([0,0,1,2]),np.array([1,0,1,2])),[[1,1,0],[0,1,0],[0,0,1]])

    def test_features_ignore_constant_frequency_shift(self):
        rng=np.random.default_rng(3)
        steps=rng.uniform(-.1,.1,4095)
        iq=np.exp(1j*np.r_[0,np.cumsum(steps)])
        other=iq*np.exp(2j*np.pi*1000/RATE*np.arange(len(iq)))
        np.testing.assert_allclose(features(iq),features(other),atol=1e-10)

if __name__=='__main__': unittest.main()
