from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from torchsig_probe import parameter_plan,add_channel,create_probe,CLASSES,SAMPLES

class ProbeTests(unittest.TestCase):
    def test_plan_reproducible_and_noise_scale(self):
        self.assertEqual(parameter_plan(12,2040),parameter_plan(12,2040))
        plan=parameter_plan(3,2040)[2]
        z=np.ones(65536,dtype=complex)
        iq,clean=add_channel(z,plan,np.random.default_rng(1))
        self.assertAlmostEqual(float(np.sqrt(np.mean(abs(iq)**2))),.5,places=6)
        self.assertAlmostEqual(float(np.angle(clean[1]*clean[0].conj())/(2*np.pi)),plan['offset_hz']/32000,places=6)

    def test_probe_records_common_plan_without_training_splits(self):
        def fake(label,plan,rng): return np.exp(2j*np.pi*.03*np.arange(SAMPLES))
        with tempfile.TemporaryDirectory() as tmp,patch('torchsig_probe.plot_previews'):
            out=Path(tmp)/'probe'; meta=create_probe(out,3,2040,generate=fake)
            self.assertTrue((out/'PROBE_COMPLETE').exists())
            with np.load(out/'probe.npz') as d: self.assertEqual(d['iq'].shape,(len(CLASSES)*3,SAMPLES))
            self.assertEqual(meta['examples'][0]['offset_hz'],meta['examples'][3]['offset_hz'])
            self.assertIsNone(meta['examples'][-1]['snr_db'])
            self.assertFalse((out/'train.npz').exists())
            with self.assertRaises(ValueError): create_probe(out,3,2040,generate=fake)

if __name__=='__main__': unittest.main()
