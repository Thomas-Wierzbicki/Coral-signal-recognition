from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from audit_torchsig_generators import metrics,audited_generator,audit,BANDWIDTHS
from torchsig_probe import generator,CLASSES,SAMPLES,RATE

class AuditTests(unittest.TestCase):
    def test_phase_discriminator_known_fm(self):
        n=np.arange(SAMPLES); deviation=1000*np.sin(2*np.pi*1000*n/RATE)
        z=np.exp(2j*np.pi*np.cumsum(500+deviation)/RATE)
        result=metrics(z)
        self.assertAlmostEqual(result['phase_frequency_median_hz'],500,places=4)
        self.assertGreater(result['phase_frequency_p95_abs_centered_hz'],980)
        self.assertLess(result['envelope_cv'],1e-10)

    def test_fm_instrumentation_does_not_change_samples(self):
        plan=dict(bandwidth_hz=4000,rolloff=.35)
        first=generator('fm',plan,np.random.default_rng(5))
        second,details=audited_generator('fm',plan,np.random.default_rng(5))
        np.testing.assert_array_equal(first,second)
        self.assertAlmostEqual(details['message_rms_before_filter'],1,places=5)
        self.assertGreater(details['message_rms_after_filter'],0)

    def test_balanced_grid_and_no_overwrite(self):
        def fake(label,plan,rng): return np.exp(2j*np.pi*.02*np.arange(SAMPLES)),{}
        with tempfile.TemporaryDirectory() as tmp,patch('audit_torchsig_generators.make_plot'):
            out=Path(tmp)/'audit'; report=audit(out,2,generate=fake)
            self.assertEqual(report['cases'],(len(CLASSES)-1)*len(BANDWIDTHS)*2)
            self.assertTrue((out/'AUDIT_COMPLETE').exists())
            self.assertEqual(set(report['groups']),set(CLASSES[:-1]))
            with self.assertRaises(ValueError): audit(out,2,generate=fake)

if __name__=='__main__': unittest.main()
