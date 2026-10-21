"""Tests fuer Metadaten und Diagnoseauswertung ohne Hardware."""
import unittest
import numpy as np
from signal_lab import generate_iq, SAMPLES
from diagnose_signal import run_diagnosis, summary, SNR_VALUES


class FakeClassifier:
    def classify(self,iq):
        assert iq.shape==(SAMPLES,)
        return dict(label="noise",scores=dict(noise=0.9,carrier=0.05,fm=0.05),invoke_ms=0.1)


class DiagnosisTests(unittest.TestCase):
    def test_metadata_does_not_change_samples(self):
        for label in ("noise","carrier","fm"):
            original=generate_iq(label,np.random.default_rng(12))
            iq,metadata=generate_iq(label,np.random.default_rng(12),return_metadata=True)
            np.testing.assert_array_equal(original,iq)
            self.assertEqual(metadata["label"],label)
            self.assertEqual(metadata["deviation_hz"] is not None,label=="fm")
    def test_fixed_snr_changes_only_noise_scale(self):
        a,meta=generate_iq("noise",np.random.default_rng(15),return_metadata=True,snr_db=0)
        b,_=generate_iq("noise",np.random.default_rng(15),return_metadata=True,snr_db=20)
        self.assertEqual(meta["snr_db"],0)
        np.testing.assert_allclose(a/10,b,rtol=1e-6,atol=1e-8)
    def test_diagnostic_matrix_and_rows(self):
        written=[]
        rows=run_diagnosis(FakeClassifier(),2,2028,written.append)
        self.assertEqual(len(rows),len(SNR_VALUES)*3*2)
        self.assertEqual(len(written),len(rows))
        stats=summary(rows)
        self.assertAlmostEqual(stats["accuracy"],1/3)
        self.assertEqual(stats["confusion_matrix"],[[14,0,0],[14,0,0],[14,0,0]])
        self.assertEqual(summary([])["accuracy"],None)


if __name__=="__main__":
    unittest.main()
