"""Prueft RMS-Invarianz, Manifest und unveraenderten Rohpfad ohne Hardware."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from signal_lab import generate_iq,spectrogram
from signal_preprocessing import prepare_iq,model_preprocessing,TARGET_RMS


class PreprocessingTests(unittest.TestCase):
    def test_scale_invariance_and_rms(self):
        for label in ("noise","carrier","fm"):
            iq=generate_iq(label,np.random.default_rng(44))
            a=prepare_iq(iq,"rms-v1")
            b=prepare_iq(iq*3,"rms-v1")
            self.assertAlmostEqual(float(np.sqrt(np.mean(np.abs(a)**2))),TARGET_RMS,places=6)
            np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-7)
            np.testing.assert_allclose(spectrogram(iq,"rms-v1"),
                                       spectrogram(iq*3,"rms-v1"),rtol=1e-4,atol=1e-5)
    def test_raw_zero_and_invalid(self):
        iq=generate_iq("carrier",np.random.default_rng(2))
        np.testing.assert_array_equal(prepare_iq(iq),iq)
        self.assertTrue(np.isfinite(prepare_iq(np.zeros(16,np.complex64),"rms-v1")).all())
        with self.assertRaises(ValueError):
            prepare_iq(iq,"unknown")
        with self.assertRaises(ValueError):
            prepare_iq(np.array([np.nan]),"rms-v1")
    def test_manifest_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            model=Path(directory)/"model.tflite"
            self.assertEqual(model_preprocessing(model),"raw")
            manifest=dict(schema=1,mode="rms-v1",target_rms=TARGET_RMS,
                          labels=["noise","carrier","fm"],input_shape=[64,256,1])
            path=Path(directory)/"preprocessing.json"
            path.write_text(json.dumps(manifest))
            self.assertEqual(model_preprocessing(model),"rms-v1")
            manifest["target_rms"]=1.0
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                model_preprocessing(model)


if __name__=="__main__":
    unittest.main()
