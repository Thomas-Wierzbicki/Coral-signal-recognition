"""Gepaarte Skalierung und Auswertung ohne Coral pruefen."""
from contextlib import redirect_stdout
import io
import unittest
import numpy as np
from signal_lab import generate_iq
from level_sweep import rescale_iq,run_sweep,build_report


class LevelTests(unittest.TestCase):
    def test_same_iq_scaled_including_noise(self):
        iq,params=generate_iq("fm",np.random.default_rng(77),return_metadata=True,snr_db=10)
        low=rescale_iq(iq,params["amplitude"],0.2)
        high=rescale_iq(iq,params["amplitude"],0.8)
        np.testing.assert_allclose(high,4*low,rtol=1e-6,atol=1e-7)
    def test_paired_decisions_and_counts(self):
        class ConstantClassifier:
            def classify(self,iq):
                return dict(label="carrier",scores=dict(noise=0,carrier=0.8,fm=0.2))
        with redirect_stdout(io.StringIO()):
            rows,pairs=run_sweep(ConstantClassifier(),1,2029,lambda r:None)
        report=build_report(rows,pairs)
        self.assertEqual(report["count"],48)
        self.assertEqual(report["base_signals"],6)
        self.assertEqual(report["pairs_with_label_changes"],0)
        self.assertEqual(len(report["by_condition"]),24)
        for pair in pairs:
            selected=[r for r in rows if r["pair_id"]==pair["pair_id"]]
            self.assertEqual(len(selected),8)
            self.assertEqual(len({r["offset_hz"] for r in selected}),1)
            self.assertEqual(len({r["audio_hz"] for r in selected}),1)
            self.assertEqual(len({r["source_amplitude"] for r in selected}),1)


if __name__=="__main__":
    unittest.main()
