import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from diagnose_rtl_domain import features,diagnose
from classify_rtl_channel import LABELS

class DomainTests(unittest.TestCase):
    def test_tone_features_and_invariance(self):
        z=np.exp(2j*np.pi*.03125*np.arange(1024))
        f=features(z)
        self.assertAlmostEqual(f['phase_offset_cycles_per_sample'],.03125,places=6)
        self.assertAlmostEqual(f['lag1_phase_coherence'],1,places=6)
        self.assertLess(f['envelope_cv'],1e-6)
        self.assertLess(f['phase_step_change_rms_rad'],1e-6)
        other=features(z*7*np.exp(.7j))
        for key in f: self.assertAlmostEqual(f[key],other[key],places=5)

    def test_phase_varies_and_invalid_input(self):
        tone=np.exp(2j*np.pi*.02*np.arange(1024))
        fm=np.exp(1j*np.cumsum(.2*np.sin(2*np.pi*np.arange(1024)/40)))
        self.assertGreater(features(fm)['phase_spread_rad'],features(tone)['phase_spread_rad']+.1)
        with self.assertRaises(ValueError): features(np.zeros(1024,dtype=complex))

    def test_report_uses_only_training_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); channel=p/'channel'; channel.mkdir(); ref=p/'ref'; ref.mkdir()
            tone=np.exp(2j*np.pi*.02*np.arange(33+2*1024+10)).astype('<c8')
            tone.tofile(channel/'channel.cf32'); (channel/'CHANNEL_COMPLETE').touch()
            (channel/'channel_report.json').write_text(json.dumps(dict(output_rate_hz_nominal=32000,
                format='little-endian complex64 interleaved I,Q',samples=len(tone),statistic_start_sample=33)))
            (ref/'IMPORT_COMPLETE').touch()
            (ref/'metadata.json').write_text(json.dumps(dict(source='RadioML2018.01A',representation='iq-rms-v1',
                input_shape=[1,1024,2],target_rms=.5,labels=LABELS)))
            z=tone[:1024]
            x=np.tile(np.stack((z.real,z.imag),axis=-1)[None,None],(5,1,1,1))
            np.savez(ref/'train.npz',x=x,y=np.arange(5),snr_db=np.full(5,10),source_index=np.arange(5))
            result=diagnose(channel,ref,p/'out')
            self.assertEqual(result['groups']['real/all']['count'],2)
            self.assertEqual(result['groups']['radioml_train_snr_ge10/FM']['count'],1)
            self.assertTrue((p/'out/features.csv').exists())
            with self.assertRaises(ValueError): diagnose(channel,ref,p/'out')

if __name__=='__main__': unittest.main()
