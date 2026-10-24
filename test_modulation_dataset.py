import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from build_modulation_dataset import controlled_fm,condition_plan,build,CLASSES,RATE,SAMPLES,sha

class DatasetTests(unittest.TestCase):
    def test_controlled_fm_has_constant_envelope_and_bounded_hub(self):
        first,meta=controlled_fm(8000,np.random.default_rng(5))
        second,_=controlled_fm(8000,np.random.default_rng(5))
        np.testing.assert_array_equal(first,second)
        np.testing.assert_allclose(abs(first),1,atol=1e-7)
        frequency=np.angle(first[1:]*first[:-1].conj())*RATE/(2*np.pi)
        self.assertLessEqual(float(max(abs(frequency))),meta['peak_deviation_parameter_hz']+.01)
        self.assertGreater(meta['crop_rms_deviation_hz'],0)
        self.assertLess(meta['message_band_high_hz']+meta['peak_deviation_parameter_hz'],4001)

    def test_condition_plans_differ_by_split(self):
        self.assertEqual(condition_plan(20,2050,1),condition_plan(20,2050,1))
        self.assertNotEqual(condition_plan(20,2050,1),condition_plan(20,2050,2))

    def test_disjoint_complete_files_and_hashes(self):
        def fake(label,plan,rng):
            return np.exp(1j*np.cumsum(rng.normal(size=SAMPLES)*.1))
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'data'; meta=build(out,20,2050,generate=fake)
            seen=set()
            for split,stats in meta['splits'].items():
                with np.load(out/(split+'.npz')) as data:
                    ids=set(data['source_id'].tolist())
                    self.assertFalse(ids&seen); seen.update(ids)
                    np.testing.assert_allclose(np.sqrt(np.mean(abs(data['iq'])**2,axis=1)),.5,atol=1e-6)
                    self.assertEqual(data['iq'].shape[1],SAMPLES)
                    self.assertTrue(np.isnan(data['snr_db'][data['y']==CLASSES.index('noise')]).all())
                self.assertEqual(stats['sha256'],sha(out/(split+'.npz')))
                self.assertEqual(stats['parameters_sha256'],sha(out/(split+'_parameters.jsonl')))
            self.assertEqual(len(seen),240)
            self.assertTrue((out/'DATASET_COMPLETE').exists())
            with self.assertRaises(ValueError): build(out,20,generate=fake)

if __name__=='__main__': unittest.main()
