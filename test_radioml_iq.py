import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from radioml_iq_import import iq_features
from train_radioml_iq import load_dataset

class IQTests(unittest.TestCase):
    def test_phase_and_scale(self):
        t=np.arange(1024)*0.1
        pairs=np.stack((np.cos(t),np.sin(t)),axis=-1)
        x=iq_features(pairs)
        self.assertEqual(x.shape,(1,1024,2))
        np.testing.assert_allclose(x,iq_features(7*pairs),atol=1e-7)
        self.assertAlmostEqual(float(np.sqrt(np.mean(np.sum(x*x,axis=-1)))),0.5,places=6)
        np.testing.assert_allclose(iq_features(np.stack((-pairs[:,1],pairs[:,0]),axis=-1))[...,0],-x[...,1],atol=1e-7)
        with self.assertRaises(ValueError):
            iq_features(np.full((1024,2),np.nan))

    def test_split_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'IMPORT_COMPLETE').write_text('OK')
            meta=dict(source='RadioML2018.01A',preprocessing='rms-v1',target_rms=0.5,
                      representation='iq-rms-v1',input_shape=[1,1024,2],labels=list('ABCDE'))
            (p/'metadata.json').write_text(json.dumps(meta))
            for k,name in enumerate(('train','validation','test')):
                np.savez(p/(name+'.npz'),x=np.full((5,1,1024,2),-0.5),y=np.arange(5),
                         snr_db=np.zeros(5),source_index=np.arange(5)+k*5)
            load_dataset(p)
            np.savez(p/'test.npz',x=np.zeros((5,1,1024,2)),y=np.arange(5),snr_db=np.zeros(5),source_index=np.arange(5))
            with self.assertRaises(ValueError):
                load_dataset(p)

if __name__=='__main__':
    unittest.main()
