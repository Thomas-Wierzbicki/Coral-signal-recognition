import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import classical_modulation_baseline as baseline


class BaselineTests(unittest.TestCase):
    def test_features_known_tone_scale_and_phase(self):
        z=.5*np.exp(2j*np.pi*.125*np.arange(4096))
        f=baseline.features(z)
        self.assertEqual(f.shape,(35,));self.assertEqual(len(baseline.FEATURE_NAMES),35)
        np.testing.assert_allclose(f,baseline.features(z*3*np.exp(.7j)),atol=1e-5,rtol=1e-5)
        for lag in baseline.LAGS:
            self.assertAlmostEqual(f[baseline.FEATURE_NAMES.index(f'complex_coherence_{lag}')],1.,places=10)
        self.assertAlmostEqual(f[baseline.FEATURE_NAMES.index('envelope_fourth_moment')],1.)
        with self.assertRaises(ValueError):baseline.features(np.zeros(128,dtype=complex))

    def test_neighbors_self_exclusion_and_ties(self):
        x=np.arange(6,dtype=float)[:,None];y=np.arange(6)
        pred,votes=baseline.neighbors(x,y,x,exclude_self=True)
        self.assertEqual(pred[0],1);self.assertEqual(votes[0,0],0)
        self.assertTrue(np.all(votes.sum(1)==5))
        with self.assertRaises(ValueError):baseline.neighbors(x,y,x[::-1],exclude_self=True)

    def test_scaler_constant_feature_and_validation_independence(self):
        train=np.array([[0.,7.],[2.,7.]])
        mean,std=baseline.fit_scaler(train)
        np.testing.assert_array_equal(mean,[1.,7.]);np.testing.assert_array_equal(std,[1.,1e-6])
        self.assertTrue(np.isfinite((np.array([[1000.,7.]])-mean)/std).all())
        self.assertEqual(mean[0],1.)

    def test_complete_run_and_saved_classifier_roundtrip(self):
        rng=np.random.default_rng(9);y=np.tile(np.arange(12),2)
        arrays={}
        for split in ('train','validation'):
            iq=(rng.normal(size=(24,128))+1j*rng.normal(size=(24,128)))
            arrays[split]=dict(iq=iq,y=y,source_id=np.array([f'{split}{i}' for i in range(24)]),
                condition_id=np.array([f'{split}-condition{i%2}' for i in range(24)]),
                snr_db=np.where(y==11,np.nan,15.))
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);data=root/'data';source=root/'source';data.mkdir();source.mkdir()
            for filename in ('metadata.json','train.npz','validation.npz'):(data/filename).write_bytes(b'fixture')
            (source/'training_selection.npz').write_bytes(b'fixture')
            with patch.object(baseline,'load_data',return_value=arrays):
                report=baseline.run(data,source,root/'out')
            self.assertTrue(report['complete']);self.assertTrue(report['classifier_roundtrip_exact'])
            self.assertEqual(report['validation']['count'],24)
            self.assertIsNone(report['validation']['snr_ge15']['recall']['noise'])
            self.assertTrue((root/'out/CLASSICAL_BASELINE_COMPLETE').is_file())
            with np.load(root/'out/classifier.npz',allow_pickle=False) as model, np.load(root/'out/train_features.npz') as features:
                np.testing.assert_array_equal(model['mean'],features['features'].mean(0))
            json.dumps(report,allow_nan=False)
            for filename,h in report['artifact_sha256'].items():self.assertEqual(baseline.sha(root/'out'/filename),h)
            self.assertFalse((data/'test.npz').exists())

    def test_provenance_failure_preserved_without_success_marker(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            with patch.object(baseline,'load_data',side_effect=ValueError('selection hash differs')):
                with self.assertRaises(ValueError):baseline.run(root,root,root/'out')
            report=json.loads((root/'out/report.json').read_text())
            self.assertFalse(report['complete']);self.assertIn('selection hash',report['error'])
            self.assertFalse((root/'out/CLASSICAL_BASELINE_COMPLETE').exists())

    def test_real_loader_selection_hash_and_complete_groups(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);data=root/'data';source=root/'source';data.mkdir();source.mkdir()
            (data/'DATASET_COMPLETE').touch();(source/'DEVELOPMENT_COMPLETE').touch()
            (data/'metadata.json').write_text(json.dumps(dict(classes=baseline.CLASSES)))
            training=None
            for split,n,prefix in [('train',140,'t'),('validation',90,'v')]:
                y=np.repeat(np.arange(12),n);count=len(y)
                snr=np.tile(np.resize(np.array(baseline.LEVELS),n),12);snr[y==11]=np.nan
                a=dict(iq=np.full((count,4096),.5,dtype=np.complex64),y=y,
                    source_id=np.array([f'{prefix}-id{i}' for i in range(count)]),
                    condition_id=np.tile(np.array([f'{prefix}-group{i}' for i in range(n)]),12),snr_db=snr)
                np.savez_compressed(data/(split+'.npz'),**a)
                if split=='train':training={k:v for k,v in a.items() if k!='iq'}
                del a
            np.savez_compressed(source/'training_selection.npz',**training)
            plan=json.loads(baseline.PLAN.read_text())
            spec=plan['data'];spec.update(metadata_sha256=baseline.sha(data/'metadata.json'),
                train_sha256=baseline.sha(data/'train.npz'),validation_sha256=baseline.sha(data/'validation.npz'),
                selection_sha256=baseline.sha(source/'training_selection.npz'))
            loaded=baseline.load_data(data,source,plan)
            self.assertEqual(loaded['train']['iq'].shape,(1680,4096));del loaded
            training['y']=training['y'].copy();training['y'][0]=1
            np.savez_compressed(source/'training_selection.npz',**training)
            with self.assertRaisesRegex(ValueError,'hash differs'):baseline.load_data(data,source,plan)
            spec['selection_sha256']=baseline.sha(source/'training_selection.npz')
            with self.assertRaisesRegex(ValueError,'row mismatch: y'):baseline.load_data(data,source,plan)


if __name__=='__main__':unittest.main()
