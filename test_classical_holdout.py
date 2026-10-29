import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import evaluate_classical_holdout as holdout
import classical_modulation_baseline as baseline


def fake_generator(label,item,signal_seed,noise_seed):
    rng=np.random.default_rng(signal_seed)
    z=rng.normal(size=4096)+1j*rng.normal(size=4096)
    return (.5*z/np.sqrt(np.mean(abs(z)**2))).astype(np.complex64),{}


class HoldoutTests(unittest.TestCase):
    def fixture(self,root):
        data=root/'data';source=root/'source';data.mkdir();source.mkdir()
        rng=np.random.default_rng(8);train=rng.normal(size=(1680,35));val=rng.normal(size=(1080,35))
        y=np.repeat(np.arange(12),140);vy=np.repeat(np.arange(12),90)
        ids=np.array([f'old-train-{k}' for k in range(1680)]);vids=np.array([f'old-val-{k}' for k in range(1080)])
        mean,std=baseline.fit_scaler(train);scaled=(train-mean)/std
        np.savez_compressed(source/'classifier.npz',neighbors=5,feature_names=np.array(baseline.FEATURE_NAMES),
            train_scaled=scaled,train_labels=y,train_source_id=ids,mean=mean,std=std)
        np.savez_compressed(source/'train_features.npz',features=train,y=y,source_id=ids)
        np.savez_compressed(source/'validation_features.npz',features=val,y=vy,source_id=vids)
        pred,_=baseline.neighbors(scaled,y,(val-mean)/std)
        np.savez_compressed(source/'validation_predictions.npz',predicted=pred,source_id=vids)
        (source/'protocol.json').write_text('{}');(source/'CLASSICAL_BASELINE_COMPLETE').touch()
        versions=dict(torchsig='2.2.0',torch='fixture',numpy='fixture',scipy='fixture')
        meta=dict(classes=holdout.probe.CLASSES,seed=2050,versions=dict(versions,
            builder_source_sha256=baseline.sha(holdout.builder.__file__)),splits={})
        for split,identities in [('train',ids),('validation',vids)]:
            path=data/(split+'_parameters.jsonl')
            path.write_text(''.join(json.dumps(dict(source_id=str(sid),iq_sha256='old-'+str(sid)))+'\n' for sid in identities))
            meta['splits'][split]=dict(parameters_sha256=baseline.sha(path))
        (data/'metadata.json').write_text(json.dumps(meta))
        plan=json.loads(baseline.PLAN.read_text());plan['data']['metadata_sha256']=baseline.sha(data/'metadata.json')
        planfile=root/'baseline-plan.json';planfile.write_text(json.dumps(plan))
        report=dict(complete=True,classifier_roundtrip_exact=True,plan=plan,plan_sha256=baseline.sha(planfile),
            script_sha256=baseline.sha(baseline.__file__),feature_names=baseline.FEATURE_NAMES,numpy_version='fixture',
            inputs={key:plan['data'][key] for key in ('metadata_sha256','train_sha256','validation_sha256','selection_sha256')},
            validation=dict(baseline.metrics(vy,pred),fsk_family_mean_recall=.1),
            artifact_sha256={p.name:baseline.sha(p) for p in source.iterdir() if p.suffix in ('.npz','.json')})
        (source/'report.json').write_text(json.dumps(report))
        return data,source,planfile,versions

    def test_freeze_source_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);data,source,plan,versions=self.fixture(root)
            with patch.object(baseline,'PLAN',plan):
                _,_,model,ids,hashes=holdout.freeze_source(data,source)
                self.assertEqual(len(ids),2760);self.assertEqual(model['train_scaled'].shape,(1680,35))
                path=source/'classifier.npz';path.write_bytes(path.read_bytes()+b'tamper')
                with self.assertRaisesRegex(ValueError,'artifact hash'):holdout.freeze_source(data,source)

    def test_generation_rejects_overlap(self):
        plan=json.loads(holdout.PLAN.read_text())
        item=holdout.builder.condition_plan(140,2100,4)[0]
        z,_=fake_generator('am-dsb',item,[2100,4,0,0,1],[2100,4,0,0,2])
        digest=holdout.hashlib.sha256(z.tobytes()).hexdigest()
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(ValueError,'overlap'):
                holdout.generate_set(Path(name),plan,set(),{digest},fake_generator)

    def test_complete_pipeline_frozen_model_and_persisted_iq(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);data,source,plan,versions=self.fixture(root)
            before=baseline.sha(source/'classifier.npz')
            original_generate_set=holdout.generate_set
            with (patch.object(baseline,'PLAN',plan),patch.object(holdout,'environment',return_value=(versions,'fixture')),
                 patch.object(holdout,'generate_set',side_effect=lambda out,plan,ids,hashes:original_generate_set(out,plan,ids,hashes,fake_generator))):
                report=holdout.run(data,source,root/'out')
            self.assertTrue(report['complete']);self.assertTrue(report['classifier_unchanged'])
            self.assertEqual(report['holdout']['count'],1680)
            self.assertEqual(baseline.sha(source/'classifier.npz'),before)
            self.assertTrue((root/'out/CLASSICAL_HOLDOUT_COMPLETE').exists())
            self.assertFalse((data/'test.npz').exists())
            for filename,h in report['artifact_sha256'].items():self.assertEqual(baseline.sha(root/'out'/filename),h)
            with np.load(root/'out/holdout.npz',allow_pickle=False) as saved:
                self.assertTrue(np.array_equal(np.bincount(saved['y']),np.full(12,140)))
                self.assertTrue(np.isnan(saved['snr_db'][saved['y']==11]).all())

    def test_preflight_failure_retains_report(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            with patch.object(holdout,'freeze_source',side_effect=ValueError('frozen model mismatch')):
                with self.assertRaises(ValueError):holdout.run(root,root,root/'out')
            self.assertFalse(json.loads((root/'out/report.json').read_text())['complete'])
            self.assertFalse((root/'out/CLASSICAL_HOLDOUT_COMPLETE').exists())


if __name__=='__main__':unittest.main()
