import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import classical_robustness as robust


class RobustnessTests(unittest.TestCase):
    def test_known_frequency_shift_and_common_crop(self):
        n=np.arange(4096);iq=(.5*np.exp(2j*np.pi*n/16))[None].astype(np.complex64)
        control=robust.transform(iq,'control')
        self.assertEqual(control.shape,(1,4032))
        np.testing.assert_array_equal(control,robust.normalize(iq[:,32:-32].astype(np.complex128)))
        np.testing.assert_array_equal(control,robust.transform(iq,'sro',0))
        shifted=robust.transform(iq,'cfo',800)
        frequency=np.angle(shifted[:,1:]*shifted[:,:-1].conj()).mean()/(2*np.pi)
        self.assertAlmostEqual(float(frequency),1/16+800/32000,places=6)

    def test_resampling_tone_direction_and_no_extrapolation(self):
        iq=(.5*np.exp(2j*np.pi*.05*np.arange(4096)))[None].astype(np.complex64)
        z=robust.transform(iq,'sro',100)
        frequency=np.angle(z[:,1:]*z[:,:-1].conj()).mean()/(2*np.pi)
        self.assertAlmostEqual(float(frequency),.05*1.0001,places=6)
        taps,weights=robust.interpolation_kernel(100)
        self.assertGreaterEqual(taps.min(),0);self.assertLess(taps.max(),4096)
        np.testing.assert_allclose(weights.sum(1),1.,atol=1e-14)
        with self.assertRaisesRegex(ValueError,'margins'):robust.interpolation_kernel(1e6)

    def test_iq_gain_and_phase_definitions(self):
        iq=(.5*np.exp(2j*np.pi*np.arange(4096)/16))[None].astype(np.complex64)
        gain=robust.transform(iq,'gain',1)
        self.assertAlmostEqual(float(gain.real.std()/gain.imag.std()),10**(1/20),places=6)
        phase=robust.transform(iq,'phase',5)
        correlation=np.corrcoef(phase.real[0],phase.imag[0])[0,1]
        self.assertAlmostEqual(float(correlation),np.sin(np.deg2rad(5)),places=6)
        np.testing.assert_allclose(np.sqrt(np.mean(abs(phase)**2,axis=1)),.5,atol=1e-7)

    def test_full_fifteen_variant_pipeline(self):
        rng=np.random.default_rng(9);iq=(rng.normal(size=(24,4096))+1j*rng.normal(size=(24,4096)))
        iq=robust.normalize(iq);y=np.tile(np.arange(12),2)
        arrays=dict(iq=iq,y=y,snr_db=np.where(y==11,np.nan,15.),source_id=np.array([f'id{k}' for k in range(24)]),
            condition_id=np.array([f'g{k%2}' for k in range(24)]))
        features=np.stack([robust.baseline.features(z) for z in iq]);mean,std=robust.baseline.fit_scaler(features)
        model=dict(train_scaled=(features-mean)/std,train_labels=y.copy(),mean=mean,std=std)
        original,_=robust.baseline.neighbors(model['train_scaled'],y,model['train_scaled'])
        for a in model.values():a.flags.writeable=False
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);source=root/'source';folder=root/'holdout';source.mkdir();folder.mkdir()
            (source/'classifier.npz').write_bytes(b'unchanged fixture');(folder/'holdout.npz').write_bytes(b'fixed IQ fixture')
            (folder/'report.json').write_text('{}')
            args=SimpleNamespace(output=root/'out',holdout=folder,data=root,classical_source=None)
            with patch.object(robust,'load_source',return_value=(arrays,model,original,source)):
                report=robust.run(args)
            self.assertTrue(report['complete']);self.assertTrue(report['classifier_unchanged'])
            self.assertEqual(len(report['cases']),15)
            for name,result in report['cases'].items():
                self.assertEqual(result['metrics']['count'],24)
                pair=result['paired'];self.assertEqual(sum(pair[k] for k in ('both_correct','control_only_correct','variant_only_correct','both_wrong')),24)
                if name=='control':self.assertEqual(pair['prediction_changes'],0)
            for file,h in report['artifact_sha256'].items():self.assertEqual(robust.baseline.sha(args.output/file),h)
            self.assertTrue((args.output/'CLASSICAL_ROBUSTNESS_COMPLETE').is_file())

    def test_failure_retained(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);args=SimpleNamespace(output=root/'out',holdout=root,data=root,classical_source=None)
            with patch.object(robust,'load_source',side_effect=ValueError('source hash differs')):
                with self.assertRaises(ValueError):robust.run(args)
            self.assertFalse(json.loads((args.output/'report.json').read_text())['complete'])
            self.assertFalse((args.output/'CLASSICAL_ROBUSTNESS_COMPLETE').exists())

    def test_actual_source_reproduction_and_tamper_rejection(self):
        from test_classical_holdout import HoldoutTests,fake_generator
        module=robust.holdout
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);data,source,plan,versions=HoldoutTests().fixture(root)
            generate=module.generate_set
            with (patch.object(robust.baseline,'PLAN',plan),patch.object(module,'environment',return_value=(versions,'fixture')),
                  patch.object(module,'generate_set',side_effect=lambda out,plan,ids,hashes:generate(out,plan,ids,hashes,fake_generator))):
                module.run(data,source,root/'holdout')
                with patch.object(robust.np,'__version__','fixture'):
                    arrays,model,pred,_=robust.load_source(root/'holdout',data)
                    self.assertEqual(len(pred),1680);self.assertFalse(model['mean'].flags.writeable)
                    path=root/'holdout/features_predictions.npz';path.write_bytes(path.read_bytes()+b'tamper')
                    with self.assertRaisesRegex(ValueError,'artifact path/hash'):robust.load_source(root/'holdout',data)


if __name__=='__main__':unittest.main()
