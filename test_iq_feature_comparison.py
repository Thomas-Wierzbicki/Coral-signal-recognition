import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from iq_feature_experiment import (feature_tensor,make_feature_model,restore_feature_checkpoint,
    ARCHITECTURE,PREPROCESSING,ARMS)
from se_gain_experiment import make_gain_model
from check_modulation_learning import state_digest
from train_iq_feature_comparison import train_case,compare,detailed_score
import train_iq_feature_comparison as comparison


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_known_tone_and_zero_phase(self):
        iq=(.5*np.exp(2j*np.pi*.125*np.arange(32)))[None].astype(np.complex64)
        result=feature_tensor(iq,ARMS[1])
        np.testing.assert_array_equal(result[:,0,0].numpy(),iq.real)
        np.testing.assert_array_equal(result[:,1,0].numpy(),iq.imag)
        np.testing.assert_allclose(result[:,2,0].numpy(),.5,atol=1e-7)
        np.testing.assert_allclose(result[:,3,0,1:].numpy(),.25,atol=1e-7)
        self.assertEqual(result[0,3,0,0],0)
        self.assertEqual(torch.count_nonzero(feature_tensor(iq,ARMS[0])[:,2:]),0)
        self.assertTrue(torch.isfinite(feature_tensor(np.zeros((1,32),np.complex64),ARMS[1])).all())

    def test_paired_initial_outputs_and_extra_gradients(self):
        torch.manual_seed(12);base=make_gain_model(2.)
        before=state_digest(base.state_dict());model=make_feature_model(base.state_dict())
        rng=np.random.default_rng(10);iq=(rng.normal(size=(12,64))+1j*rng.normal(size=(12,64))).astype(np.complex64)
        a,b=(feature_tensor(iq,arm) for arm in ARMS)
        self.assertTrue(torch.equal(model(a),model(b)))
        self.assertEqual(state_digest(base.state_dict()),before)
        np.testing.assert_allclose(model(a).detach().numpy(),base(a[:,:2]).detach().numpy(),rtol=1e-5,atol=1e-7)
        y=torch.arange(12)
        nn.functional.cross_entropy(model(a),y).backward()
        self.assertEqual(torch.count_nonzero(model.features[0].weight.grad[:,2:]),0)
        model.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(b),y).backward()
        self.assertGreater(torch.count_nonzero(model.features[0].weight.grad[:,2:]).item(),0)

    def test_training_checkpoint_pairing_and_restore(self):
        torch.manual_seed(13);initial=make_gain_model(2.).state_dict()
        rng=np.random.default_rng(11);iq=(rng.normal(size=(24,32))+1j*rng.normal(size=(24,32))).astype(np.complex64)
        y=torch.arange(24)%12;plan=dict(epochs=2,batch_size=8,batch_order_seed=2080,optimizer=dict(lr=.0001,eps=1e-8))
        histories=[]
        with tempfile.TemporaryDirectory() as name:
            for arm in ARMS:
                folder=Path(name)/arm;folder.mkdir();model=make_feature_model(initial)
                x=feature_tensor(iq,arm)
                context=dict(model_seed=2081,arm=arm,architecture=ARCHITECTURE,preprocessing=PREPROCESSING,
                    se_gain=2.,activation='tf-compatible')
                histories.append(train_case(model,x,y,x,y,folder,'cpu',plan,context))
                saved=torch.load(folder/'last_checkpoint.pt',weights_only=True)
                restored=restore_feature_checkpoint(saved)
                self.assertEqual(state_digest(model.state_dict()),state_digest(restored.state_dict()))
                expected_generator=torch.Generator().manual_seed(2080)
                for _ in range(2):torch.randperm(24,generator=expected_generator)
                self.assertTrue(torch.equal(saved['generator_state'],expected_generator.get_state()))
                self.assertTrue(all(int(state['step'])==6 for state in saved['optimizer']['state'].values()))
                self.assertEqual(saved['steps'],6)
                if arm==ARMS[0]:self.assertEqual(torch.count_nonzero(restored.features[0].weight[:,2:]),0)
                else:self.assertGreater(torch.count_nonzero(restored.features[0].weight[:,2:]).item(),0)
                saved['preprocessing']='unknown'
                with self.assertRaises(ValueError):restore_feature_checkpoint(saved)
            self.assertEqual([r['batch_order_sha256'] for r in histories[0]],
                [r['batch_order_sha256'] for r in histories[1]])

    def test_high_snr_report_serializes_without_noise_nan(self):
        model=nn.Sequential(nn.Flatten(),nn.Linear(4*32,12))
        x=torch.randn(24,4,1,32);y=torch.arange(24)%12
        snr=np.array([15. if n%12!=11 else np.nan for n in range(24)])
        result=detailed_score(model,x,y,snr,'cpu')
        self.assertIsNone(result['snr_ge15']['recall']['noise'])
        json.dumps(result,allow_nan=False)

    def test_primary_rule_requires_both_metrics_in_both_seeds(self):
        baseline=dict(loss=1.,macro_f1=.4,accuracy=.5)
        better=dict(loss=.9,macro_f1=.5,accuracy=.6)
        runs={str(s):{ARMS[0]:dict(selected_validation=baseline),ARMS[1]:dict(selected_validation=better)} for s in (2081,2082)}
        self.assertTrue(compare(runs,[2081,2082])['directional_development_advantage'])
        runs['2082'][ARMS[1]]['selected_validation']=dict(better,macro_f1=.3)
        self.assertFalse(compare(runs,[2081,2082])['directional_development_advantage'])
        del runs['2082']
        self.assertIsNone(compare(runs,[2081,2082])['directional_development_advantage'])

    def test_complete_four_case_workflow(self):
        rng=np.random.default_rng(9)
        iq=(rng.normal(size=(24,32))+1j*rng.normal(size=(24,32))).astype(np.complex64)
        y=torch.arange(24)%12;snr=np.array([15. if k%12!=11 else np.nan for k in range(24)])
        initial={str(s):make_gain_model(2.).state_dict() for s in (2081,2082)}
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);source=root/'source';source.mkdir()
            (source/'training_selection.npz').write_bytes(b'fixture selection')
            for seed,state in initial.items():
                (source/f'seed{seed}').mkdir();torch.save(state,source/f'seed{seed}'/'initial_state.pt')
            plan=json.loads(comparison.PLAN.read_text());plan.update(epochs=2,batch_size=8,steps_per_epoch=3)
            plan_path=root/'plan.json';plan_path.write_text(json.dumps(plan))
            def prepared(data,src,plan,output):
                (output/'training_selection.npz').write_bytes(b'fixture selection')
                return iq,y,snr,iq,y,snr,initial,['fixture']
            args=SimpleNamespace(data=root/'unused',source=source,output=root/'out',device='cpu')
            previous=torch.are_deterministic_algorithms_enabled()
            try:
                with patch.object(comparison,'PLAN',plan_path),patch.object(comparison,'prepare',side_effect=prepared):
                    comparison.run(args)
            finally:torch.use_deterministic_algorithms(previous)
            report=json.loads((args.output/'report.json').read_text())
            self.assertTrue(report['complete']);self.assertTrue(report['summary']['complete'])
            self.assertTrue((args.output/'FEATURE_COMPARISON_COMPLETE').is_file())
            for seed,cases in report['runs'].items():
                self.assertEqual(cases[ARMS[0]]['initial_four_channel_sha256'],cases[ARMS[1]]['initial_four_channel_sha256'])
                self.assertEqual(cases[ARMS[0]]['parameter_count'],cases[ARMS[1]]['parameter_count'])
                for arm,result in cases.items():
                    self.assertEqual(result['steps'],6)
                    self.assertEqual(result['best_checkpoint_sha256'],comparison.sha(args.output/f'seed{seed}'/arm/'best_checkpoint.pt'))

    def test_prepare_rejects_wrong_selected_labels_even_with_updated_file_hash(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);data=root/'data';source=root/'source';output=root/'out'
            for path in (data,source,output):path.mkdir()
            (source/'DEVELOPMENT_COMPLETE').touch()
            labels=np.repeat(np.arange(12),14);conditions=np.tile(np.array([f'g{k}' for k in range(14)]),12)
            snr=np.tile(np.array(comparison.LEVELS*2),12);snr[labels==11]=np.nan
            ids=np.array([f'id{k}' for k in range(len(labels))]);iq=np.ones((len(labels),32),np.complex64)*.5
            for split in ('train','validation'):
                np.savez(data/(split+'.npz'),condition_id=conditions if split=='train' else np.array(['validation']*len(labels)))
            (data/'metadata.json').write_text('{}')
            indices,_=comparison.select_condition_groups(labels,conditions,snr,1,2080)
            selection=dict(source_id=ids[indices],condition_id=conditions[indices],y=labels[indices].copy(),snr_db=snr[indices])
            selection['y'][0]=(selection['y'][0]+1)%12
            np.savez(source/'training_selection.npz',**selection)
            plan=json.loads(comparison.PLAN.read_text())
            plan['data'].update(metadata_sha256=comparison.sha(data/'metadata.json'),train_sha256=comparison.sha(data/'train.npz'),
                validation_sha256=comparison.sha(data/'validation.npz'),original_selection_sha256=comparison.sha(source/'training_selection.npz'),
                condition_groups_per_snr=1)
            with patch.object(comparison,'load_metadata',return_value={}),patch.object(comparison,'load_split',
                    side_effect=[(iq,labels,ids,snr),(iq,labels,np.array(['v'+s for s in ids]),snr)]):
                with self.assertRaisesRegex(ValueError,'selected rows differ: y'):
                    comparison.prepare(data,source,plan,output)


if __name__=='__main__':unittest.main()
