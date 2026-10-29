import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
import compare_cnn_classical_holdout as compare
from iq_feature_experiment import make_feature_model,ARCHITECTURE,PREPROCESSING
from se_gain_experiment import make_gain_model


class PairedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_paired_counts(self):
        result=compare.paired_counts(np.array([0,1,2,3]),np.array([0,1,0,0]),np.array([0,0,2,0]))
        self.assertEqual(result,dict(count=4,both_correct=1,classical_only_correct=1,cnn_only_correct=1,both_wrong=1,prediction_disagreement=2))

    def test_actual_checkpoint_restore_and_arm_mismatch_rejection(self):
        with tempfile.TemporaryDirectory() as name:
            source=Path(name);folder=source/'seed2081/iq_zero';folder.mkdir(parents=True)
            model=make_feature_model(make_gain_model(2.).state_dict())
            (source/'protocol.json').write_text('{}');(source/'training_selection.npz').write_bytes(b'fixture')
            history=[dict(epoch=k,steps=k*53,validation=dict(loss=float(k))) for k in range(1,101)]
            (folder/'history.json').write_text(json.dumps(history));(folder/'CASE_COMPLETE').touch()
            saved=dict(model=model.state_dict(),model_seed=2081,arm='iq_zero',epoch=1,steps=53,
                protocol_sha256=compare.classical.sha(source/'protocol.json'),selection_sha256=compare.classical.sha(source/'training_selection.npz'),
                architecture=ARCHITECTURE,preprocessing=PREPROCESSING,se_gain=2.,activation='tf-compatible')
            path=folder/'best_checkpoint.pt';torch.save(saved,path)
            expected=dict(best_checkpoint_sha256=compare.classical.sha(path),history_sha256=compare.classical.sha(folder/'history.json'),
                best_epoch=1,selected_state_sha256=compare.state_digest(model.state_dict()))
            restored,_=compare.load_checkpoint(source,'2081','iq_zero',expected)
            self.assertEqual(compare.state_digest(restored.state_dict()),expected['selected_state_sha256'])
            saved['arm']='iq_envelope_phase';torch.save(saved,path);expected['best_checkpoint_sha256']=compare.classical.sha(path)
            with self.assertRaisesRegex(ValueError,'provenance'):compare.load_checkpoint(source,'2081','iq_zero',expected)

    def workflow(self,mismatch=False):
        rng=np.random.default_rng(5);iq=(rng.normal(size=(24,64))+1j*rng.normal(size=(24,64))).astype(np.complex64)
        y=np.tile(np.arange(12),2);snr=np.where(y==11,np.nan,15.)
        arrays=dict(iq=iq,y=y,snr_db=snr,source_id=np.array([str(k) for k in range(24)]))
        initial=make_gain_model(2.).state_dict();model=make_feature_model(initial)
        previous={}
        for seed in ('2081','2082'):
            previous[seed]={}
            for arm in compare.ARMS:
                val=compare.score(model,compare.feature_tensor(iq,arm),torch.from_numpy(y),'cpu')
                if mismatch:val['loss']+=.1
                previous[seed][arm]=dict(selected_validation=val,best_epoch=1)
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);source=root/'source';holdout=root/'holdout';source.mkdir();holdout.mkdir()
            (source/'report.json').write_text('{}');(holdout/'report.json').write_text('{}');(holdout/'holdout.npz').write_bytes(b'fixture')
            path=source/'checkpoint.pt';path.write_bytes(b'fixed checkpoint')
            for cases in previous.values():
                for expected in cases.values():expected['best_checkpoint_sha256']=compare.classical.sha(path)
            def checkpoint(*args):return make_feature_model(initial),path
            args=SimpleNamespace(output=root/'out',data=root,source=source,holdout=holdout,classical_source=None,device='cpu')
            previous_deterministic=torch.are_deterministic_algorithms_enabled()
            try:
                with (patch.object(compare,'load_holdout',return_value=(arrays,y.copy(),compare.classical.evaluation(y,y,snr))),
                      patch.object(compare,'load_cnn_source',return_value=(dict(runs=previous),iq,torch.from_numpy(y))),
                      patch.object(compare,'load_checkpoint',side_effect=checkpoint)):
                    if mismatch:
                        with self.assertRaisesRegex(ValueError,'validation reproduction'):compare.run(args)
                    else:
                        report=compare.run(args)
                        self.assertTrue(report['complete']);self.assertEqual(len(report['cases']),4)
                        for case in report['cases'].values():
                            paired=case['paired'];self.assertEqual(sum(paired[k] for k in ('both_correct','classical_only_correct','cnn_only_correct','both_wrong')),24)
                            self.assertTrue(case['checkpoint_unchanged'])
                            with np.load(args.output/case['prediction_file']) as saved:
                                np.testing.assert_array_equal(saved['source_id'],arrays['source_id'])
            finally:torch.use_deterministic_algorithms(previous_deterministic)
            report=json.loads((args.output/'report.json').read_text())
            self.assertEqual(report['complete'],not mismatch)
            self.assertEqual((args.output/'CNN_CLASSICAL_HOLDOUT_COMPLETE').exists(),not mismatch)

    def test_complete_all_four_actual_inference_cases(self):self.workflow()
    def test_validation_mismatch_retains_failure(self):self.workflow(True)

    def test_real_holdout_readback_and_parameter_tamper_rejection(self):
        from test_classical_holdout import HoldoutTests,fake_generator
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);data,source,plan,versions=HoldoutTests().fixture(root)
            module=compare.holdout_module;original_generate=module.generate_set
            with (patch.object(compare.classical,'PLAN',plan),
                  patch.object(module,'environment',return_value=(versions,'fixture')),
                  patch.object(module,'generate_set',side_effect=lambda out,plan,ids,hashes:original_generate(out,plan,ids,hashes,fake_generator))):
                module.run(data,source,root/'holdout')
                arrays,pred,result=compare.load_holdout(root/'holdout',data)
                self.assertEqual(result['count'],1680);self.assertEqual(len(pred),1680)
                parameter_file=root/'holdout/parameters.jsonl'
                rows=[json.loads(line) for line in parameter_file.read_text().splitlines()]
                rows[0]['label']='fm'
                parameter_file.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                report=json.loads((root/'holdout/report.json').read_text())
                report['artifact_sha256']['parameters.jsonl']=compare.classical.sha(parameter_file)
                (root/'holdout/report.json').write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError,'IQ/label/ID'):
                    compare.load_holdout(root/'holdout',data)


if __name__=='__main__':unittest.main()
