import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import train_se_gain_development as development
from check_modulation_learning import state_digest


def fixture(prefix,groups_per_snr=2):
    labels=[];conditions=[];snr=[]
    for level in development.LEVELS:
        for k in range(groups_per_snr):
            for cid in range(12):
                labels.append(cid);conditions.append(f'{prefix}:{level}:{k}')
                snr.append(level if cid!=11 else np.nan)
    return np.asarray(labels,dtype=np.int64),np.asarray(conditions),np.asarray(snr)


class DevelopmentTests(unittest.TestCase):
    def test_grouped_selection_balances_every_snr_and_class(self):
        labels,conditions,snr=fixture('train',3)
        a,groups=development.select_condition_groups(labels,conditions,snr,2)
        b,_=development.select_condition_groups(labels,conditions,snr,2)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(np.bincount(labels[a]),np.full(12,14))
        self.assertEqual(len(groups),14)
        for level in development.LEVELS:
            self.assertEqual(int((snr[a]==level).sum()),22)
        for group in groups:
            self.assertEqual(sorted(labels[a][conditions[a]==group]),list(range(12)))
        broken=snr.copy();broken[0]=0.
        with self.assertRaisesRegex(ValueError,'Inconsistent'):development.select_condition_groups(labels,conditions,broken,2)
        with self.assertRaisesRegex(ValueError,'each class'):development.select_condition_groups(labels[1:],conditions[1:],snr[1:],2)

    def test_larger_workflow_selects_best_validation_not_last_and_never_test(self):
        torch.set_num_threads(1);rng=np.random.default_rng(1)
        ty,tc,ts=fixture('train',2);vy,vc,vs=fixture('val',1)
        def iq(n):return (rng.normal(size=(n,64))+1j*rng.normal(size=(n,64))).astype(np.complex64)
        train=(iq(len(ty)),ty,np.asarray(['t'+str(i) for i in range(len(ty))]),ts)
        val=(iq(len(vy)),vy,np.asarray(['v'+str(i) for i in range(len(vy))]),vs)
        expected=json.loads(Path(development.__file__).with_name('SE_GAIN_DEVELOPMENT_PLAN_V1.json').read_text())
        real_sha=development.sha;real_score=development.score
        captured=[]
        def controlled_score(model,x,y,device):
            # Deliberately make epoch1 better than epoch2, independent of optimizer noise.
            position=len(captured)%4
            result=real_score(model,x,y,device)
            result['loss']=[1.,2.,3.,1.][position]
            captured.append(state_digest(model.state_dict()))
            return result
        def file_hash(path):
            if path.name=='metadata.json':return expected['controls']['metadata_sha256']
            if path.name=='train.npz':return expected['controls']['train_sha256']
            return real_sha(path)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);data=root/'data';data.mkdir()
            (data/'metadata.json').write_text('{}')
            np.savez(data/'train.npz',condition_id=tc);np.savez(data/'validation.npz',condition_id=vc)
            args=argparse.Namespace(data=data,output=root/'output',device='cpu',epochs=2,per_snr=1)
            def split(folder,name,meta,seen):
                self.assertIn(name,['train','validation'])
                return train if name=='train' else val
            with patch.object(development,'load_metadata',return_value={}),patch.object(
                development,'load_split',side_effect=split) as loader,patch.object(
                development,'sha',side_effect=file_hash),patch.object(development,'score',side_effect=controlled_score):
                development.run(args)
            self.assertEqual([c.args[1] for c in loader.call_args_list],['train','validation'])
            self.assertFalse((data/'test.npz').exists())
            report=json.loads((args.output/'report.json').read_text())
            self.assertEqual(report['protocol']['train_examples'],84)
            self.assertFalse(report['protocol']['default_design'])
            self.assertFalse(report['protocol']['test_opened'])
            case_index=0
            for seed,cases in report['runs'].items():
                self.assertEqual(cases['gain1']['initial_sha256'],cases['gain2']['initial_sha256'])
                for name,result in cases.items():
                    self.assertEqual(result['best_epoch'],1);self.assertEqual(result['steps'],6)
                    self.assertEqual(result['selected_validation']['loss'],1.)
                    self.assertEqual(result['final_validation']['loss'],2.)
                    folder=args.output/('seed'+seed)/name
                    best=torch.load(folder/'best_checkpoint.pt',weights_only=True)
                    last=torch.load(folder/'last_checkpoint.pt',weights_only=True)
                    self.assertEqual(state_digest(best['model']),captured[case_index*4])
                    self.assertEqual(state_digest(last['model']),captured[case_index*4+1])
                    self.assertNotEqual(state_digest(best['model']),state_digest(last['model']))
                    self.assertTrue((folder/'CASE_COMPLETE').is_file());case_index+=1
            self.assertEqual(case_index,4)
            self.assertTrue(report['summary']['complete'])
            self.assertFalse(report['summary']['directional_development_advantage'])
            self.assertTrue((args.output/'DEVELOPMENT_COMPLETE').is_file())
            with self.assertRaisesRegex(ValueError,'Output exists'):development.run(args)
            # Cross-split nuisance group overlap must fail before output creation.
            np.savez(data/'validation.npz',condition_id=tc[:len(vy)])
            args.output=root/'overlap'
            with patch.object(development,'load_metadata',return_value={}),patch.object(
                development,'load_split',side_effect=split),patch.object(development,'sha',side_effect=file_hash):
                with self.assertRaisesRegex(ValueError,'overlap'):development.run(args)
            self.assertFalse(args.output.exists())


if __name__=='__main__':unittest.main()
