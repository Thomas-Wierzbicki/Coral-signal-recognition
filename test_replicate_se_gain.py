import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import replicate_se_gain as replication


class ReplicationTests(unittest.TestCase):
    def test_both_seeds_required_and_mixed_not_success(self):
        def report(a,b):
            return {'cases':{'gain1':{'final':{'training':{'accuracy':a,'loss':1.}}},
                             'gain2':{'final':{'training':{'accuracy':b,'loss':.5}}}}}
        good=report(.5,.6); tie=report(.5,.5)
        self.assertIsNone(replication.summarize({'2071':good})['directional_replication'])
        self.assertTrue(replication.summarize({'2071':good,'2072':good})['directional_replication'])
        self.assertFalse(replication.summarize({'2071':good,'2072':tie})['directional_replication'])

    def test_four_real_optimization_cases_and_seed_pairing(self):
        torch.set_num_threads(1)
        rng=np.random.default_rng(9)
        iq=(rng.normal(size=(36,64))+1j*rng.normal(size=(36,64))).astype(np.complex64)
        labels=np.repeat(np.arange(12),3).astype(np.int64)
        snr=np.full(36,20.); snr[labels==11]=np.nan
        plan=json.loads(Path(replication.__file__).with_name('SE_GAIN_REPLICATION_PLAN_V1.json').read_text())
        engine=replication.engine; original_sha=engine.sha
        def fixture_sha(path):
            if path.name=='metadata.json':return plan['controls']['metadata_sha256']
            if path.name=='train.npz':return plan['controls']['train_sha256']
            return original_sha(path)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); data=root/'data'; data.mkdir()
            (data/'metadata.json').write_text('{}');(data/'train.npz').write_bytes(b'fixture')
            args=argparse.Namespace(data=data,output=root/'output',device='cpu',epochs=1,per_class=3)
            with patch.object(engine,'load_metadata',return_value={}),patch.object(
                engine,'load_split',return_value=(iq,labels,np.arange(36),snr)) as loader,patch.object(
                engine,'sha',side_effect=fixture_sha):
                replication.run(args)
            self.assertEqual(loader.call_count,2)
            self.assertTrue(all(c.args[1]=='train' for c in loader.call_args_list))
            report=json.loads((args.output/'report.json').read_text())
            self.assertEqual(set(report['runs']),{'2071','2072'})
            initial=[]
            for seed,r in report['runs'].items():
                self.assertEqual(r['protocol']['model_seed'],int(seed))
                a,b=(r['cases'][name] for name in ['gain1','gain2'])
                self.assertEqual(a['initial_sha256'],b['initial_sha256'])
                self.assertEqual(a['initial_sha256'],plan['controls']['initial_sha256_by_seed'][seed])
                initial.append(a['initial_sha256'])
                self.assertEqual(a['final']['steps'],1);self.assertEqual(b['final']['steps'],1)
            self.assertNotEqual(*initial)
            self.assertFalse(report['protocol']['default_design'])
            self.assertTrue((args.output/'SE_GAIN_REPLICATION_COMPLETE').is_file())
            with self.assertRaisesRegex(ValueError,'Ausgabe existiert'):replication.run(args)


if __name__=='__main__':unittest.main()
