import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import diagnose_training_se_gain as diagnostic


class SEGainDiagnosticTest(unittest.TestCase):
    def test_paired_batches_outputs_and_training_only(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(1)
        iq = (rng.normal(size=(36,64))+1j*rng.normal(size=(36,64))).astype(np.complex64)
        labels = np.repeat(np.arange(12),3).astype(np.int64)
        snr = np.full(36,20.); snr[labels == 11] = np.nan
        with tempfile.TemporaryDirectory() as temporary:
            expected=json.loads(Path(diagnostic.__file__).with_name('SE_GAIN_PLAN_V1.json').read_text())['controls']
            original_sha=diagnostic.sha
            def fixture_sha(path):
                if path.name == 'metadata.json': return expected['metadata_sha256']
                if path.name == 'train.npz': return expected['train_sha256']
                return original_sha(path)
            root = Path(temporary); data = root/'data'; data.mkdir()
            (data/'metadata.json').write_text('{}'); (data/'train.npz').write_bytes(b'test fixture')
            args = argparse.Namespace(data=data,output=root/'output',epochs=1,per_class=3,
                                      device='cpu',activation='tf-compatible')
            with patch.object(diagnostic,'load_metadata',return_value={}), patch.object(
                diagnostic,'load_split',return_value=(iq,labels,np.arange(36),snr)) as loader, patch.object(diagnostic,'sha',side_effect=fixture_sha):
                diagnostic.run(args)
            self.assertEqual(loader.call_args.args[1],'train')
            self.assertEqual(loader.call_count,1)
            report = json.loads((args.output/'report.json').read_text())
            self.assertEqual(report['cases']['gain1']['final']['steps'],1)
            self.assertEqual(report['cases']['gain2']['final']['steps'],1)
            for case in ('gain1','gain2'):
                final = report['cases'][case]['final']
                self.assertEqual(final['examples_processed'],36)
                self.assertEqual(sum(map(sum,final['training']['confusion_matrix'])),36)
                self.assertEqual(report['cases'][case]['initial_sha256'],expected['initial_sha256'])
                self.assertEqual(len(final['batches'][0]['parameters']),32)
                self.assertEqual(sum(sum(b['class_counts']) for b in final['batches']),36)
                self.assertTrue((args.output/case/'last_checkpoint.pt').is_file())
                checkpoint=torch.load(args.output/case/'last_checkpoint.pt',map_location='cpu')
                self.assertEqual(checkpoint['optimizer']['param_groups'][0]['lr'],.0001)
                self.assertIn('largest_loss_increase',report['cases'][case]['trajectory'])
                self.assertEqual(checkpoint['se_gain'],float(case[-1]))
                from se_gain_experiment import restore_gain_checkpoint, ScaledSqueezeExcitation
                restored=restore_gain_checkpoint(checkpoint)
                self.assertTrue(all(m.gain==float(case[-1]) for m in restored.modules() if isinstance(m,ScaledSqueezeExcitation)))
                with self.assertRaisesRegex(ValueError,'tagged'):
                    restore_gain_checkpoint({'model':checkpoint['model']})
                self.assertEqual(len(report['cases'][case]['initial_activation_rms_balanced12']),21)
            self.assertTrue((args.output/'SE_GAIN_DIAGNOSTIC_COMPLETE').is_file())
            with self.assertRaisesRegex(ValueError,'Ausgabe existiert'):
                diagnostic.run(args)


class SEEquationsTest(unittest.TestCase):
    def test_gain1_forward_and_gradients_match_original(self):
        from modulation_reference import Harper2023
        from se_gain_experiment import make_gain_model, set_gain
        from check_modulation_learning import state_digest
        torch.set_num_threads(1); torch.manual_seed(5)
        original=Harper2023(tensorflow_selu=True)
        candidate=make_gain_model(1.)
        candidate.load_state_dict(original.state_dict())
        self.assertEqual(state_digest(original.state_dict()),state_digest(candidate.state_dict()))
        x=torch.randn(2,2,1,4096)
        a=original(x); b=candidate(x)
        self.assertTrue(torch.equal(a,b))
        a.square().sum().backward(); b.square().sum().backward()
        for pa,pb in zip(original.parameters(),candidate.parameters()):
            self.assertTrue(torch.equal(pa.grad,pb.grad))
        digest=state_digest(candidate.state_dict()); set_gain(candidate,2.)
        self.assertEqual(digest,state_digest(candidate.state_dict()))
        self.assertFalse(torch.equal(original(x),candidate(x)))

    def test_block_gain_and_gradients(self):
        from modulation_reference import SqueezeExcitation
        from se_gain_experiment import ScaledSqueezeExcitation
        import copy
        torch.manual_seed(6)
        base=SqueezeExcitation(4); doubled=ScaledSqueezeExcitation(copy.deepcopy(base),2.)
        x=torch.randn(3,4,16,requires_grad=True); xx=x.detach().clone().requires_grad_(True)
        a=base(x); b=doubled(xx)
        self.assertTrue(torch.equal(b,2*a))
        a.sum().backward(); b.sum().backward()
        self.assertTrue(torch.equal(xx.grad,2*x.grad))
        for p,q in zip(base.parameters(),doubled.parameters()):
            self.assertTrue(torch.equal(q.grad,2*p.grad))


if __name__ == '__main__':
    unittest.main()
