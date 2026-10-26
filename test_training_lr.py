import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import diagnose_training_lr as diagnostic


class LearningRateDiagnosticTest(unittest.TestCase):
    def test_paired_batches_outputs_and_training_only(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(1)
        iq = (rng.normal(size=(36,64))+1j*rng.normal(size=(36,64))).astype(np.complex64)
        labels = np.repeat(np.arange(12),3).astype(np.int64)
        snr = np.full(36,20.); snr[labels == 11] = np.nan
        with tempfile.TemporaryDirectory() as temporary:
            expected=json.loads(Path(diagnostic.__file__).with_name('TRAINING_LR_PLAN_V1.json').read_text())['controls']
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
            self.assertEqual(report['cases']['lr0.001']['final']['steps'],1)
            self.assertEqual(report['cases']['lr0.0001']['final']['steps'],1)
            for case in ('lr0.001','lr0.0001'):
                final = report['cases'][case]['final']
                self.assertEqual(final['examples_processed'],36)
                self.assertEqual(sum(map(sum,final['training']['confusion_matrix'])),36)
                self.assertEqual(report['cases'][case]['initial_sha256'],expected['initial_sha256'])
                self.assertEqual(len(final['batches'][0]['parameters']),32)
                self.assertEqual(sum(sum(b['class_counts']) for b in final['batches']),36)
                self.assertTrue((args.output/case/'last_checkpoint.pt').is_file())
                checkpoint=torch.load(args.output/case/'last_checkpoint.pt',map_location='cpu')
                self.assertEqual(checkpoint['optimizer']['param_groups'][0]['lr'],float(case[2:]))
                self.assertIn('largest_loss_increase',report['cases'][case]['trajectory'])
            self.assertTrue((args.output/'LR_DIAGNOSTIC_COMPLETE').is_file())
            with self.assertRaisesRegex(ValueError,'Ausgabe existiert'):
                diagnostic.run(args)


if __name__ == '__main__':
    unittest.main()
