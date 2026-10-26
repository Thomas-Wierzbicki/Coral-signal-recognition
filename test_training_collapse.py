import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import diagnose_training_collapse as diagnostic


class TrainingDiagnosticTest(unittest.TestCase):
    def test_paired_batches_outputs_and_training_only(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(1)
        iq = (rng.normal(size=(36,64))+1j*rng.normal(size=(36,64))).astype(np.complex64)
        labels = np.repeat(np.arange(12),3).astype(np.int64)
        snr = np.full(36,20.); snr[labels == 11] = np.nan
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); data = root/'data'; data.mkdir()
            (data/'metadata.json').write_text('{}'); (data/'train.npz').write_bytes(b'test fixture')
            args = argparse.Namespace(data=data,output=root/'output',epochs=1,per_class=3,
                                      device='cpu',activation='tf-compatible')
            with patch.object(diagnostic,'load_metadata',return_value={}), patch.object(
                diagnostic,'load_split',return_value=(iq,labels,np.arange(36),snr)) as loader:
                diagnostic.run(args)
            self.assertEqual(loader.call_args.args[1],'train')
            self.assertEqual(loader.call_count,1)
            report = json.loads((args.output/'report.json').read_text())
            self.assertEqual(report['cases']['full']['final']['steps'],1)
            self.assertEqual(report['cases']['mini32']['final']['steps'],2)
            for case in ('full','mini32'):
                final = report['cases'][case]['final']
                self.assertEqual(final['examples_processed'],36)
                self.assertEqual(sum(map(sum,final['training']['confusion_matrix'])),36)
                self.assertEqual(len(final['batches'][0]['parameters']),32)
                self.assertEqual(sum(sum(b['class_counts']) for b in final['batches']),36)
                self.assertTrue((args.output/case/'last_checkpoint.pt').is_file())
            self.assertTrue((args.output/'DIAGNOSTIC_COMPLETE').is_file())
            with self.assertRaisesRegex(ValueError,'Ausgabe existiert'):
                diagnostic.run(args)


if __name__ == '__main__':
    unittest.main()
