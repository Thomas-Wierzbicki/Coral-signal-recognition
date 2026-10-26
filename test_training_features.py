import unittest
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
import diagnose_training_features as diagnostic
from modulation_reference import Harper2023
from check_modulation_learning import state_digest


class FeatureTests(unittest.TestCase):
    def test_descriptor_uses_population_variance(self):
        z=diagnostic.descriptor(torch.tensor([[[1.,3.],[2.,6.]]]))
        np.testing.assert_allclose(z.numpy(),[[2.,4.,1.,4.]])

    def test_leave_one_out_excludes_self(self):
        labels=np.repeat(np.arange(12),2)
        z=np.repeat(np.arange(12)*100.,2)[:,None]
        result=diagnostic.separation(z,labels)
        self.assertEqual(result['loo_centroid']['accuracy'],1.)
        self.assertIsNone(result['between_within_ratio'])
        # Two distant members of class 0: using self would spuriously help.
        z[0]=0.; z[1]=10000.
        result=diagnostic.separation(z,labels)
        self.assertLess(result['loo_centroid']['recall']['am-dsb'],1.)

    def test_read_only_hooks_full_length_and_repeated_inference(self):
        torch.set_num_threads(1); torch.manual_seed(2)
        model=Harper2023(tensorflow_selu=True)
        x=torch.randn(24,2,1,4096); labels=np.repeat(np.arange(12),2)
        initial=state_digest(model.state_dict())
        with torch.inference_mode(): expected=model(x).argmax(1).numpy()
        result,arrays=diagnostic.inspect(model,x,labels,'cpu')
        self.assertEqual(state_digest(model.state_dict()),initial)
        self.assertEqual(result['model_predictions']['confusion_matrix'],diagnostic.metrics(labels,expected)['confusion_matrix'])
        self.assertEqual(len(result['layers']),26)
        self.assertEqual(arrays['pool'].shape,(24,216))
        self.assertEqual(sum('gate' in info for info in result['layers'].values()),6)
        self.assertTrue(all(len(module._forward_hooks)==0 for module in model.modules()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))


    def test_complete_file_workflow_and_hash_rejection(self):
        torch.set_num_threads(1); torch.manual_seed(3)
        model=Harper2023(tensorflow_selu=True)
        with torch.no_grad(): model.head[4].bias[0]=1.
        state=model.state_dict(); digest=state_digest(state)
        labels=np.repeat(np.arange(12),10)
        rng=np.random.default_rng(4)
        iq=(rng.normal(size=(120,4096))+1j*rng.normal(size=(120,4096))).astype(np.complex64)
        expected=diagnostic.metrics(labels,np.zeros(120,dtype=np.int64))
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); source=root/'source'; source.mkdir()
            (source/'LR_DIAGNOSTIC_COMPLETE').write_text('OK')
            np.savez_compressed(source/'selection.npz',iq=iq,y=labels,source_id=np.arange(120))
            torch.save(state,source/'initial_state.pt')
            protocol=dict(default_design=True,activation='tf-compatible',
                selection_sha256=diagnostic.sha(source/'selection.npz'),initial_sha256=digest,
                sources={name:diagnostic.sha(Path(diagnostic.__file__).with_name(name))
                         for name in ['modulation_reference.py','train_modulation12.py']})
            cases={}
            for case in ['lr0.001','lr0.0001']:
                folder=source/case; folder.mkdir()
                torch.save(dict(model=state),folder/'last_checkpoint.pt')
                cases[case]=dict(checkpoint_sha256=diagnostic.sha(folder/'last_checkpoint.pt'),
                                 final_sha256=digest,final=dict(training=expected))
            diagnostic.write_json(source/'report.json',dict(protocol=protocol,cases=cases))
            args=argparse.Namespace(run=source,output=root/'output',device='cpu')
            diagnostic.run(args)
            result=json.loads((args.output/'report.json').read_text())
            self.assertEqual(len(result['cases']),3)
            self.assertTrue(result['cases']['lr0.001']['prediction_matrix_matches_training_report'])
            self.assertTrue((args.output/'FEATURE_DIAGNOSTIC_COMPLETE').is_file())
            with self.assertRaisesRegex(ValueError,'Output exists'): diagnostic.run(args)
            with (source/'selection.npz').open('ab') as file: file.write(b'tampered')
            args.output=root/'rejected'
            with self.assertRaisesRegex(ValueError,'Selection hash mismatch'): diagnostic.run(args)
            self.assertFalse(args.output.exists())


if __name__=='__main__': unittest.main()
