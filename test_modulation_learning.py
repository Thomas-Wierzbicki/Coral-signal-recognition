import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from check_modulation_learning import select_examples,fit_case,state_digest
from torchsig_probe import CLASSES

class LearningTests(unittest.TestCase):
    def test_selection_is_balanced_high_snr_and_deterministic(self):
        y=np.repeat(np.arange(12),6)
        iq=np.arange(len(y)*4).reshape(len(y),4).astype(np.complex64)
        ids=np.array([str(k) for k in range(len(y))])
        snr=np.tile(np.array([-5,0,10,15,20,25.]),12); snr[y==11]=np.nan
        a=select_examples(iq,y,ids,snr); b=select_examples(iq,y,ids,snr)
        self.assertEqual(len(a[1]),24)
        np.testing.assert_array_equal(np.bincount(a[1]),np.full(12,2))
        np.testing.assert_array_equal(a[2],b[2])
        self.assertTrue((a[3][a[1]!=11]>=15).all())
        self.assertTrue(np.isnan(a[3][a[1]==11]).all())
        self.assertEqual(len(set(a[2])),24)

    def test_training_loop_learns_and_records_updates(self):
        torch.manual_seed(0)
        model=torch.nn.Linear(12,12)
        torch.nn.init.zeros_(model.bias)
        x=torch.eye(12); y=torch.arange(12)
        before=state_digest(model.state_dict())
        with tempfile.TemporaryDirectory() as d:
            result=fit_case(model,x,y,100,.1,10,Path(d))
            self.assertTrue(result['memorization_pass'])
            self.assertLess(result['final']['loss'],result['initial']['loss'])
            import json
            rows=json.loads((Path(d)/'history.json').read_text())
            self.assertGreater(rows[0]['parameters']['weight']['update_l2'],0)
            self.assertIsNone(rows[0]['parameters']['bias']['relative_update'])
        self.assertNotEqual(before,state_digest(model.state_dict()))

    def test_state_digest_ignores_mapping_order_and_detects_weight_change(self):
        a={'a':torch.tensor([1.,2.]),'b':torch.tensor([3.])}
        b={'b':a['b'].clone(),'a':a['a'].clone()}
        self.assertEqual(state_digest(a),state_digest(b))
        b['a'][0]+=1
        self.assertNotEqual(state_digest(a),state_digest(b))

if __name__=='__main__': unittest.main()
