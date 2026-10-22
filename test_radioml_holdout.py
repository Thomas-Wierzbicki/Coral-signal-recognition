import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from radioml_holdout import file_hash,load_holdout
from radioml_holdout_import import choose_new_rows

class HoldoutTests(unittest.TestCase):
    def test_selection_excludes_all_splits_and_is_repeatable(self):
        classes=np.repeat(np.arange(5),24)
        snr=np.tile(np.repeat([0.,2.],12),5)
        excluded=np.arange(0,120,6)
        def scan(_):
            for start in range(0,120,17):
                yield start,classes[start:start+17],snr[start:start+17]
        first=choose_new_rows(None,list(range(5)),[0.,2.],excluded,5,2032,scan)
        self.assertEqual(first,choose_new_rows(None,list(range(5)),[0.,2.],excluded,5,2032,scan))
        self.assertEqual(len(first),50)
        self.assertFalse(set(r[0] for r in first)&set(excluded))
        for label in range(5):
            for level in (0.,2.):
                self.assertEqual(sum(r[1:]==(label,level) for r in first),5)
        with self.assertRaises(ValueError):
            choose_new_rows(None,list(range(5)),[0.,2.],excluded,20,2032,scan)

    def test_loader_rejects_overlap_and_changed_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); ref=p/'ref'; ref.mkdir(); new=p/'new'; new.mkdir()
            meta=dict(source='RadioML2018.01A',preprocessing='rms-v1',target_rms=0.5,
                representation='iq-rms-v1',input_shape=[1,1024,2],labels=list('ABCDE'),
                selected_y_columns=[19,21,3,4,12],snr_values=[0.],per_class_per_snr=1)
            (ref/'metadata.json').write_text(json.dumps(meta)); (ref/'IMPORT_COMPLETE').touch()
            for k,name in enumerate(('train','validation','test')):
                np.savez(ref/(name+'.npz'),x=np.zeros((5,1,1024,2)),y=np.arange(5),
                         snr_db=np.zeros(5),source_index=np.arange(5)+5*k)
            model=p/'model.tflite'; model.write_bytes(b'frozen')
            meta.update(purpose='holdout-only',frozen_model_sha256=file_hash(model))
            (new/'metadata.json').write_text(json.dumps(meta)); (new/'IMPORT_COMPLETE').touch()
            def save(ids):
                np.savez(new/'test.npz',x=np.zeros((5,1,1024,2)),y=np.arange(5),
                         snr_db=np.zeros(5),source_index=ids)
            save(np.arange(20,25)); self.assertEqual(len(load_holdout(new,ref,model)[2]),5)
            save(np.arange(5))
            with self.assertRaises(ValueError): load_holdout(new,ref,model)
            save(np.arange(20,25)); model.write_bytes(b'changed')
            with self.assertRaises(ValueError): load_holdout(new,ref,model)

if __name__=='__main__': unittest.main()
