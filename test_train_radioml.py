"""Datenvertrag, Quellzeilentrennung und Kennzahlen ohne TensorFlow testen."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from train_radioml import load_dataset,score


class RadioTrainingTests(unittest.TestCase):
    def fixture(self,root):
        meta=dict(source="RadioML2018.01A",preprocessing="rms-v1",target_rms=0.5,
                  input_shape=[29,128,1],fft_size=128,hop=32,
                  labels=["AM-DSB-WC","FM","BPSK","QPSK","16QAM"])
        (root/"metadata.json").write_text(json.dumps(meta))
        (root/"IMPORT_COMPLETE").write_text("OK")
        for i,name in enumerate(("train","validation","test")):
            np.savez_compressed(root/(name+".npz"),x=np.zeros((10,29,128,1),np.float32),
                                y=np.tile(np.arange(5),2),snr_db=np.zeros(10),
                                source_index=np.arange(i*10,(i+1)*10))
    def test_valid_and_overlapping_source_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.fixture(root)
            metadata,splits=load_dataset(root)
            self.assertEqual(splits["test"][0].shape,(10,29,128,1))
            with np.load(root/"test.npz") as data:
                arrays={k:data[k] for k in data.files}
            arrays["source_index"][0]=0
            np.savez_compressed(root/"test.npz",**arrays)
            with self.assertRaises(ValueError):
                load_dataset(root)
    def test_missing_completion_and_wrong_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.fixture(root)
            path=root/"metadata.json"
            meta=json.loads(path.read_text())
            meta["preprocessing"]="raw"
            path.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                load_dataset(root)
            (root/"IMPORT_COMPLETE").unlink()
            with self.assertRaises(ValueError):
                load_dataset(root)
    def test_confusion_matrix_class_order(self):
        labels=["AM-DSB-WC","FM","BPSK","QPSK","16QAM"]
        result=score(np.arange(5),np.array([0,2,1,3,4]),labels)
        self.assertAlmostEqual(result["accuracy"],0.6)
        self.assertEqual(result["recall"]["FM"],0)
        self.assertEqual(result["confusion_matrix"][1][2],1)


if __name__=="__main__":
    unittest.main()
