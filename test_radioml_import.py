"""RadioML-Import mit kleinen kuenstlichen HDF5-Dateien pruefen."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
from radioml_import import load_classes,scan,prepare,radio_features


class RadioImportTests(unittest.TestCase):
    def fixture(self,root):
        labels=["AM-DSB-WC","FM"]+["other"+str(i) for i in range(22)]
        mapping=root/"classes.json"
        mapping.write_text(json.dumps(labels))
        path=root/"fixture.hdf5"
        y=np.repeat(np.arange(24),40)
        snr=np.tile(np.repeat([0.,10.],20),24)
        with h5py.File(path,"w") as f:
            f["X"]=np.random.default_rng(1).normal(size=(len(y),1024,2)).astype(np.float32)
            f["Y"]=np.eye(24,dtype=np.float32)[y]
            f["Z"]=snr[:,None]
        return path,mapping
    def test_stratified_disjoint_rows_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            source,mapping=self.fixture(root)
            output=root/"prepared"
            with h5py.File(source,"r") as f,redirect_stdout(io.StringIO()):
                prepare(f,source,mapping,["FM","AM-DSB-WC"],0,10,20,2030,output)
            all_indices=[]
            for split,n in (("train",56),("validation",12),("test",12)):
                with np.load(output/(split+".npz")) as d:
                    self.assertEqual(d["x"].shape,(n,29,128,1))
                    self.assertTrue(np.isfinite(d["x"]).all())
                    self.assertEqual(set(d["y"]),{0,1})
                    self.assertEqual(set(d["snr_db"]),{0,10})
                    all_indices.extend(d["source_index"].tolist())
            self.assertEqual(len(all_indices),len(set(all_indices)))
            metadata=json.loads((output/"metadata.json").read_text())
            self.assertEqual(metadata["selected_y_columns"],[1,0])
            self.assertIsNone(metadata["sampling_rate_hz"])
            self.assertTrue((output/"IMPORT_COMPLETE").is_file())
            with h5py.File(source,"r") as f:
                with self.assertRaises(ValueError):
                    prepare(f,source,mapping,["FM"],0,10,20,2030,output)
    def test_corrupt_one_hot_and_invalid_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            source,mapping=self.fixture(root)
            with h5py.File(source,"r+") as f:
                f["Y"][0,1]=1
                with self.assertRaises(ValueError):
                    list(scan(f,chunk_size=23))
            mapping.write_text(json.dumps(["same"]*24))
            with self.assertRaises(ValueError):
                load_classes(mapping)
    def test_features_use_1024_samples_and_rms(self):
        x=np.random.default_rng(4).normal(size=(1024,2)).astype(np.float32)
        a=radio_features(x)
        self.assertEqual(a.shape,(29,128,1))
        np.testing.assert_allclose(a,radio_features(x*2),atol=1e-6)
        with self.assertRaises(ValueError):
            radio_features(x[:128])


if __name__=="__main__":
    unittest.main()
