import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from checkpoint_radioml import create_checkpoint,verify_archive

class BackupTests(unittest.TestCase):
    def test_roundtrip_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); root=p/'project'; root.mkdir()
            (root/'models').mkdir(); (root/'models/model.tflite').write_bytes(b'\x00model')
            (root/'classes.json').write_text('["I","Q"]')
            source=p/'source.hdf5'; source.write_bytes(b'fixture')
            archive=create_checkpoint(root,p/'backups',source)
            self.assertGreater(verify_archive(archive),3)
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(z.read('project/models/model.tflite'),b'\x00model')
                self.assertEqual(z.read('external-source/source.hdf5'),b'fixture')
                contents={name:z.read(name) for name in z.namelist()}
            contents['project/models/model.tflite']=b'changed'
            bad=p/'bad.zip'
            with zipfile.ZipFile(bad,'w') as z:
                for name,raw in contents.items(): z.writestr(name,raw)
            with self.assertRaises(ValueError): verify_archive(bad)
            with self.assertRaises(ValueError): create_checkpoint(root,root/'backup')

    def test_source_excluded_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); root=p/'project'; root.mkdir()
            archive=create_checkpoint(root,p/'backup')
            with zipfile.ZipFile(archive) as z:
                self.assertFalse(json.loads(z.read('environment.json'))['source_hdf5_included'])

if __name__=='__main__': unittest.main()
