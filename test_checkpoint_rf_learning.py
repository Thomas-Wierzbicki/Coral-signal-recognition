import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile
import checkpoint_rf_learning as backup


class BackupTests(unittest.TestCase):
    def test_roundtrip_and_manifest_detects_extra_entry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'rf-learning';output=Path(temporary)/'backups'
            for name in ['project','models','data','.venv','.git']:(root/name).mkdir(parents=True)
            (root/'project/code.py').write_text('print(1)')
            (root/'models/weights.pt').write_bytes(b'weights')
            (root/'data/iq.npz').write_bytes(b'iq')
            (root/'.venv/excluded').write_text('excluded')
            (root/'.git/excluded').write_text('excluded')
            archive=backup.create_backup(root,output)
            self.assertEqual(backup.verify_archive(archive),5)
            self.assertEqual(archive.with_suffix('.zip.sha256').read_text().split()[0],backup.digest(archive))
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(z.read('rf-learning/models/weights.pt'),b'weights')
                self.assertFalse(any('.venv/' in name or '.git/' in name for name in z.namelist()))
            with zipfile.ZipFile(archive,'a') as z:z.writestr('unexpected',b'bad')
            with self.assertRaisesRegex(ValueError,'manifest'):backup.verify_archive(archive)
            with self.assertRaisesRegex(ValueError,'outside'):backup.create_backup(root,root/'backup')

    def test_changed_source_never_becomes_final_backup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'rf-learning';output=Path(temporary)/'backups'
            for name in ['project','models','data']:(root/name).mkdir(parents=True)
            source=root/'project/a.py';source.write_text('before')
            verify=backup.verify_archive
            def change(path):
                result=verify(path);source.write_text('after change');return result
            with patch.object(backup,'verify_archive',side_effect=change):
                with self.assertRaisesRegex(ValueError,'changed'):backup.create_backup(root,output)
            self.assertEqual(list(output.glob('*.zip')),[])
            self.assertEqual(len(list(output.glob('*.partial'))),1)


if __name__=='__main__':unittest.main()
