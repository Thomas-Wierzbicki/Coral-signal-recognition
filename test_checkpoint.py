"""Prueft die echte Archivierung in einem temporaeren Git-Repository."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from checkpoint_rms import create_checkpoint, validate_rms_artifacts


class CheckpointTests(unittest.TestCase):
    def test_backup_contains_ignored_models_json_and_git_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"project"
            root.mkdir()
            subprocess.run(["git","init","-q",str(root)],check=True)
            (root/"README.md").write_text("Test snapshot",encoding="utf-8")
            subprocess.run(["git","-C",str(root),"add","README.md"],check=True)
            subprocess.run(["git","-C",str(root),"-c","user.name=Test",
                            "-c","user.email=test@example.invalid","commit","-qm","Fixture"],check=True)
            folder=root/"models/signal-rms/compiled"
            folder.mkdir(parents=True)
            payload=b"model-fixture\x00\xff"
            (folder/"signal_int8_edgetpu.tflite").write_bytes(payload)
            (folder/"preprocessing.json").write_text('{"mode":"rms-v1"}')
            with patch("checkpoint_rms.environment_info",return_value={"test":True}):
                archive=create_checkpoint(root,Path(directory)/"backups")
            self.assertTrue(archive.is_file())
            self.assertFalse(archive.with_suffix(".zip.partial").exists())
            with zipfile.ZipFile(archive) as z:
                checks=json.loads(z.read("SHA256SUMS.json"))
                name="project/models/signal-rms/compiled/signal_int8_edgetpu.tflite"
                self.assertEqual(z.read(name),payload)
                self.assertEqual(checks[name],hashlib.sha256(payload).hexdigest())
                self.assertIn("project/models/signal-rms/compiled/preprocessing.json",checks)
                self.assertIn("repository.bundle",checks)
                self.assertIn("environment.json",checks)
                z.extract("repository.bundle",directory)
            subprocess.run(["git","-C",str(root),"bundle","verify",
                            str(Path(directory)/"repository.bundle")],
                           check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    def test_missing_model_stops_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                validate_rms_artifacts(Path(directory))
    def test_archive_cannot_be_created_inside_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaises(ValueError):
                create_checkpoint(root,root/"backups")


if __name__=="__main__":
    unittest.main()
