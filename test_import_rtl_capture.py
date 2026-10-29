import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from import_rtl_capture import import_capture
from inspect_rtl_iq import inspect


class ImportTests(unittest.TestCase):
    def test_copy_hash_and_downstream_inspection(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);source=root/'external.cu8'
            raw=np.random.default_rng(9).integers(40,210,size=8192,dtype=np.uint8);source.write_bytes(raw.tobytes())
            meta=import_capture(source,root/'out',433675000,1024000,4096/1024000,20.7)
            self.assertTrue(meta['complete']);self.assertEqual(source.read_bytes(),(root/'out/capture.cu8').read_bytes())
            self.assertEqual(meta['reference_label'],'unknown')
            result=inspect(root/'out');self.assertEqual(result['samples'],4096)
            self.assertEqual(result['adc_rail_fraction'],0.)

    def test_bad_length_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);source=root/'external.cu8';source.write_bytes(b'bad')
            with self.assertRaises(ValueError):import_capture(source,root/'out',433675000)
            self.assertFalse((root/'out').exists())

    def test_existing_destination_not_overwritten(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);source=root/'external.cu8';source.write_bytes(bytes(2048))
            (root/'out').mkdir();(root/'out/keep.txt').write_text('keep')
            with self.assertRaises(FileExistsError):import_capture(source,root/'out',433675000,1024000,1024/1024000)
            self.assertEqual((root/'out/keep.txt').read_text(),'keep')


if __name__=='__main__':unittest.main()
