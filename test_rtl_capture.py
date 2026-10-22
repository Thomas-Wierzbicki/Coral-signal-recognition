import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from capture_rtl import capture
from inspect_rtl_iq import decode_cu8,inspect
from radioml_holdout import file_hash

class CaptureTests(unittest.TestCase):
    def test_decode_and_quality(self):
        np.testing.assert_allclose(decode_cu8(np.array([0,255,255,0],dtype=np.uint8)),[-1+1j,1-1j])
        with self.assertRaises(ValueError): decode_cu8(np.array([0],dtype=np.uint8))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); n=8192; t=np.arange(n)
            iq=.4*np.exp(2j*np.pi*64/1024*t)
            pairs=np.stack((iq.real,iq.imag),axis=-1)
            raw=np.clip(np.rint(pairs*127.5+127.5),0,255).astype(np.uint8)
            raw.tofile(p/'capture.cu8')
            meta=dict(complete=True,format='cu8',channel_order=['I','Q'],actual_samples=n,
                raw_sha256=file_hash(p/'capture.cu8'),sample_rate_hz_requested=1024000,
                center_frequency_hz_requested=100000000)
            (p/'metadata.json').write_text(json.dumps(meta)); (p/'CAPTURE_COMPLETE').touch()
            r=inspect(p)
            self.assertEqual(r['preview_peak_offset_hz'],64000)
            self.assertAlmostEqual(r['rms'],.4,places=2)
            (p/'capture.cu8').write_bytes(b'changed')
            with self.assertRaises(ValueError): inspect(p)

    def test_recorder_command_and_incomplete_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); exe=p/'rtl_sdr.exe'; exe.write_bytes(b'fake recorder')
            class Process:
                def __init__(self,command,**kwargs):
                    self.command=command
                    count=int(command[command.index('-n')+1])
                    Path(command[-1]).write_bytes(b'\x80\x80'*count)
                def wait(self,**kwargs): return 0
            with patch('capture_rtl.subprocess.Popen',Process):
                out=capture(str(exe),p/'good',100000000,1024000,.001,20.7,0,'0','unknown')
                self.assertTrue((out/'CAPTURE_COMPLETE').exists())
            class Short(Process):
                def __init__(self,command,**kwargs): Path(command[-1]).write_bytes(b'\x80\x80')
            with patch('capture_rtl.subprocess.Popen',Short):
                with self.assertRaises(ValueError):
                    capture(str(exe),p/'short',100000000,1024000,.001,20.7,0,'0','unknown')
            self.assertFalse((p/'short/CAPTURE_COMPLETE').exists())
            self.assertFalse(json.loads((p/'short/metadata.json').read_text())['complete'])

if __name__=='__main__': unittest.main()
