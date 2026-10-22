import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from classify_rtl_channel import prepare_block,run

class FakeInterpreter:
    def __init__(self,*args): self.calls=0
    def allocate_tensors(self): pass
    def get_input_details(self):
        return [dict(shape=[1,1,1024,2],dtype=np.uint8,quantization=(.01,128),index=0)]
    def get_output_details(self):
        return [dict(shape=[1,5],dtype=np.uint8,quantization=(1/256,0),index=1)]
    def set_tensor(self,index,value):
        assert value.shape==(1,1,1024,2) and value.dtype==np.uint8
    def invoke(self): self.calls+=1
    def get_tensor(self,index): return np.array([[0,250,1,2,3]],dtype=np.uint8)

class ClassificationTests(unittest.TestCase):
    def test_rms_and_channel_order(self):
        iq=np.exp(2j*np.pi*np.arange(1024)/32).astype(np.complex64)
        x,rms=prepare_block(iq*7)
        self.assertEqual(x.shape,(1,1024,2)); self.assertAlmostEqual(rms,7,places=5)
        np.testing.assert_allclose(x[0,:,0],.5*iq.real,atol=1e-7)
        np.testing.assert_allclose(x[0,:,1],.5*iq.imag,atol=1e-7)
        with self.assertRaises(ValueError): prepare_block(np.zeros(1024,dtype=np.complex64))
        with self.assertRaises(ValueError): prepare_block(np.full(1024,np.nan))

    def test_full_run_csv_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); channel=p/'channel'; channel.mkdir(); modeldir=p/'model'; modeldir.mkdir()
            compiled=modeldir/'compiled'; compiled.mkdir(); model=compiled/'model.tflite'; model.write_bytes(b'fake')
            labels=['AM-DSB-WC','FM','BPSK','QPSK','16QAM']
            (modeldir/'preprocessing.json').write_text(json.dumps(dict(representation='iq-rms-v1',
                input_shape=[1,1024,2],labels=labels,channel_order=['I','Q'],target_rms=.5,mode='rms-v1')))
            n=33+3*1024+7
            np.exp(2j*np.pi*np.arange(n)/32).astype('<c8').tofile(channel/'channel.cf32')
            (channel/'channel_report.json').write_text(json.dumps(dict(output_rate_hz_nominal=32000,
                format='little-endian complex64 interleaved I,Q',samples=n,statistic_start_sample=33)))
            (channel/'CHANNEL_COMPLETE').touch()
            fake=FakeInterpreter()
            report=run(channel,model,p/'result',interpreter_factory=lambda _:fake)
            self.assertEqual(report['blocks'],3); self.assertEqual(report['tail_samples_omitted'],7)
            self.assertEqual(report['predictions']['FM'],3); self.assertEqual(fake.calls,4)
            with (p/'result/predictions.csv').open() as stream: rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),3)
            self.assertAlmostEqual(float(rows[1]['time_seconds_channel']),(33+1024)/32000)
            self.assertTrue((p/'result/CLASSIFICATION_COMPLETE').exists())
            with self.assertRaises(ValueError): run(channel,model,p/'result',interpreter_factory=lambda _:fake)

if __name__=='__main__': unittest.main()
