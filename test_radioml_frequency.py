import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from check_radioml_frequency import group_stats,run,LABELS
from test_rtl_classify import FakeInterpreter

class RadioFrequencyTests(unittest.TestCase):
    def test_fm_rates_use_true_classes(self):
        y=np.tile(np.arange(5),2); pred=np.ones(10,dtype=int)
        result=group_stats(y,pred,np.tile([10.,20.],5),np.zeros(10))
        self.assertEqual(result['accuracy'],.2)
        self.assertEqual(result['confusion_matrix'][3][1],2)
        self.assertEqual(result['fm_prediction_fraction_by_true_class']['QPSK'],1)

    def test_training_only_full_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); data=p/'data'; data.mkdir(); root=p/'model'; root.mkdir()
            compiled=root/'compiled'; compiled.mkdir(); model=compiled/'model.tflite'; model.write_bytes(b'fake')
            manifest=dict(representation='iq-rms-v1',mode='rms-v1',labels=LABELS,
                          input_shape=[1,1024,2],channel_order=['I','Q'],target_rms=.5)
            (root/'preprocessing.json').write_text(json.dumps(manifest))
            meta=dict(manifest,source='RadioML2018.01A')
            (data/'metadata.json').write_text(json.dumps(meta)); (data/'IMPORT_COMPLETE').touch()
            z=.5*np.exp(2j*np.pi*.02*np.arange(1024))
            x=np.tile(np.stack((z.real,z.imag),axis=-1)[None,None],(5,1,1,1)).astype(np.float32)
            np.savez(data/'train.npz',x=x,y=np.arange(5),snr_db=np.full(5,10),source_index=np.arange(5))
            fake=FakeInterpreter(); result=run(data,model,p/'out',interpreter_factory=lambda _:fake)
            self.assertEqual(fake.calls,16)
            self.assertEqual(len(result['variants']),3)
            self.assertEqual(result['variants']['radioml_fm_offset']['fm_prediction_fraction_by_true_class']['BPSK'],1)
            self.assertEqual(result['paired']['centered']['changed'],0)
            self.assertTrue((p/'out/FREQUENCY_CHECK_COMPLETE').exists())
            with self.assertRaises(ValueError): run(data,model,p/'out',interpreter_factory=lambda _:fake)

if __name__=='__main__': unittest.main()
