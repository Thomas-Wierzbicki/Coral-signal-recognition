import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from sweep_rtl_frequency import phase_offset,shift_block,transitions,run,TARGET
from test_rtl_classify import FakeInterpreter
from classify_rtl_channel import LABELS

class FrequencyTests(unittest.TestCase):
    def test_shift_preserves_envelope_and_modulation(self):
        t=np.arange(1024)
        z=((1+.1*np.sin(t/20))*np.exp(1j*(2*np.pi*.013*t+.4*np.sin(t/31)))).astype(np.complex64)
        offset=phase_offset(z)
        for target in (0,TARGET):
            shifted=shift_block(z,offset,target)
            np.testing.assert_allclose(abs(shifted),abs(z),atol=2e-7)
            self.assertAlmostEqual(phase_offset(shifted),target,places=7)
            a=np.angle(z[1:]*z[:-1].conj())
            b=np.angle(shifted[1:]*shifted[:-1].conj())
            residual=np.angle(np.exp(1j*(b-a-2*np.pi*(target-offset))))
            np.testing.assert_allclose(residual,0,atol=3e-7)
        with self.assertRaises(ValueError): phase_offset(np.zeros(1024,dtype=complex))

    def test_transitions(self):
        result=transitions(['FM','QPSK','FM'],['FM','FM','QPSK'])
        self.assertEqual(result['changed'],2)
        self.assertEqual(result['matrix'][1][3],1)

    def test_full_three_variant_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); folder=p/'channel'; folder.mkdir(); modeldir=p/'model'; modeldir.mkdir()
            compiled=modeldir/'compiled'; compiled.mkdir(); model=compiled/'model.tflite'; model.write_bytes(b'fake')
            (modeldir/'preprocessing.json').write_text(json.dumps(dict(representation='iq-rms-v1',mode='rms-v1',
                target_rms=.5,input_shape=[1,1024,2],channel_order=['I','Q'],labels=LABELS)))
            n=33+2*1024+5
            np.exp(2j*np.pi*.013*np.arange(n)).astype('<c8').tofile(folder/'channel.cf32')
            (folder/'CHANNEL_COMPLETE').touch()
            (folder/'channel_report.json').write_text(json.dumps(dict(output_rate_hz_nominal=32000,
                format='little-endian complex64 interleaved I,Q',samples=n,statistic_start_sample=33)))
            fake=FakeInterpreter()
            result=run(folder,model,p/'out',interpreter_factory=lambda _:fake)
            self.assertEqual(fake.calls,9)
            self.assertEqual(len(result['variants']),3)
            self.assertEqual(result['variants']['original']['blocks'],2)
            self.assertEqual(result['transitions']['centered']['changed'],0)
            self.assertTrue((p/'out/FREQUENCY_TEST_COMPLETE').exists())
            with self.assertRaises(ValueError): run(folder,model,p/'out',interpreter_factory=lambda _:fake)

if __name__=='__main__': unittest.main()
