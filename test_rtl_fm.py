import json
from pathlib import Path
import tempfile
import unittest
import wave
import numpy as np
from demodulate_rtl_fm import fm_discriminator,speech_filter,demodulate

class FMTests(unittest.TestCase):
    def test_discriminator_recovers_tone_and_offset(self):
        rate=32000; n=32000
        frequency=437.5+1200*np.sin(2*np.pi*1000*np.arange(n)/rate)
        iq=np.exp(2j*np.pi*np.cumsum(frequency)/rate)
        np.testing.assert_allclose(fm_discriminator(iq,rate),frequency[1:],atol=1e-7)
        with self.assertRaises(ValueError): fm_discriminator(np.array([1,np.nan]),rate)

    def test_audio_filter(self):
        h=speech_filter(32000); t=np.arange(len(h))
        gain=lambda f: abs(np.sum(h*np.exp(-2j*np.pi*f/32000*t)))
        self.assertAlmostEqual(gain(1000),1,places=6)
        self.assertLess(gain(0),.01); self.assertLess(gain(6000),.01)

    def test_wav_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); rate=32000; n=32000; t=np.arange(n)/rate
            iq=np.exp(2j*np.pi*np.cumsum(400+1500*np.sin(2*np.pi*1000*t))/rate)
            iq.astype('<c8').tofile(p/'channel.cf32')
            (p/'CHANNEL_COMPLETE').touch()
            (p/'channel_report.json').write_text(json.dumps(dict(output_rate_hz_nominal=rate,
                format='little-endian complex64 interleaved I,Q',samples=n,statistic_start_sample=33)))
            report=demodulate(p,p/'audio')
            with wave.open(str(p/'audio/fm_audio.wav'),'rb') as w:
                self.assertEqual(w.getframerate(),rate); self.assertEqual(w.getnchannels(),1)
                audio=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2')
            self.assertEqual(len(audio),report['audio_samples'])
            self.assertLessEqual(np.max(np.abs(audio)),29491)
            peak=np.argmax(np.abs(np.fft.rfft(audio)))
            self.assertAlmostEqual(peak*rate/len(audio),1000,delta=2)
            with self.assertRaises(ValueError): demodulate(p,p/'audio')

if __name__=='__main__': unittest.main()
