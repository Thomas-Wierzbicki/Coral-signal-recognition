import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from analyze_rtl_channel import lowpass,filtered_blocks,analyze
from radioml_holdout import file_hash

class ChannelTests(unittest.TestCase):
    def test_streaming_matches_full_convolution_and_mixing(self):
        rng=np.random.default_rng(4)
        raw=rng.integers(0,256,2*12000,dtype=np.uint8)
        from inspect_rtl_iq import decode_cu8
        h=lowpass(1024000)
        mixed=decode_cu8(raw)*np.exp(-2j*np.pi*20000/1024000*np.arange(12000))
        expected=np.convolve(mixed,h)[:12000:32]
        got=np.concatenate(list(filtered_blocks(raw,1024000,20000,32,h,block_size=1301)))
        np.testing.assert_allclose(got,expected,atol=1e-7)

    def test_filter_passband_and_alias_rejection(self):
        h=lowpass(1024000)
        t=np.arange(len(h))
        gain=lambda f: abs(np.sum(h*np.exp(-2j*np.pi*f/1024000*t)))
        self.assertGreater(gain(3000),.99)
        self.assertLess(gain(16000),.002)
        self.assertLess(gain(32000),.002)

    def test_complete_recording_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); n=102400
            tone=.2*np.exp(2j*np.pi*3000/1024000*np.arange(n))
            pairs=np.stack((tone.real,tone.imag),axis=-1)
            np.clip(np.rint(pairs*127.5+127.5),0,255).astype(np.uint8).tofile(folder/'capture.cu8')
            meta=dict(complete=True,format='cu8',channel_order=['I','Q'],actual_samples=n,
                      raw_sha256=file_hash(folder/'capture.cu8'),sample_rate_hz_requested=1024000,
                      center_frequency_hz_requested=438800000)
            (folder/'metadata.json').write_text(json.dumps(meta))
            (folder/'CAPTURE_COMPLETE').touch()
            result=analyze(folder,438800000,folder/'channel')
            self.assertEqual(result['samples'],3200)
            self.assertEqual(result['spectrum_peak_offset_hz'],3000)
            self.assertTrue((folder/'channel/CHANNEL_COMPLETE').exists())
            self.assertEqual((folder/'channel/channel.cf32').stat().st_size,3200*8)

if __name__=='__main__': unittest.main()
