import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import classify_classical_real_channel as real


def fixture_model():
    tone=(.5*np.exp(2j*np.pi*np.arange(4096)/32)).astype(np.complex64)
    feature=real.baseline.features(tone)
    return dict(train_scaled=np.tile(feature,(6,1)),train_labels=np.full(6,1,dtype=int),
        mean=np.zeros(35),std=np.ones(35)),tone


class RealChannelTests(unittest.TestCase):
    def test_invalid_blocks_explicit_and_votes_not_percentages(self):
        model,tone=fixture_model();bad=tone.copy();bad[0]=np.nan+0j
        iq=np.r_[tone,np.zeros(4096,np.complex64),bad,np.ones(17,np.complex64)]
        stream=io.StringIO();report=real.classify_blocks(model,iq,0,'fm',stream)
        self.assertEqual(report['blocks'],3);self.assertEqual(report['valid_blocks'],1)
        self.assertEqual(report['invalid_reasons'],dict(zero_power=1,nonfinite=1))
        self.assertEqual(report['reference_match_fraction_valid_blocks'],1.)
        self.assertEqual(report['tail_samples_omitted'],17)
        rows=list(csv.DictReader(io.StringIO(stream.getvalue())))
        self.assertEqual(rows[0]['prediction'],'fm');self.assertEqual(rows[0]['top_neighbor_votes'],'5')
        self.assertEqual(rows[1]['prediction'],'');self.assertEqual(rows[2]['prediction'],'')
        self.assertEqual(float(rows[1]['start_seconds_nominal']),.128)

    def test_low_amplitude_same_normalization_and_unlabeled_reference(self):
        model,tone=fixture_model();stream=io.StringIO()
        report=real.classify_blocks(model,np.r_[tone,tone*1e-5],0,'unlabeled',stream)
        self.assertEqual(report['predictions']['fm'],2)
        self.assertIsNone(report['reference_match_fraction_valid_blocks'])

    def create_channel(self,root):
        folder=root/'channel';folder.mkdir();_,tone=fixture_model()
        iq=np.r_[np.zeros(33,np.complex64),tone,tone,np.ones(15,np.complex64)]
        iq.astype('<c8').tofile(folder/'channel.cf32')
        meta=dict(output_rate_hz_nominal=32000,format='little-endian complex64 interleaved I,Q',
            samples=len(iq),statistic_start_sample=33,channel_frequency_hz_nominal=438800000)
        (folder/'channel_report.json').write_text(json.dumps(meta));(folder/'CHANNEL_COMPLETE').touch()
        return folder

    def test_metadata_rate_and_size_rejection(self):
        with tempfile.TemporaryDirectory() as name:
            folder=self.create_channel(Path(name));meta,iq,start=real.read_channel(folder)
            self.assertEqual(start,33)
            meta['output_rate_hz_nominal']=48000;(folder/'channel_report.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError):real.read_channel(folder)

    def test_complete_real_file_readback_and_frozen_model(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);channel=self.create_channel(root);source=root/'source';source.mkdir()
            (source/'classifier.npz').write_bytes(b'frozen model');(source/'report.json').write_text('{}')
            model,_=fixture_model();before=real.frozen.array_digest(model)
            args=SimpleNamespace(output=root/'out',source=source,data=root,channel=channel,expected='fm')
            with patch.object(real.frozen,'freeze_source',return_value=(dict(numpy_version=np.__version__),{},model,set(),set())):
                report=real.run(args)
            self.assertTrue(report['complete']);self.assertTrue(report['classifier_and_channel_unchanged'])
            self.assertEqual(report['real_channel']['valid_blocks'],2)
            self.assertEqual(real.frozen.array_digest(model),before)
            self.assertEqual(report['predictions_sha256'],real.baseline.sha(args.output/'predictions.csv'))
            self.assertTrue((args.output/'CLASSICAL_REAL_CHANNEL_COMPLETE').exists())

    def test_failure_report_without_success_marker(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);args=SimpleNamespace(output=root/'out',source=root,data=root,channel=root,expected='fm')
            with patch.object(real.frozen,'freeze_source',side_effect=ValueError('model hash differs')):
                with self.assertRaises(ValueError):real.run(args)
            self.assertFalse(json.loads((args.output/'report.json').read_text())['complete'])
            self.assertFalse((args.output/'CLASSICAL_REAL_CHANNEL_COMPLETE').exists())

    def test_locator_known_relative_recording_and_missing_files(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);folder=root/'data/recordings'/real.RECORDING/'channel-analysis-v1';folder.mkdir(parents=True)
            with patch.object(real.Path,'cwd',return_value=root),patch.object(real.Path,'home',return_value=root/'fixturehome'):
                with self.assertRaisesRegex(ValueError,'nicht gefunden'):real.locate_channel()
                for filename in real.FILES:(folder/filename).touch()
                self.assertEqual(real.locate_channel(),folder.resolve())


if __name__=='__main__':unittest.main()
