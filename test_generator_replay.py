import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import replay_modulation_generators as replay


class ReplayTests(unittest.TestCase):
    def test_observation_preserves_rng_and_restores_on_error(self):
        def draw(rng):
            return (rng.uniform(), .3, 3)
        module=types.SimpleNamespace(fsk_random_params=draw)
        reference=np.random.default_rng(9)
        expected=draw(reference)
        rng=np.random.default_rng(9)
        with self.assertRaises(RuntimeError):
            with replay.capture_params(module) as calls:
                self.assertEqual(module.fsk_random_params(rng),expected)
                self.assertEqual(rng.random(),reference.random())
                self.assertEqual(calls[0]['h'],expected[0])
                raise RuntimeError('stop')
        self.assertIs(module.fsk_random_params,draw)

    def fixture(self,root):
        fsk=types.ModuleType('torchsig.signals.builders.fsk')
        fsk.fsk_random_params=lambda rng:(float(rng.uniform(.1,.5)),.3,3)
        def generate(label,params,rng):
            if label in ('2fsk','2gfsk','2gmsk'):
                fsk.fsk_random_params(rng)
            return np.exp(1j*rng.normal(size=replay.probe.SAMPLES))
        modules={'torchsig':types.ModuleType('torchsig'),
                 'torchsig.signals':types.ModuleType('torchsig.signals'),
                 'torchsig.signals.builders':types.ModuleType('torchsig.signals.builders'),
                 'torchsig.signals.builders.fsk':fsk}
        modules['torchsig.signals.builders'].fsk=fsk
        data=root/'data';data.mkdir(); manifests={}
        for split_id,split in enumerate(('train','validation')):
            records=[];iq=[]
            for cid,label in enumerate(replay.probe.CLASSES):
                row=dict(source_id=f'{split}:{cid}',condition_id=split,label=label,
                    signal_seed=[3,split_id,cid],noise_seed=[4,split_id,cid],
                    parameters=dict(bandwidth_hz=4000.,offset_hz=300.,phase_rad=.2,snr_db=15.,rolloff=.3),
                    generator='controlled-fm-v1' if label=='fm' else ('gaussian-noise-v1' if label=='noise' else 'torchsig-2.2.0'))
                z=replay.regenerate(row,generate);row['iq_sha256']=replay.digest(z)
                records.append(row);iq.append(z)
            np.savez(data/(split+'.npz'),iq=np.array(iq),y=np.arange(12),
                source_id=[r['source_id'] for r in records],condition_id=[split]*12,
                snr_db=[15.]*11+[np.nan])
            (data/(split+'_parameters.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records))
            manifests[split]=dict(examples=12,sha256=replay.builder.sha(data/(split+'.npz')),
                parameters_sha256=replay.builder.sha(data/(split+'_parameters.jsonl')))
        metadata=dict(classes=replay.probe.CLASSES,splits=manifests,
            versions=dict(torchsig='2.2.0',torch='fixture',numpy='fixture',scipy='fixture',
                builder_source_sha256=replay.builder.sha(replay.builder.__file__)))
        (data/'metadata.json').write_text(json.dumps(metadata));(data/'DATASET_COMPLETE').touch()
        return data,modules,generate

    def exercise(self,tamper=False):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);data,modules,generate=self.fixture(root)
            if tamper:
                original=generate
                def generate(label,params,rng):
                    return original(label,params,rng)*1j
            # regenerate default is bound at definition; replace explicit call target for fixture.
            original_regenerate=replay.regenerate
            with patch.dict('sys.modules',modules), patch.object(replay.importlib.metadata,'version',
                    side_effect=lambda n:'2.2.0' if n=='torchsig' else 'fixture'), \
                 patch.object(replay.inspect,'getsource',return_value='fixture'), \
                 patch.object(replay,'regenerate',side_effect=lambda row:original_regenerate(row,generate)):
                if tamper:
                    with self.assertRaises(ValueError):
                        replay.run(data,root/'out')
                else:
                    report=replay.run(data,root/'out')
                    self.assertTrue(report['complete'])
                    self.assertEqual(report['splits']['train']['fsk_instrumentation_exact'],3)
            report=json.loads((root/'out/report.json').read_text())
            self.assertEqual(report['parameter_interpretation_allowed'],not tamper)
            self.assertEqual((root/'out/REPLAY_COMPLETE').exists(),not tamper)
            self.assertFalse((data/'test.npz').exists())

    def test_complete_replay_without_test_file(self):
        self.exercise()

    def test_mismatch_retains_failure_and_blocks_interpretation(self):
        self.exercise(True)


if __name__=='__main__':
    unittest.main()
