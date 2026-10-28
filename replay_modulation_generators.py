"""Exact train/validation generator replay; never trains or opens test data."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import numpy as np
import build_modulation_dataset as builder
import torchsig_probe as probe


def digest(iq):
    return hashlib.sha256(np.asarray(iq, dtype=np.complex64).tobytes()).hexdigest()


@contextmanager
def capture_params(module):
    """Observe actual returned draws without consuming additional random values."""
    original = module.fsk_random_params
    calls = []
    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(dict(h=float(result[0]), bt=None if result[1] is None else float(result[1]),
                          gaussian_span=None if result[2] is None else float(result[2])))
        return result
    module.fsk_random_params = wrapped
    try:
        yield calls
    finally:
        module.fsk_random_params = original


def regenerate(record, generate=probe.generator):
    label = record['label']
    rng = np.random.default_rng(np.random.SeedSequence(record['signal_seed']))
    noise_rng = np.random.default_rng(np.random.SeedSequence(record['noise_seed']))
    if label == 'noise':
        return (.5 * probe.normalize(noise_rng.normal(size=probe.SAMPLES) +
                1j * noise_rng.normal(size=probe.SAMPLES))).astype(np.complex64)
    params = record['parameters']
    clean = builder.controlled_fm(params['bandwidth_hz'], rng)[0] if label == 'fm' else generate(label, params, rng)
    return probe.add_channel(clean, params, noise_rng)[0]


def run(data, output):
    from torchsig.signals.builders import fsk
    data, output = Path(data), Path(output)
    if output.exists():
        raise ValueError('Ausgabe existiert; neuen --output verwenden')
    metadata = json.loads((data/'metadata.json').read_text())
    if not (data/'DATASET_COMPLETE').is_file() or metadata['classes'] != probe.CLASSES:
        raise ValueError('Datensatz unvollstaendig oder andere Klassen')
    versions = {n: importlib.metadata.version(n) for n in ('torchsig','torch','numpy','scipy')}
    if versions['torchsig'] != '2.2.0':
        raise ValueError('TorchSig 2.2.0 erforderlich')
    for name, value in versions.items():
        if metadata['versions'].get(name) != value:
            raise ValueError('Generatorumgebung abweichend: '+name)
    if metadata['versions']['builder_source_sha256'] != builder.sha(builder.__file__):
        raise ValueError('Builder-Code weicht von Datenerzeugung ab')
    output.mkdir(parents=True)
    report = dict(plan=json.loads(Path(__file__).with_name('GENERATOR_REPLAY_PLAN_V1.json').read_text()),
        versions=versions, metadata_sha256=builder.sha(data/'metadata.json'),
        sources={Path(p).name:builder.sha(p) for p in (__file__,builder.__file__,probe.__file__)},
        fsk_source_sha256=hashlib.sha256(inspect.getsource(fsk).encode()).hexdigest(),
        test_opened=False, training_performed=False, splits={}, complete=False)
    seen_ids, seen_hashes, conditions = set(), set(), set()
    try:
        for split in ('train','validation'):
            npz, params = data/(split+'.npz'), data/(split+'_parameters.jsonl')
            manifest = metadata['splits'][split]
            if builder.sha(npz) != manifest['sha256'] or builder.sha(params) != manifest['parameters_sha256']:
                raise ValueError('Dateihash abweichend: '+split)
            records = [json.loads(line) for line in params.read_text().splitlines()]
            rows = {r['source_id']:r for r in records}
            if len(rows) != len(records):
                raise ValueError('Doppelte Parameter-IDs')
            counts = dict(examples=0, exact=0, mismatches=0, fsk_instrumentation_exact=0,
                          mismatch_examples=[], fsk_parameters={})
            report['splits'][split] = counts
            split_conditions = set()
            with np.load(npz,allow_pickle=False) as saved, (output/(split+'_replay.jsonl')).open('x') as stream:
                if len(saved['iq']) != len(rows) or len(rows) != manifest['examples']:
                    raise ValueError('Beispielzahlen abweichend')
                used = set()
                for index, iq in enumerate(saved['iq']):
                    sid = str(saved['source_id'][index]); row = rows[sid]
                    if sid in seen_ids or sid in used:
                        raise ValueError('Quell-ID mehrfach vorhanden')
                    used.add(sid)
                    label = row['label']; condition = str(saved['condition_id'][index])
                    if label != probe.CLASSES[int(saved['y'][index])] or condition != row['condition_id']:
                        raise ValueError('Label-/Bedingungszuordnung abweichend')
                    if condition in conditions:
                        raise ValueError('Train/Validierungs-Bedingung ueberlappt')
                    split_conditions.add(condition)
                    expected_generator = 'controlled-fm-v1' if label=='fm' else ('gaussian-noise-v1' if label=='noise' else 'torchsig-2.2.0')
                    if row['generator'] != expected_generator:
                        raise ValueError('Unbekannter Generator')
                    snr = float(saved['snr_db'][index])
                    if not (np.isnan(snr) if label=='noise' else snr == row['parameters']['snr_db']):
                        raise ValueError('SNR-Zuordnung abweichend')
                    stored = digest(iq)
                    if stored != row['iq_sha256'] or stored in seen_hashes:
                        raise ValueError('Gespeicherter IQ-Hash abweichend oder doppelt')
                    seen_hashes.add(stored)
                    replayed = digest(regenerate(row)); captured = []
                    instrumented = None
                    if label in ('2fsk','2gfsk','2gmsk'):
                        with capture_params(fsk) as captured:
                            instrumented = digest(regenerate(row))
                        if len(captured) != 1:
                            raise ValueError('Erwartet genau einen FSK-Parameteraufruf')
                        if instrumented == replayed:
                            counts['fsk_instrumentation_exact'] += 1
                    exact = stored == replayed and (instrumented is None or instrumented == replayed)
                    counts['examples'] += 1
                    counts['exact' if exact else 'mismatches'] += 1
                    if not exact and len(counts['mismatch_examples']) < 10:
                        counts['mismatch_examples'].append(sid)
                    if captured:
                        counts['fsk_parameters'].setdefault(label, []).append(captured[0])
                    stream.write(json.dumps(dict(source_id=sid,label=label,stored_sha256=stored,
                        replay_sha256=replayed,instrumented_sha256=instrumented,exact=exact,
                        actual_fsk_parameters=captured))+'\n')
                if used != set(rows):
                    raise ValueError('Parameter-IDs nicht vollstaendig verwendet')
                seen_ids.update(used); conditions.update(split_conditions)
            print(split,counts['exact'],'/',counts['examples'],'bitidentisch',flush=True)
        report['complete'] = all(v['mismatches']==0 for v in report['splits'].values())
        report['parameter_interpretation_allowed'] = report['complete']
        if not report['complete']:
            raise ValueError('IQ-Rekonstruktion nicht exakt; Parameterauswertung gesperrt')
        for split, counts in report['splits'].items():
            counts['replay_jsonl_sha256'] = builder.sha(output/(split+'_replay.jsonl'))
    except Exception as error:
        report['complete'] = False
        report['parameter_interpretation_allowed'] = False
        report['error'] = str(error)
        raise
    finally:
        (output/'report.json').write_text(json.dumps(report,indent=2))
        print('Bericht:',output/'report.json',flush=True)
    (output/'REPLAY_COMPLETE').write_text('OK\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/modulation12-v1'))
    p.add_argument('--output',type=Path,default=Path('data')/('generator-replay-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%fZ')))
    args=p.parse_args()
    try:
        run(args.data,args.output)
    except (OSError,ValueError,KeyError,ImportError,RuntimeError) as error:
        p.exit(1,'Generator-Replay fehlgeschlagen: '+str(error)+'\n')


if __name__=='__main__':
    main()
