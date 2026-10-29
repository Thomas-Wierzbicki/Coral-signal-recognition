"""Prospective synthetic realization check for the frozen classical classifier."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import numpy as np
import classical_modulation_baseline as baseline
import build_modulation_dataset as builder
import torchsig_probe as probe
from replay_modulation_generators import capture_params

PLAN=Path(__file__).with_name('CLASSICAL_HOLDOUT_PLAN_V1.json')
PROBE_SHA='544405d44981568f7b46055d9d11d4894cf5f40ac497bd2e5461834ea277a2e6'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def array_digest(arrays):
    h=hashlib.sha256()
    for key,value in sorted(arrays.items()):
        value=np.ascontiguousarray(value)
        h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(value.shape).encode());h.update(value.tobytes())
    return h.hexdigest()


def source_artifact(source,report,name):
    path=source/name
    if baseline.sha(path)!=report['artifact_sha256'][name]:raise ValueError('Source artifact hash differs: '+name)
    return path


def freeze_source(data,source):
    report=read(source/'report.json');meta=read(data/'metadata.json')
    if not (source/'CLASSICAL_BASELINE_COMPLETE').is_file() or not report['complete'] or not report['classifier_roundtrip_exact']:
        raise ValueError('Source baseline incomplete')
    if report['plan']!=read(baseline.PLAN) or report['plan_sha256']!=baseline.sha(baseline.PLAN):
        raise ValueError('Source baseline plan differs')
    for key in ('metadata_sha256','train_sha256','validation_sha256','selection_sha256'):
        if report['inputs'][key]!=report['plan']['data'][key]:raise ValueError('Source data differs from baseline plan')
    if report['script_sha256']!=baseline.sha(baseline.__file__) or report['feature_names']!=baseline.FEATURE_NAMES:
        raise ValueError('Frozen baseline feature implementation differs')
    if baseline.sha(data/'metadata.json')!=report['inputs']['metadata_sha256']:
        raise ValueError('Source metadata differs')
    if meta['classes']!=probe.CLASSES or meta['seed']==2100:raise ValueError('Source classes/seed incompatible')
    if meta['versions']['builder_source_sha256']!=baseline.sha(builder.__file__) or baseline.sha(probe.__file__)!=PROBE_SHA:
        raise ValueError('Original generator wrapper source differs')
    source_artifact(source,report,'protocol.json')
    with np.load(source_artifact(source,report,'classifier.npz'),allow_pickle=False) as saved:
        model={key:saved[key].copy() for key in saved.files}
    if (int(model['neighbors'])!=5 or model['feature_names'].tolist()!=baseline.FEATURE_NAMES
            or model['train_scaled'].shape!=(1680,35) or model['mean'].shape!=(35,) or model['std'].shape!=(35,)
            or not np.isfinite(model['train_scaled']).all() or not np.isfinite(model['mean']).all()
            or not np.isfinite(model['std']).all() or np.any(model['std']<1e-6)):
        raise ValueError('Invalid frozen classifier arrays')
    if (model['train_labels'].shape!=(1680,) or model['train_source_id'].shape!=(1680,)
            or len(set(model['train_source_id'].tolist()))!=1680
            or not np.issubdtype(model['train_labels'].dtype,np.integer)
            or not np.array_equal(np.bincount(model['train_labels'],minlength=12),np.full(12,140))):
        raise ValueError('Frozen training labels/IDs differ')
    with np.load(source_artifact(source,report,'train_features.npz'),allow_pickle=False) as train:
        mean,std=baseline.fit_scaler(train['features'])
        if not all((np.array_equal(mean,model['mean']),np.array_equal(std,model['std']),
            np.array_equal((train['features']-mean)/std,model['train_scaled']),
            np.array_equal(train['y'],model['train_labels']),np.array_equal(train['source_id'],model['train_source_id']))):
            raise ValueError('Saved classifier does not match original training features')
    with np.load(source_artifact(source,report,'validation_features.npz'),allow_pickle=False) as val:
        pred,_=baseline.neighbors(model['train_scaled'],model['train_labels'],(val['features']-model['mean'])/model['std'])
        old_metrics=baseline.metrics(val['y'],pred)
        with np.load(source_artifact(source,report,'validation_predictions.npz'),allow_pickle=False) as recorded:
            if not np.array_equal(pred,recorded['predicted']) or not np.array_equal(val['source_id'],recorded['source_id']):
                raise ValueError('Source validation prediction reproduction failed')
        if old_metrics['confusion_matrix']!=report['validation']['confusion_matrix']:
            raise ValueError('Source validation matrix reproduction failed')
    old_ids=set();old_hashes=set()
    for split in ('train','validation'):
        path=data/(split+'_parameters.jsonl')
        if baseline.sha(path)!=meta['splits'][split]['parameters_sha256']:raise ValueError('Original parameter hash differs')
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if row['source_id'] in old_ids:raise ValueError('Duplicate original source ID')
            old_ids.add(row['source_id']);old_hashes.add(row['iq_sha256'])
    return report,meta,model,old_ids,old_hashes


def generate_one(label,item,signal_seed,noise_seed):
    rng=np.random.default_rng(np.random.SeedSequence(signal_seed))
    noise_rng=np.random.default_rng(np.random.SeedSequence(noise_seed));extra={}
    if label=='noise':
        iq=(.5*probe.normalize(noise_rng.normal(size=probe.SAMPLES)+1j*noise_rng.normal(size=probe.SAMPLES))).astype(np.complex64)
    else:
        if label=='fm':clean,extra=builder.controlled_fm(item['bandwidth_hz'],rng)
        elif label in ('2fsk','2gfsk','2gmsk'):
            from torchsig.signals.builders import fsk
            with capture_params(fsk) as captured:clean=probe.generator(label,item,rng)
            if len(captured)!=1:raise ValueError('Unexpected FSK parameter call count')
            extra['actual_fsk_parameters']=captured[0]
        else:clean=probe.generator(label,item,rng)
        iq,shifted=probe.add_channel(clean,item,noise_rng)
        extra['clean_metrics']=probe.spectral_metrics(shifted)
    return iq,extra


def generate_set(output,plan,old_ids,old_hashes,generate=generate_one):
    conditions=builder.condition_plan(plan['examples_per_class'],plan['seed'],plan['split_id'])
    levels,counts=np.unique([p['snr_db'] for p in conditions],return_counts=True)
    if levels.tolist()!=plan['snr_levels_db'] or not np.all(counts==plan['conditions_per_snr']):
        raise ValueError('Registered SNR design differs')
    iq=[];labels=[];ids=[];condition_ids=[];snr=[];seen=set()
    with (output/'parameters.jsonl').open('x',encoding='utf-8') as stream:
        for cid,label in enumerate(probe.CLASSES):
            for k,item in enumerate(conditions):
                sid=f'{plan["seed"]}:{plan["split_id"]}:{cid}:{k}'
                signal_seed=[plan['seed'],plan['split_id'],cid,k,1];noise_seed=signal_seed[:-1]+[2]
                z,extra=generate(label,item,signal_seed,noise_seed)
                z=np.asarray(z)
                if z.dtype!=np.complex64 or z.shape!=(probe.SAMPLES,) or not np.isfinite(z).all():
                    raise ValueError('Invalid generated IQ')
                if not np.isclose(np.sqrt(np.mean(abs(z)**2)),.5,atol=1e-6):raise ValueError('Generated RMS differs')
                digest=hashlib.sha256(z.tobytes()).hexdigest()
                if sid in old_ids or digest in old_hashes or digest in seen:raise ValueError('Old/new IQ or source overlap')
                seen.add(digest);condition=f'{plan["seed"]}:{plan["split_id"]}:{k}'
                record=dict(source_id=sid,condition_id=condition,label=label,
                    parameters=item if label!='noise' else None,signal_seed=signal_seed,noise_seed=noise_seed,
                    iq_sha256=digest,**extra)
                stream.write(json.dumps(record,allow_nan=False)+'\n')
                iq.append(z);labels.append(cid);ids.append(sid);condition_ids.append(condition)
                snr.append(item['snr_db'] if label!='noise' else np.nan)
            print(label,len(conditions),'neue Beispiele',flush=True)
    if len(iq)!=plan['total_examples']:raise ValueError('Registered holdout size differs')
    np.savez_compressed(output/'holdout.npz',iq=np.stack(iq),y=np.array(labels),source_id=np.array(ids),
        condition_id=np.array(condition_ids),snr_db=np.array(snr))
    with np.load(output/'holdout.npz',allow_pickle=False) as saved:
        # Verify persisted IQ bytes against records, not only in-memory generation.
        records=[json.loads(line) for line in (output/'parameters.jsonl').read_text().splitlines()]
        for index,z in enumerate(saved['iq']):
            if hashlib.sha256(z.tobytes()).hexdigest()!=records[index]['iq_sha256']:
                raise ValueError('Saved holdout IQ roundtrip differs')
        return {key:saved[key] for key in saved.files}


def environment():
    from torchsig.signals.builders import fsk
    versions={name:importlib.metadata.version(name) for name in ('torchsig','torch','numpy','scipy')}
    if versions['torchsig']!='2.2.0':raise ValueError('TorchSig2.2.0 required')
    return versions,hashlib.sha256(inspect.getsource(fsk).encode()).hexdigest()


def run(data,source,output):
    output.mkdir(parents=True,exist_ok=False)
    report=dict(complete=False,fitting_performed=False,old_test_opened=False,plan=read(PLAN),plan_sha256=baseline.sha(PLAN))
    try:
        previous,meta,model,old_ids,old_hashes=freeze_source(data,source)
        versions,fsk_hash=environment()
        if versions!= {name:meta['versions'][name] for name in versions}:raise ValueError('Original generation environment differs')
        if versions['numpy']!=previous['numpy_version']:raise ValueError('Frozen classifier NumPy environment differs')
        before=array_digest(model);classifier_sha=baseline.sha(source/'classifier.npz')
        protocol=dict(plan=report['plan'],plan_sha256=report['plan_sha256'],source_report_sha256=baseline.sha(source/'report.json'),
            classifier_sha256=classifier_sha,fitted_arrays_sha256=before,versions=versions,fsk_source_sha256=fsk_hash,
            source=str(source.resolve()),metadata_sha256=baseline.sha(data/'metadata.json'),
            sources={Path(path).name:baseline.sha(path) for path in (__file__,baseline.__file__,builder.__file__,probe.__file__,Path(__file__).with_name('replay_modulation_generators.py'))})
        # Freeze recorded BEFORE new signal generation or scoring.
        baseline.write(output/'protocol.json',protocol);report['protocol']=protocol
        generated=generate_set(output,report['plan'],old_ids,old_hashes)
        x=np.stack([baseline.features(z) for z in generated['iq']])
        pred,votes=baseline.neighbors(model['train_scaled'],model['train_labels'],(x-model['mean'])/model['std'])
        report['holdout']=baseline.evaluation(generated['y'],pred,generated['snr_db'])
        old=previous['validation'];new=report['holdout']
        report['development_reference']=dict(accuracy=old['accuracy'],macro_f1=old['macro_f1'],fsk_family_mean_recall=old['fsk_family_mean_recall'])
        report['descriptive_delta']=dict(accuracy=new['accuracy']-old['accuracy'],macro_f1=new['macro_f1']-old['macro_f1'])
        np.savez_compressed(output/'features_predictions.npz',features=x,predicted=pred,neighbor_votes=votes,
            source_id=generated['source_id'])
        if array_digest(model)!=before or baseline.sha(source/'classifier.npz')!=classifier_sha:
            raise ValueError('Frozen classifier changed')
        report['classifier_unchanged']=True
        report['artifact_sha256']={p.name:baseline.sha(p) for p in sorted(output.iterdir()) if p.is_file()}
        report['complete']=True
        print('Neuer synthetischer Pruefsatz:',f'{new["accuracy"]:.1%}', 'Macro-F1:',f'{new["macro_f1"]:.4f}',flush=True)
    except Exception as error:
        report['error']=str(error);raise
    finally:
        baseline.write(output/'report.json',report);print('Bericht:',output/'report.json',flush=True)
    (output/'CLASSICAL_HOLDOUT_COMPLETE').write_text('OK\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/modulation12-v1'))
    p.add_argument('--source',type=Path)
    p.add_argument('--output',type=Path,default=Path('models')/('classical-holdout-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%fZ')))
    args=p.parse_args()
    try:
        if args.source is None:
            sources=[p.parent for p in Path('models').glob('classical-modulation-*/CLASSICAL_BASELINE_COMPLETE')]
            if len(sources)!=1:raise ValueError('Expected one frozen baseline; specify --source')
            args.source=sources[0]
        for done in Path('models').glob('classical-holdout-*/CLASSICAL_HOLDOUT_COMPLETE'):
            if read(done.parent/'report.json')['plan']['id']==read(PLAN)['id']:
                raise ValueError('This fixed holdout already completed; reuse its report, do not count reruns as new evidence')
        run(args.data,args.source,args.output)
    except (OSError,ValueError,KeyError,RuntimeError,ImportError,AssertionError) as error:
        p.exit(1,'Neuer Pruefsatz fehlgeschlagen: '+str(error)+'\n')
