"""Paired digital robustness grid; frozen classical classifier, NumPy CPU only."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import classical_modulation_baseline as baseline
import evaluate_classical_holdout as holdout

PLAN=Path(__file__).with_name('CLASSICAL_ROBUSTNESS_PLAN_V1.json')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def checked(folder,name,expected):
    path=folder/name
    if path.resolve().parent!=folder.resolve() or baseline.sha(path)!=expected:
        raise ValueError('Source artifact path/hash differs: '+name)
    return path


def load_source(folder,data,classical_source=None):
    report=read(folder/'report.json')
    if not (folder/'CLASSICAL_HOLDOUT_COMPLETE').is_file() or report['complete'] is not True or report['classifier_unchanged'] is not True:
        raise ValueError('Source holdout incomplete')
    if report['plan']!=read(holdout.PLAN) or report['plan_sha256']!=baseline.sha(holdout.PLAN):
        raise ValueError('Source holdout plan differs')
    protocol=read(checked(folder,'protocol.json',report['artifact_sha256']['protocol.json']))
    if protocol!=report['protocol']:raise ValueError('Source protocol/report differs')
    for name,digest in protocol['sources'].items():
        if baseline.sha(Path(__file__).with_name(name))!=digest:raise ValueError('Source generator code differs')
    source=classical_source or Path(protocol['source'])
    if baseline.sha(source/'report.json')!=protocol['source_report_sha256']:raise ValueError('Classical report differs')
    previous,meta,model,old_ids,old_hashes=holdout.freeze_source(data,source)
    if (baseline.sha(source/'classifier.npz')!=protocol['classifier_sha256']
            or holdout.array_digest(model)!=protocol['fitted_arrays_sha256']):raise ValueError('Frozen classifier differs')
    if np.__version__!=previous['numpy_version']:raise ValueError('Frozen NumPy version differs')
    with np.load(checked(folder,'holdout.npz',report['artifact_sha256']['holdout.npz']),allow_pickle=False) as saved:
        a={name:saved[name] for name in ('iq','y','source_id','condition_id','snr_db')}
    if (a['iq'].shape!=(1680,4096) or a['iq'].dtype!=np.complex64 or not np.isfinite(a['iq']).all()
            or any(a[key].shape!=(1680,) for key in ('y','source_id','condition_id','snr_db'))
            or not np.array_equal(np.bincount(a['y'],minlength=12),np.full(12,140))
            or len(set(a['source_id'].tolist()))!=1680):raise ValueError('Invalid holdout arrays/counts')
    parameter_file=checked(folder,'parameters.jsonl',report['artifact_sha256']['parameters.jsonl'])
    records=[json.loads(line) for line in parameter_file.read_text().splitlines()]
    if len(records)!=1680:raise ValueError('Record count differs')
    seen=set()
    for i,row in enumerate(records):
        label=baseline.CLASSES[int(a['y'][i])];digest=hashlib.sha256(a['iq'][i].tobytes()).hexdigest()
        if (row['source_id']!=a['source_id'][i] or row['condition_id']!=a['condition_id'][i]
                or row['label']!=label or row['iq_sha256']!=digest
                or row['source_id'] in old_ids or digest in old_hashes or digest in seen):
            raise ValueError('IQ/ID/label/condition join or independence differs')
        seen.add(digest)
        if label=='noise':
            if row['parameters'] is not None or not np.isnan(a['snr_db'][i]):raise ValueError('Noise SNR differs')
        elif float(a['snr_db'][i])!=row['parameters']['snr_db']:raise ValueError('Source SNR differs')
    with np.load(checked(folder,'features_predictions.npz',report['artifact_sha256']['features_predictions.npz']),allow_pickle=False) as saved:
        features=np.stack([baseline.features(z) for z in a['iq']])
        if not np.array_equal(features,saved['features']) or not np.array_equal(a['source_id'],saved['source_id']):
            raise ValueError('Original features/IDs do not reproduce')
        predictions,_=baseline.neighbors(model['train_scaled'],model['train_labels'],(features-model['mean'])/model['std'])
        if not np.array_equal(predictions,saved['predicted']):raise ValueError('Original predictions differ')
    if baseline.evaluation(a['y'],predictions,a['snr_db'])!=report['holdout']:raise ValueError('Original holdout matrix differs')
    for value in model.values():value.flags.writeable=False
    return a,model,predictions,source


def interpolation_kernel(ppm,crop=32,n=4096):
    indices=np.arange(crop,n-crop,dtype=np.float64);center=(n-1)/2
    positions=center+(indices-center)*(1+ppm*1e-6)
    base=np.floor(positions).astype(np.int64)
    taps=base[:,None]+np.arange(-16,17)[None,:]
    if taps.min()<0 or taps.max()>=n:raise ValueError('Resampling needs unavailable source margins')
    weights=np.sinc(positions[:,None]-taps)*np.hamming(33)[None,:]
    weights/=weights.sum(axis=1,keepdims=True)
    if not np.isfinite(weights).all():raise ValueError('Nonfinite interpolation kernel')
    return taps,weights


def normalize(z):
    rms=np.sqrt(np.mean(np.abs(z)**2,axis=1))
    if not np.isfinite(rms).all() or np.any(rms<=0):raise ValueError('Invalid transformed RMS')
    return (.5*z/rms[:,None]).astype(np.complex64)


def transform(iq,kind,value=0.,crop=32):
    iq=np.asarray(iq)
    if iq.ndim!=2 or iq.shape[1]!=4096 or not np.iscomplexobj(iq) or not np.isfinite(iq).all():
        raise ValueError('Invalid source IQ')
    if crop!=32:raise ValueError('Registered crop must be32')
    z=iq[:,crop:-crop].astype(np.complex128)
    if kind=='control':
        if value!=0:raise ValueError('Control must have zero impairment')
    elif kind=='cfo':
        n=np.arange(crop,4096-crop)
        z*=np.exp(2j*np.pi*value*n/32000)[None,:]
    elif kind=='sro':
        if value!=0:
            indices,weights=interpolation_kernel(value,crop)
            for start in range(0,len(iq),8):
                samples=iq[start:start+8].astype(np.complex128)[:,indices]
                z[start:start+8]=np.einsum('bnt,nt->bn',samples,weights,optimize=True)
    elif kind in ('gain','phase'):
        gain=value if kind=='gain' else 0.;theta=np.deg2rad(value if kind=='phase' else 0.)
        z=10**(gain/40)*z.real+1j*10**(-gain/40)*(np.cos(theta)*z.imag+np.sin(theta)*z.real)
    else:raise ValueError('Unknown stress factor')
    if not np.isfinite(z).all():raise ValueError('Nonfinite transformed IQ')
    return normalize(z)


def variants(plan):
    cases=[('control','control',0.)]
    for kind,key in [('cfo','frequency_offsets_hz'),('sro','sample_rate_ppm'),('gain','iq_gain_mismatch_db'),('phase','iq_phase_error_degrees')]:
        cases.extend((f'{kind}_{value:+g}',kind,float(value)) for value in plan[key])
    return cases


def paired(y,control,pred):
    if y.shape!=control.shape or y.shape!=pred.shape:raise ValueError('Unpaired prediction shapes')
    a=control==y;b=pred==y
    return dict(count=len(y),both_correct=int(np.sum(a&b)),control_only_correct=int(np.sum(a&~b)),
        variant_only_correct=int(np.sum(~a&b)),both_wrong=int(np.sum(~a&~b)),prediction_changes=int(np.sum(control!=pred)))


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(complete=False,fitting_performed=False,old_test_opened=False,plan=read(PLAN),cases={})
    try:
        a,model,original,source=load_source(args.holdout,args.data,args.classical_source)
        model_digest=holdout.array_digest(model);model_file_sha=baseline.sha(source/'classifier.npz')
        protocol=dict(plan=report['plan'],plan_sha256=baseline.sha(PLAN),numpy_version=np.__version__,
            holdout_report_sha256=baseline.sha(args.holdout/'report.json'),holdout_sha256=baseline.sha(args.holdout/'holdout.npz'),
            classifier_sha256=model_file_sha,fitted_arrays_sha256=model_digest,
            sources={Path(path).name:baseline.sha(path) for path in (__file__,baseline.__file__,holdout.__file__)})
        baseline.write(args.output/'protocol.json',protocol);report['protocol']=protocol
        report['full_length_reference']=baseline.evaluation(a['y'],original,a['snr_db'])
        np.savez_compressed(args.output/'source_rows.npz',source_id=a['source_id'],condition_id=a['condition_id'],
            y=a['y'],source_snr_db=a['snr_db'])
        control=None;control_metrics=None
        for name,kind,value in variants(report['plan']):
            start=time.perf_counter();z=transform(a['iq'],kind,value,report['plan']['crop_each_side'])
            if z.shape!=(len(a['y']),report['plan']['output_samples']):raise ValueError('Registered output length differs')
            iq_digest=hashlib.sha256(z.tobytes()).hexdigest()
            features=np.stack([baseline.features(row) for row in z])
            pred,votes=baseline.neighbors(model['train_scaled'],model['train_labels'],(features-model['mean'])/model['std'])
            metrics=baseline.evaluation(a['y'],pred,a['snr_db'])
            if kind=='control':
                control=pred.copy();control_metrics=metrics
                report['crop_effect_vs_full']=dict(accuracy=metrics['accuracy']-report['full_length_reference']['accuracy'],
                    macro_f1=metrics['macro_f1']-report['full_length_reference']['macro_f1'],
                    paired=paired(a['y'],original,pred))
            filename=name+'-predictions.npz'
            np.savez_compressed(args.output/filename,predicted=pred,neighbor_votes=votes,source_id=a['source_id'])
            report['cases'][name]=dict(kind=kind,value=value,metrics=metrics,
                delta_vs_cropped_control=dict(accuracy=metrics['accuracy']-control_metrics['accuracy'],
                    macro_f1=metrics['macro_f1']-control_metrics['macro_f1']),paired=paired(a['y'],control,pred),
                transformed_iq_sha256=iq_digest,prediction_file=filename,prediction_sha256=baseline.sha(args.output/filename),
                seconds=time.perf_counter()-start)
            baseline.write(args.output/'report.json',report)
            print(name,f'Zuordnung {metrics["accuracy"]:.1%}, Macro-F1 {metrics["macro_f1"]:.4f}',flush=True)
            del z,features
        if holdout.array_digest(model)!=model_digest or baseline.sha(source/'classifier.npz')!=model_file_sha:
            raise ValueError('Frozen classifier changed')
        if baseline.sha(args.holdout/'holdout.npz')!=protocol['holdout_sha256']:raise ValueError('Original holdout changed')
        report['classifier_unchanged']=True
        report['artifact_sha256']={p.name:baseline.sha(p) for p in sorted(args.output.iterdir()) if p.is_file() and p.name!='report.json'}
        report['complete']=True
    except Exception as error:
        report['error']=str(error);raise
    finally:
        baseline.write(args.output/'report.json',report);print('Bericht:',args.output/'report.json',flush=True)
    (args.output/'CLASSICAL_ROBUSTNESS_COMPLETE').write_text('OK\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/modulation12-v1'))
    p.add_argument('--holdout',type=Path);p.add_argument('--classical-source',type=Path)
    p.add_argument('--output',type=Path,default=Path('models')/('classical-robustness-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%fZ')))
    args=p.parse_args()
    try:
        if args.holdout is None:
            found=[p.parent for p in Path('models').glob('classical-holdout-*/CLASSICAL_HOLDOUT_COMPLETE')]
            if len(found)!=1:raise ValueError('Expected one source holdout; specify --holdout')
            args.holdout=found[0]
        run(args)
    except (OSError,ValueError,KeyError,RuntimeError,AssertionError) as error:
        p.exit(1,'Robustheitstest fehlgeschlagen: '+str(error)+'\n')
