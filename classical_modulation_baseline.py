"""Fixed 35-feature, five-neighbor development control; CPU/NumPy only."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np

CLASSES=['am-dsb','fm','bpsk','qpsk','16qam','64qam','2fsk','2gfsk','2gmsk','ofdm-64','ook','noise']
Q=[.01,.1,.25,.5,.75,.9,.99]
LAGS=[1,2,4,8,16,32,64]
LEVELS=[-5.,0.,5.,10.,15.,20.,25.]
PLAN=Path(__file__).with_name('CLASSICAL_BASELINE_PLAN_V1.json')
FEATURE_NAMES=(['phase_step_std']+[f'phase_quantile_{q}' for q in Q]+
    [f'phase_autocorr_{lag}' for lag in LAGS]+['envelope_cv']+
    [f'envelope_quantile_{q}' for q in Q]+['envelope_fourth_moment','envelope_diff_rms']+
    [f'complex_coherence_{lag}' for lag in LAGS]+['spectral_width90','spectral_entropy','spectral_flatness'])


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)


def features(iq):
    z=np.asarray(iq,dtype=np.complex128)
    if z.ndim!=1 or len(z)<=max(LAGS)+1 or not np.isfinite(z).all():
        raise ValueError('Invalid IQ block')
    amplitude=np.abs(z);rms=max(float(np.sqrt(np.mean(amplitude**2))),1e-12)
    envelope=amplitude/rms
    step=np.angle(z[1:]*z[:-1].conj());step-=step.mean()
    scale=max(float(step.std()),1e-9);phase=step/scale
    psd=np.abs(np.fft.fftshift(np.fft.fft(z*np.hanning(len(z)))))**2
    if psd.sum()<=0:raise ValueError('Zero-power signal')
    probability=psd/psd.sum();cumulative=np.cumsum(probability)
    lo=min(len(z)-1,int(np.searchsorted(cumulative,.05)))
    hi=min(len(z)-1,int(np.searchsorted(cumulative,.95)))
    coherence=[]
    for lag in LAGS:
        left,right=z[lag:],z[:-lag]
        energy=max(float(np.sqrt(np.mean(abs(left)**2)*np.mean(abs(right)**2))),1e-12)
        coherence.append(float(abs(np.mean(left*right.conj()))/energy))
    result=np.r_[scale,np.quantile(phase,Q),[np.mean(phase[k:]*phase[:-k]) for k in LAGS],
        amplitude.std()/max(float(amplitude.mean()),1e-12),np.quantile(envelope,Q),
        np.mean(envelope**4),np.sqrt(np.mean(np.diff(envelope)**2)),coherence,
        (hi-lo)/len(z),-np.sum(probability*np.log(np.maximum(probability,1e-15)))/np.log(len(z)),
        np.exp(np.mean(np.log(np.maximum(probability,1e-15))))/probability.mean()]
    if result.shape!=(35,) or not np.isfinite(result).all():raise ValueError('Invalid feature vector')
    return result


def fit_scaler(x):
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Invalid training features')
    return x.mean(0),np.maximum(x.std(0),1e-6)


def neighbors(train,y,query,exclude_self=False):
    if len(train)<6 or train.ndim!=2 or query.ndim!=2 or train.shape[1]!=query.shape[1]:
        raise ValueError('Invalid neighbor input shapes')
    if not np.isfinite(train).all() or not np.isfinite(query).all():raise ValueError('Nonfinite neighbor input')
    if exclude_self and (query.shape!=train.shape or not np.array_equal(query,train)):
        raise ValueError('Self exclusion requires identical ordered training queries')
    predictions=[];votes=[]
    for start in range(0,len(query),32):
        distance=np.sum((query[start:start+32,None,:]-train[None,:,:])**2,axis=2)
        if not np.isfinite(distance).all():raise ValueError('Nonfinite distances')
        if exclude_self:
            distance[np.arange(len(distance)),np.arange(start,start+len(distance))]=np.inf
        indices=np.argsort(distance,axis=1,kind='stable')[:,:5]
        for row in indices:
            counts=np.bincount(y[row],minlength=12)
            predictions.append(int(counts.argmax()));votes.append(counts)
    return np.asarray(predictions),np.asarray(votes)


def metrics(y,pred):
    matrix=np.zeros((12,12),dtype=np.int64);np.add.at(matrix,(y,pred),1)
    count=matrix.sum(1);tp=matrix.diagonal();pred_count=matrix.sum(0)
    recall=np.divide(tp,count,out=np.zeros(12),where=count>0)
    f1=np.divide(2*tp,count+pred_count,out=np.zeros(12),where=count+pred_count>0)
    return dict(count=len(y),accuracy=float(tp.sum()/len(y)),macro_f1=float(f1.mean()),
        confusion_matrix=matrix.tolist(),recall={label:float(recall[i]) if count[i] else None for i,label in enumerate(CLASSES)})


def evaluation(y,pred,snr):
    result=metrics(y,pred)
    result['fsk_family_mean_recall']=float(np.mean([result['recall'][label] for label in ('2fsk','2gfsk','2gmsk')]))
    high=np.isfinite(snr)&(snr>=15)
    result['snr_ge15']=metrics(y[high],pred[high]);result['snr_ge15'].pop('macro_f1')
    result['snr_accuracy_signal_classes_only']={}
    for level in LEVELS:
        mask=snr==level
        result['snr_accuracy_signal_classes_only'][str(level)]=dict(count=int(mask.sum()),
            accuracy=float(np.mean(y[mask]==pred[mask])) if mask.any() else None)
    return result


def load_data(data,source,plan):
    spec=plan['data'];meta=json.loads((data/'metadata.json').read_text())
    if not (data/'DATASET_COMPLETE').is_file() or meta['classes']!=CLASSES:
        raise ValueError('Dataset incomplete/classes differ')
    for filename,key in [('metadata.json','metadata_sha256'),('train.npz','train_sha256'),('validation.npz','validation_sha256')]:
        if sha(data/filename)!=spec[key]:raise ValueError('Registered data hash differs: '+filename)
    selection=source/'training_selection.npz'
    if not (source/'DEVELOPMENT_COMPLETE').is_file() or sha(selection)!=spec['selection_sha256']:
        raise ValueError('Original selection incomplete/hash differs')
    arrays={};used_ids=set();used_conditions=set()
    for split in ('train','validation'):
        with np.load(data/(split+'.npz'),allow_pickle=False) as saved:
            a={name:saved[name] for name in ('iq','y','source_id','condition_id','snr_db')}
        n=len(a['y'])
        if a['iq'].shape!=(n,4096) or not np.iscomplexobj(a['iq']) or not np.isfinite(a['iq']).all():
            raise ValueError('Invalid IQ array')
        if not np.issubdtype(a['y'].dtype,np.integer) or set(a['y'].tolist())!=set(range(12)):
            raise ValueError('Invalid labels')
        if any(a[name].shape!=(n,) for name in ('source_id','condition_id','snr_db')):
            raise ValueError('Invalid metadata array shape')
        ids=set(a['source_id'].tolist());conditions=set(a['condition_id'].tolist())
        if len(ids)!=n or ids&used_ids or conditions&used_conditions:raise ValueError('Duplicate/overlapping IDs or conditions')
        used_ids.update(ids);used_conditions.update(conditions)
        if not np.isfinite(a['snr_db'][a['y']!=11]).all() or not np.isnan(a['snr_db'][a['y']==11]).all():
            raise ValueError('Invalid SNR/noise assignment')
        np.testing.assert_allclose(np.sqrt(np.mean(abs(a['iq'])**2,axis=1)),.5,atol=1e-5)
        arrays[split]=a
    with np.load(selection,allow_pickle=False) as selected:
        ids=selected['source_id'];lookup={sid:i for i,sid in enumerate(arrays['train']['source_id'])}
        if len(set(ids.tolist()))!=len(ids) or not set(ids.tolist())<=set(lookup):raise ValueError('Invalid selection IDs')
        indices=np.array([lookup[sid] for sid in ids]);a=arrays['train']
        if not np.array_equal(indices,np.sort(indices)):raise ValueError('Selection is not original dataset order')
        for name in ('source_id','y','condition_id','snr_db'):
            value=a[name][indices]
            equal=np.array_equal(value,selected[name],equal_nan=True) if value.dtype.kind=='f' else np.array_equal(value,selected[name])
            if not equal:raise ValueError('Selection row mismatch: '+name)
        arrays['train']={name:value[indices] for name,value in a.items()}
    for split,a in arrays.items():
        if len(a['y'])!=spec[split+'_examples']:raise ValueError('Registered count differs')
        if not np.all(np.bincount(a['y'],minlength=12)==len(a['y'])//12):raise ValueError('Class counts differ')
    groups={};a=arrays['train']
    for i,group in enumerate(a['condition_id']):groups.setdefault(str(group),[]).append(i)
    counts={level:0 for level in LEVELS}
    for indices in groups.values():
        if sorted(a['y'][indices].tolist())!=list(range(12)):raise ValueError('Incomplete condition group')
        signal=[i for i in indices if a['y'][i]!=11];values=a['snr_db'][signal]
        if not np.all(values==values[0]) or float(values[0]) not in counts:raise ValueError('Invalid group SNR')
        counts[float(values[0])]+=1
    if set(counts.values())!={20}:raise ValueError('Expected twenty groups per SNR')
    return arrays


def run(data,source,output):
    output.mkdir(parents=True,exist_ok=False)
    report=dict(complete=False,test_opened=False,plan=json.loads(PLAN.read_text()),
        plan_sha256=sha(PLAN),script_sha256=sha(__file__),feature_names=FEATURE_NAMES,numpy_version=np.__version__)
    try:
        arrays=load_data(data,source,report['plan'])
        report['inputs']=dict(metadata_sha256=sha(data/'metadata.json'),train_sha256=sha(data/'train.npz'),
            validation_sha256=sha(data/'validation.npz'),selection_sha256=sha(source/'training_selection.npz'))
        write(output/'protocol.json',dict(report,source=str(source.resolve())))
        start=time.perf_counter();features_by_split={}
        for split,a in arrays.items():
            features_by_split[split]=np.stack([features(z) for z in a['iq']])
            np.savez_compressed(output/(split+'_features.npz'),features=features_by_split[split],
                y=a['y'],source_id=a['source_id'],condition_id=a['condition_id'],snr_db=a['snr_db'])
            print(split,len(a['y']),'Merkmalsvektoren berechnet',flush=True)
        report['feature_seconds']=time.perf_counter()-start
        mean,std=fit_scaler(features_by_split['train'])
        train=(features_by_split['train']-mean)/std
        val=(features_by_split['validation']-mean)/std
        np.savez_compressed(output/'classifier.npz',mean=mean,std=std,train_scaled=train,
            train_labels=arrays['train']['y'],train_source_id=arrays['train']['source_id'],
            feature_names=np.array(FEATURE_NAMES),neighbors=5)
        for split,query in [('train',train),('validation',val)]:
            pred,votes=neighbors(train,arrays['train']['y'],query,exclude_self=split=='train')
            report[split]=evaluation(arrays[split]['y'],pred,arrays[split]['snr_db'])
            report[split]['evaluation_role']='training leave-one-out descriptive' if split=='train' else 'reused development validation'
            np.savez_compressed(output/(split+'_predictions.npz'),predicted=pred,neighbor_votes=votes,
                source_id=arrays[split]['source_id'])
        # Reload saved scaler/model and require identical validation predictions.
        with np.load(output/'classifier.npz',allow_pickle=False) as saved:
            restored,_=neighbors(saved['train_scaled'],saved['train_labels'],
                (features_by_split['validation']-saved['mean'])/saved['std'])
        if not np.array_equal(restored,pred):raise ValueError('Saved classifier roundtrip failed')
        report['classifier_roundtrip_exact']=True
        report['artifact_sha256']={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}
        report['complete']=True
        print('Validierung:',f'{report["validation"]["accuracy"]:.1%}',
            'Macro-F1:',f'{report["validation"]["macro_f1"]:.4f}',flush=True)
    except Exception as error:
        report['error']=str(error);raise
    finally:
        write(output/'report.json',report);print('Bericht:',output/'report.json',flush=True)
    (output/'CLASSICAL_BASELINE_COMPLETE').write_text('OK\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/modulation12-v1'))
    p.add_argument('--source',type=Path)
    p.add_argument('--output',type=Path,default=Path('models')/('classical-modulation-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%fZ')))
    args=p.parse_args()
    try:
        if args.source is None:
            sources=[p.parent for p in Path('models').glob('se-gain-development-*/DEVELOPMENT_COMPLETE')]
            if len(sources)!=1:raise ValueError('Expected one development source; specify --source')
            args.source=sources[0]
        run(args.data,args.source,args.output)
    except (OSError,ValueError,KeyError,AssertionError) as error:
        p.exit(1,'Klassische Baseline fehlgeschlagen: '+str(error)+'\n')
