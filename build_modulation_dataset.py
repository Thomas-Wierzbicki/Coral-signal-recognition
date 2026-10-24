"""Versionierter zwoelfklassiger I/Q-Datensatz mit kontrolliertem FM und getrennten Seeds."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from torchsig_probe import CLASSES,RATE,SAMPLES,generator,normalize,add_channel,spectral_metrics

SNR_LEVELS=[-5.,0.,5.,10.,15.,20.,25.]


def controlled_fm(bandwidth,rng):
    from scipy.signal import firwin,fftconvolve
    n=SAMPLES*4
    cutoff=float(rng.uniform(.08,.22)*bandwidth)
    lower=max(30.,.1*cutoff)
    taps=firwin(513,[lower,cutoff],pass_zero=False,fs=RATE)
    message=fftconvolve(rng.normal(size=n+len(taps)-1),taps,mode='valid')
    # Definierter Spitzenhub ueber den gesamten unabhaengigen Grundblock.
    message=message/np.max(np.abs(message))
    deviation=float((bandwidth/2-cutoff)*rng.uniform(.65,1.0))
    phase=2*np.pi*np.cumsum(deviation*message)/RATE
    iq=np.exp(1j*phase)
    start=(n-SAMPLES)//2; central=message[start:start+SAMPLES]
    return iq[start:start+SAMPLES].astype(np.complex64),dict(
        fm_generator='controlled-fm-v1',message_type='FIR bandlimited Gaussian noise, not speech',
        message_band_low_hz=lower,message_band_high_hz=cutoff,fir_taps=len(taps),
        peak_deviation_parameter_hz=deviation,full_message_peak=1.,
        crop_peak_deviation_hz=float(deviation*np.max(np.abs(central))),
        crop_rms_deviation_hz=float(deviation*np.sqrt(np.mean(central**2))))


def condition_plan(count,seed,split_id):
    rng=np.random.default_rng(np.random.SeedSequence([seed,split_id,0]))
    snrs=np.resize(np.array(SNR_LEVELS),count); rng.shuffle(snrs)
    return [dict(bandwidth_hz=float(rng.uniform(1500,12000)),
        offset_hz=float(rng.uniform(-4800,4800)),phase_rad=float(rng.uniform(-np.pi,np.pi)),
        snr_db=float(snrs[k]),rolloff=float(rng.uniform(.2,.5))) for k in range(count)]


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def build(output,per_class=600,seed=2050,generate=generator,versions=None):
    output=Path(output)
    if output.exists(): raise ValueError('Ausgabeordner existiert bereits')
    if per_class<20: raise ValueError('--per-class muss mindestens 20 sein')
    counts=[int(.7*per_class),int(.15*per_class)]
    counts.append(per_class-sum(counts))
    output.mkdir(parents=True)
    manifests={}; all_ids=set(); hashes=set(); width_groups={label:[] for label in CLASSES[:-1]}
    for split_id,(split,count) in enumerate(zip(('train','validation','test'),counts),1):
        plan=condition_plan(count,seed,split_id)
        samples=[]; targets=[]; ids=[]; param_ids=[]; snrs=[]
        with (output/(split+'_parameters.jsonl')).open('x',encoding='utf-8') as stream:
            for cid,label in enumerate(CLASSES):
                for k,item in enumerate(plan):
                    source_id=f'{seed}:{split_id}:{cid}:{k}'
                    if source_id in all_ids: raise ValueError('Quell-ID-Ueberschneidung')
                    all_ids.add(source_id)
                    signal_seed=[seed,split_id,cid,k,1]; noise_seed=[seed,split_id,cid,k,2]
                    rng=np.random.default_rng(np.random.SeedSequence(signal_seed))
                    noise_rng=np.random.default_rng(np.random.SeedSequence(noise_seed))
                    clean_metrics=None; extra={}
                    if label=='noise':
                        iq=(.5*normalize(noise_rng.normal(size=SAMPLES)+1j*noise_rng.normal(size=SAMPLES))).astype(np.complex64)
                    else:
                        if label=='fm': clean,extra=controlled_fm(item['bandwidth_hz'],rng)
                        else: clean=generate(label,item,rng)
                        if np.asarray(clean).shape!=(SAMPLES,): raise ValueError('Falsche Generatorform')
                        iq,shifted=add_channel(clean,item,noise_rng)
                        clean_metrics=spectral_metrics(shifted)
                        if split=='train': width_groups[label].append(clean_metrics['power90_width_hz'])
                    if iq.shape!=(SAMPLES,) or not np.isfinite(iq).all(): raise ValueError('Ungueltiges Beispiel')
                    digest=hashlib.sha256(iq.tobytes()).hexdigest()
                    if digest in hashes: raise ValueError('Bitidentische Beispiele entdeckt')
                    hashes.add(digest)
                    record=dict(source_id=source_id,condition_id=f'{seed}:{split_id}:{k}',label=label,
                        parameters=item if label!='noise' else None,signal_seed=signal_seed,noise_seed=noise_seed,
                        generator='controlled-fm-v1' if label=='fm' else ('gaussian-noise-v1' if label=='noise' else 'torchsig-2.2.0'),
                        clean_metrics=clean_metrics,received_metrics=spectral_metrics(iq),iq_sha256=digest,**extra)
                    stream.write(json.dumps(record)+'\n')
                    samples.append(iq); targets.append(cid); ids.append(source_id)
                    param_ids.append(f'{seed}:{split_id}:{k}'); snrs.append(item['snr_db'] if label!='noise' else np.nan)
                print(split,label,count,'Beispiele',flush=True)
        order=np.random.default_rng(np.random.SeedSequence([seed,split_id,999])).permutation(len(samples))
        path=output/(split+'.npz')
        np.savez_compressed(path,iq=np.stack(samples)[order],y=np.asarray(targets,dtype=np.int64)[order],
            source_id=np.asarray(ids)[order],condition_id=np.asarray(param_ids)[order],snr_db=np.asarray(snrs,dtype=np.float32)[order])
        with np.load(path,allow_pickle=False) as saved:
            if saved['iq'].shape!=(len(CLASSES)*count,SAMPLES): raise ValueError('Dateiform falsch')
            np.testing.assert_allclose(np.sqrt(np.mean(np.abs(saved['iq'])**2,axis=1)),.5,atol=1e-6)
            if not np.all(np.bincount(saved['y'],minlength=len(CLASSES))==count): raise ValueError('Klassenbelegung falsch')
        manifests[split]=dict(examples=len(samples),per_class=count,sha256=sha(path),parameters_sha256=sha(output/(split+'_parameters.jsonl')))
        print(split,'gespeichert und zurueckgelesen:',len(samples),flush=True)
    metadata=dict(schema=1,dataset_id='modulation12-controlled-fm-v1',classes=CLASSES,
        sample_rate_hz=RATE,samples_per_example=SAMPLES,iq_format='complex64',normalization='block-rms-0.5',
        seed=seed,versions=versions,splits=manifests,snr_levels_db=SNR_LEVELS,
        parameter_distribution=dict(requested_bandwidth_hz=[1500,12000],offset_hz=[-4800,4800],phase_rad=[-np.pi,np.pi],rolloff=[.2,.5]),
        train_clean_width90_hz={label:dict(p10=float(np.percentile(values,10)),median=float(np.median(values)),p90=float(np.percentile(values,90))) for label,values in width_groups.items()},
        limitations=['Synthetic first baseline, not independent real-world validation.',
            'Independent realizations across splits; same generators/parameter distributions.',
            'Condition IDs pair nuisance settings across classes; keep groups together if repartitioning.',
            'Requested and measured bandwidth differ across generators; no matching of measured distributions enforced.',
            'No fading, hardware imbalance, interfering transmitters or burst model yet.',
            'FM random bandlimited messages, no speech; FM does not separately label NFM/WFM.',
            'RMS normalization removes absolute received power; noise is not an unknown-signal class.',
            'Classes mix modulation orders and OFDM waveform family; not drone protocol labels.'])
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (output/'DATASET_COMPLETE').write_text('OK\n')
    print('Datensatz fertig:',output.resolve())
    print('Training/Validierung/Test:',*[manifests[name]['examples'] for name in ('train','validation','test')])
    print('Keine Erkennungsleistung nachgewiesen; bestehende Modelle/Daten bleiben erhalten.')
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('data/modulation12-v1'))
    p.add_argument('--per-class',type=int,default=600); p.add_argument('--seed',type=int,default=2050)
    args=p.parse_args()
    try:
        if importlib.metadata.version('torchsig')!='2.2.0': raise ValueError('TorchSig 2.2.0 erforderlich')
        versions={name:importlib.metadata.version(name) for name in ('torchsig','torch','numpy','scipy')}
        versions['builder_source_sha256']=sha(Path(__file__))
        build(args.output,args.per_class,args.seed,versions=versions)
    except (OSError,ValueError,ImportError,RuntimeError,AssertionError) as error:
        p.exit(1,'Datensatzerzeugung fehlgeschlagen: '+str(error)+'\n')


if __name__=='__main__': main()
