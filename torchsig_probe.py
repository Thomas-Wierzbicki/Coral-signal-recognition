"""Kleiner TorchSig-2.2.0-Generatorprueflauf; keine Trainings-/Testgenauigkeit."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import numpy as np

CLASSES=['am-dsb','fm','bpsk','qpsk','16qam','64qam','2fsk','2gfsk','2gmsk','ofdm-64','ook','noise']
RATE=32000; SAMPLES=4096


def parameter_plan(count,seed):
    rng=np.random.default_rng(seed)
    return [dict(example=k,bandwidth_hz=float(rng.uniform(.06,.22)*RATE),
                 offset_hz=float(rng.uniform(-.12,.12)*RATE),
                 phase_rad=float(rng.uniform(-np.pi,np.pi)),snr_db=float([0,10,20][k%3]),
                 rolloff=float(rng.uniform(.2,.5))) for k in range(count)]


def normalize(iq):
    iq=np.asarray(iq,dtype=np.complex128)
    if iq.ndim!=1 or not np.isfinite(iq).all() or not len(iq): raise ValueError('Ungueltige I/Q-Daten')
    rms=float(np.sqrt(np.mean(np.abs(iq)**2)))
    if rms<=0: raise ValueError('Leerer Signalinhalt')
    return iq/rms


def add_channel(clean,plan,rng):
    z=normalize(clean)*np.exp(1j*(2*np.pi*plan['offset_hz']/RATE*np.arange(len(clean))+plan['phase_rad']))
    noise=(rng.normal(size=len(z))+1j*rng.normal(size=len(z)))/np.sqrt(2)
    noisy=z+noise*10**(-plan['snr_db']/20)
    return (.5*normalize(noisy)).astype(np.complex64),z


def spectral_metrics(iq):
    iq=np.asarray(iq); psd=np.abs(np.fft.fftshift(np.fft.fft(iq*np.hanning(len(iq)))))**2
    cumulative=np.cumsum(psd)/psd.sum(); f=np.fft.fftshift(np.fft.fftfreq(len(iq),1/RATE))
    lo=int(np.searchsorted(cumulative,.05)); hi=min(len(iq)-1,int(np.searchsorted(cumulative,.95)))
    return dict(peak_hz=float(f[np.argmax(psd)]),power90_low_hz=float(f[lo]),
                power90_high_hz=float(f[hi]),power90_width_hz=float(f[hi]-f[lo]),
                envelope_cv=float(np.std(np.abs(iq))/np.mean(np.abs(iq))))


def generator(label,plan,rng):
    from torchsig.signals.builders.am import am_modulator
    from torchsig.signals.builders.fm import fm_modulator
    from torchsig.signals.builders.constellation import constellation_modulator
    from torchsig.signals.builders.fsk import fsk_modulator
    from torchsig.signals.builders.ofdm import ofdm_modulator
    # Laengeren unabhaengigen Block erzeugen, nur mittleren Abschnitt nutzen.
    n=SAMPLES*4; bw=plan['bandwidth_hz']
    if label=='am-dsb': z=am_modulator('dsb',bw,RATE,n,rng)
    elif label=='fm': z=fm_modulator(bw,RATE,n,rng)
    elif label in ('bpsk','qpsk','16qam','64qam','ook'):
        z=constellation_modulator(label,'srrc',bw,RATE,n,alpha_rolloff=plan['rolloff'],rng=rng)
    elif label in ('2fsk','2gfsk','2gmsk'):
        z=fsk_modulator(2,label[1:],bw,RATE,n,rng)
    elif label=='ofdm-64': z=ofdm_modulator(64,bw,RATE,n,rng,cyclic_prefix_len=16)
    else: raise ValueError('Unbekannte Generator-Klasse')
    if np.asarray(z).shape!=(n,): raise ValueError('Generator lieferte falsche Laenge: '+label)
    start=(n-SAMPLES)//2
    return np.asarray(z[start:start+SAMPLES])


def create_probe(output,count=12,seed=2040,generate=generator,versions=None):
    output=Path(output)
    if output.exists(): raise ValueError('Ausgabeordner existiert bereits; neuen --output verwenden')
    if count<3: raise ValueError('Mindestens drei Beispiele je Klasse erforderlich')
    plan=parameter_plan(count,seed); signals=[]; labels=[]; details=[]
    output.mkdir(parents=True)
    for cid,label in enumerate(CLASSES):
        metrics=[]
        for k,item in enumerate(plan):
            rng=np.random.default_rng(np.random.SeedSequence([seed,cid,k,1]))
            noise_rng=np.random.default_rng(np.random.SeedSequence([seed,cid,k,2]))
            if label=='noise':
                iq=(.5*normalize(noise_rng.normal(size=SAMPLES)+1j*noise_rng.normal(size=SAMPLES))).astype(np.complex64)
                clean_stats=None
            else:
                clean=generate(label,item,rng)
                if np.asarray(clean).shape!=(SAMPLES,): raise ValueError('Falsche Beispielgroesse')
                iq,shifted=add_channel(clean,item,noise_rng); clean_stats=spectral_metrics(shifted)
            metric=spectral_metrics(iq)
            row=dict(label=label,**item,clean_metrics=clean_stats,received_metrics=metric,
                     generator_seed_components=[seed,cid,k,1],noise_seed_components=[seed,cid,k,2])
            if label=='noise':
                row.update(snr_db=None,bandwidth_hz=None,offset_hz=None,phase_rad=None,rolloff=None)
            details.append(row); signals.append(iq); labels.append(cid); metrics.append(metric)
        print('{}: {} Beispiele; Median belegte 90%-Breite mit Rauschen {:.1f} Hz'.format(
            label,count,np.median([m['power90_width_hz'] for m in metrics])),flush=True)
    iq=np.stack(signals); y=np.asarray(labels,dtype=np.int64)
    np.savez_compressed(output/'probe.npz',iq=iq,y=y,example_id=np.tile(np.arange(count),len(CLASSES)))
    metadata=dict(schema=1,purpose='generator inspection only',classes=CLASSES,sample_rate_hz=RATE,
        samples_per_example=SAMPLES,examples_per_class=count,seed=seed,versions=versions,
        common_parameter_plan=plan,examples=details,
        normalization='complex block RMS 0.5 after AWGN; raw complex64 retained',
        limitations=['Requested bandwidth is not guaranteed measured occupied bandwidth.',
            'Noise class has no signal/SNR/bandwidth/offset.',
            'AM/FM random-message generator, not real speech.',
            'No fading, IQ imbalance, bursts or device impairments in this initial probe.',
            'No train/validation/test split; not a performance benchmark.',
            'OFDM/FSK classes are not drone protocol labels.'])
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    plot_previews(output,iq,count)
    (output/'PROBE_COMPLETE').write_text('OK\n')
    print('Prueflauf fertig:',output.resolve(),iq.shape)
    print('Noch kein Training, keine Drohnen-/Modulationserkennungsleistung nachgewiesen.')
    return metadata


def plot_previews(output,iq,count):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for cid,label in enumerate(CLASSES):
        # k=2 entspricht im gemeinsamen Plan 20 dB, sofern vorhanden.
        z=iq[cid*count+2]; t=np.arange(len(z))/RATE
        fig,axes=plt.subplots(3,1,figsize=(9,7),layout='constrained')
        axes[0].plot(t[:512]*1000,z.real[:512],label='I'); axes[0].plot(t[:512]*1000,z.imag[:512],label='Q',alpha=.6)
        axes[0].legend(); axes[0].set_xlabel('Time / ms'); axes[0].set_title(label+' (noise: no SNR; others: requested 20 dB)')
        axes[1].plot(t*1000,np.abs(z)); axes[1].set_xlabel('Time / ms'); axes[1].set_ylabel('Envelope')
        power=np.abs(np.fft.fftshift(np.fft.fft(z*np.hanning(len(z)))))**2
        f=np.fft.fftshift(np.fft.fftfreq(len(z),1/RATE))
        axes[2].plot(f/1000,10*np.log10(np.maximum(power/power.max(),1e-12)))
        axes[2].set_xlabel('Baseband frequency / kHz'); axes[2].set_ylabel('Relative power / dB')
        fig.savefig(output/(label+'.png'),dpi=120); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('data/torchsig-probe-v1'))
    p.add_argument('--per-class',type=int,default=12); p.add_argument('--seed',type=int,default=2040)
    args=p.parse_args()
    try:
        version=importlib.metadata.version('torchsig')
        if version!='2.2.0': raise ValueError('Dieser Prueflauf erwartet TorchSig 2.2.0')
        import torch
        versions={name:importlib.metadata.version(name) for name in ('torchsig','torch','numpy','scipy')}
        versions.update(cuda_available=torch.cuda.is_available(),torch_cuda=torch.version.cuda)
        if torch.cuda.is_available(): versions['gpu']=torch.cuda.get_device_name(0)
        create_probe(args.output,args.per_class,args.seed,versions=versions)
    except (OSError,ValueError,ImportError,RuntimeError) as error: p.exit(1,'TorchSig-Prueflauf fehlgeschlagen: '+str(error)+'\n')


if __name__=='__main__': main()
