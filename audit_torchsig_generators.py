"""TorchSig 2.2.0: saubere Bandbreiten und FM-Hub pruefen, ohne Training."""
import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from torchsig_probe import CLASSES,RATE,SAMPLES,generator,normalize,spectral_metrics

BANDWIDTHS=[2000.,4000.,8000.,12000.]


def metrics(iq):
    z=normalize(iq); phase=np.angle(z[1:]*z[:-1].conj())
    frequency=phase*RATE/(2*np.pi); centered=frequency-np.median(frequency)
    return dict(spectral_metrics(z),phase_frequency_median_hz=float(np.median(frequency)),
        phase_frequency_std_hz=float(np.std(centered)),
        phase_frequency_p95_abs_centered_hz=float(np.percentile(abs(centered),95)),
        near_nyquist_step_fraction=float(np.mean(abs(phase)>.9*np.pi)))


def audited_generator(label,plan,rng):
    if label!='fm': return generator(label,plan,rng),{}
    import torchsig.signals.builders.fm as fm
    details={}; original=fm.convolve
    def logged(message,taps,*args,**kwargs):
        filtered=original(message,taps,*args,**kwargs)
        details.update(message_rms_before_filter=float(np.sqrt(np.mean(abs(message)**2))),
            message_rms_after_filter=float(np.sqrt(np.mean(abs(filtered)**2))),
            message_peak_after_filter=float(np.max(abs(filtered))),message_filter_taps=len(taps))
        return filtered
    replay=np.random.default_rng(); replay.bit_generator.state=rng.bit_generator.state
    index=float(replay.uniform(1,10))
    details.update(generator_modulation_index=index,
        generator_fdev_parameter_hz=(plan['bandwidth_hz']/2)/(1+1/index),
        generator_message_cutoff_hz=plan['bandwidth_hz']/2/(index+1))
    with patch.object(fm,'convolve',logged): iq=generator(label,plan,rng)
    return iq,details


def audit(output,repeats=8,seed=2041,generate=audited_generator,versions=None):
    output=Path(output)
    if output.exists(): raise ValueError('Ausgabeordner existiert bereits')
    if repeats<2: raise ValueError('Mindestens zwei Wiederholungen erforderlich')
    rows=[]
    for cid,label in enumerate(CLASSES[:-1]):
        for bi,bandwidth in enumerate(BANDWIDTHS):
            for repeat in range(repeats):
                rng=np.random.default_rng(np.random.SeedSequence([seed,cid,bi,repeat]))
                iq,extra=generate(label,dict(bandwidth_hz=bandwidth,rolloff=.35),rng)
                result=metrics(iq)
                rows.append(dict(label=label,requested_bandwidth_hz=bandwidth,repeat=repeat,
                    seed_components=f'{seed},{cid},{bi},{repeat}',**result,
                    bandwidth_ratio=result['power90_width_hz']/bandwidth,**extra))
            print(label,int(bandwidth),'Hz fertig',flush=True)
    groups={}
    fields=['power90_width_hz','bandwidth_ratio','envelope_cv','phase_frequency_p95_abs_centered_hz','phase_frequency_std_hz']
    for label in CLASSES[:-1]:
        groups[label]={}
        for bandwidth in BANDWIDTHS:
            selected=[r for r in rows if r['label']==label and r['requested_bandwidth_hz']==bandwidth]
            groups[label][str(int(bandwidth))]={name:dict(
                p10=float(np.percentile([r[name] for r in selected],10)),
                median=float(np.median([r[name] for r in selected])),
                p90=float(np.percentile([r[name] for r in selected],90))) for name in fields}
    output.mkdir(parents=True)
    with (output/'cases.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader(); writer.writerows(rows)
    report=dict(seed=seed,versions=versions,sample_rate_hz=RATE,samples_per_case=SAMPLES,
        bandwidths_hz=BANDWIDTHS,repeats=repeats,cases=len(rows),groups=groups,
        fm_details=[r for r in rows if r['label']=='fm'],
        protocol='Original TorchSig modulators, clean baseband, no AWGN/shift, center crop of 16384 samples. No training.',
        limitations=['Requested 3dB and measured 90percent bandwidth are different definitions.',
            'Phase discriminator is interpretable as instantaneous frequency for FM; digital phase transitions complicate other classes.',
            'FM message RMS is measured over full filtered record, discriminator over center crop.',
            'FM internal parameters inferred from verified 2.2.0 source and first RNG draw.',
            'No speech, drone protocols or inference accuracy claim.'])
    (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    make_plot(output,groups)
    (output/'AUDIT_COMPLETE').write_text('OK\n')
    print('\nMedian 90%-Breite ohne Rauschen, Hz:')
    print('Klasse / angefordert',*[int(b) for b in BANDWIDTHS])
    for label,group in groups.items(): print(label,*[round(group[str(int(b))]['power90_width_hz']['median'],1) for b in BANDWIDTHS])
    print('\nFM: Nachrichten-RMS nach Filter und Phasendifferenz-Hub:')
    for bandwidth in BANDWIDTHS:
        selected=[r for r in rows if r['label']=='fm' and r['requested_bandwidth_hz']==bandwidth]
        rms=np.median([r.get('message_rms_after_filter',np.nan) for r in selected])
        deviation=np.median([r['phase_frequency_p95_abs_centered_hz'] for r in selected])
        print('{} Hz: Nachrichten-RMS {:.4f}; Hub-p95 {:.1f} Hz'.format(int(bandwidth),rms,deviation))
    print('Ergebnisse:',output.resolve()); return report


def make_plot(output,groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
    for label,group in groups.items():
        ax.plot(BANDWIDTHS,[group[str(int(b))]['power90_width_hz']['median'] for b in BANDWIDTHS],'o-',label=label)
    ax.set_xlabel('Requested generator bandwidth / Hz'); ax.set_ylabel('Measured 90% power bandwidth / Hz')
    ax.legend(ncol=3); ax.grid(alpha=.3); fig.savefig(output/'bandwidth_audit.png',dpi=140); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('data/torchsig-audit-v1'))
    p.add_argument('--repeats',type=int,default=8); p.add_argument('--seed',type=int,default=2041)
    args=p.parse_args()
    try:
        if importlib.metadata.version('torchsig')!='2.2.0': raise ValueError('TorchSig 2.2.0 erforderlich')
        import torchsig.signals.builders.fm as fm
        versions={name:importlib.metadata.version(name) for name in ('torchsig','numpy','scipy')}
        versions['fm_source_sha256']=hashlib.sha256(Path(inspect.getfile(fm)).read_bytes()).hexdigest()
        audit(args.output,args.repeats,args.seed,versions=versions)
    except (OSError,ValueError,ImportError,RuntimeError) as error: p.exit(1,'Generatoraudit fehlgeschlagen: '+str(error)+'\n')


if __name__=='__main__': main()
