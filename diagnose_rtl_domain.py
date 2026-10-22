"""Deskriptiver I/Q-Vergleich: reale Aufnahme gegen RadioML-Training, ohne Modellanpassung."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from radioml_holdout import file_hash
from classify_rtl_channel import prepare_block,LABELS

FEATURES=['envelope_cv','phase_offset_cycles_per_sample','phase_spread_rad',
          'phase_step_change_rms_rad','lag1_phase_coherence','dc_power_fraction',
          'spectral_width90_cycles_per_sample']


def features(iq):
    normalized,_=prepare_block(iq)
    z=normalized[0,:,0].astype(np.float64)+1j*normalized[0,:,1].astype(np.float64)
    envelope=np.abs(z)
    phase=np.angle(z[1:]*np.conj(z[:-1]))
    vector=np.mean(np.exp(1j*phase)); offset=float(np.angle(vector))
    residual=np.angle(np.exp(1j*(phase-offset)))
    changes=np.angle(np.exp(1j*np.diff(phase)))
    # Nur fuer die Spektrumskennzahl zentrieren; keine Aenderung am Modellinput.
    centered=z*np.exp(-1j*offset*np.arange(len(z)))
    power=np.abs(np.fft.fftshift(np.fft.fft(centered*np.hanning(len(z)))))**2
    cumulative=np.cumsum(power)/power.sum()
    lower=np.searchsorted(cumulative,.05); upper=np.searchsorted(cumulative,.95)
    return dict(envelope_cv=float(envelope.std()/envelope.mean()),
        phase_offset_cycles_per_sample=offset/(2*np.pi),phase_spread_rad=float(np.std(residual)),
        phase_step_change_rms_rad=float(np.sqrt(np.mean(changes**2))),
        lag1_phase_coherence=float(abs(vector)),
        dc_power_fraction=float(abs(z.mean())**2/np.mean(abs(z)**2)),
        spectral_width90_cycles_per_sample=float((upper-lower)/len(z)))


def summary(rows):
    return {name:dict(p10=float(np.percentile([r[name] for r in rows],10)),
                      median=float(np.median([r[name] for r in rows])),
                      p90=float(np.percentile([r[name] for r in rows],90))) for name in FEATURES}


def diagnose(folder,reference,output):
    folder=Path(folder); reference=Path(reference); output=Path(output)
    if output.exists(): raise ValueError('Ausgabeordner existiert bereits')
    channel_meta=json.loads((folder/'channel_report.json').read_text(encoding='utf-8'))
    source=folder/'channel.cf32'
    if (not (folder/'CHANNEL_COMPLETE').is_file() or channel_meta.get('output_rate_hz_nominal')!=32000
            or channel_meta.get('format')!='little-endian complex64 interleaved I,Q'
            or source.stat().st_size!=8*channel_meta['samples']):
        raise ValueError('Vollstaendiger 32-kS/s-Kanal erforderlich')
    meta=json.loads((reference/'metadata.json').read_text(encoding='utf-8'))
    if (not (reference/'IMPORT_COMPLETE').is_file() or meta.get('source')!='RadioML2018.01A'
            or meta.get('representation')!='iq-rms-v1' or meta.get('labels')!=LABELS
            or meta.get('input_shape')!=[1,1024,2] or meta.get('target_rms')!=.5):
        raise ValueError('RadioML-I/Q-Referenz passt nicht')
    # Ausschliesslich Trainings-NPZ, keine neuen Testdaten zur Diagnose verwenden.
    with np.load(reference/'train.npz',allow_pickle=False) as data:
        x,y,snr,ids=(data[k] for k in ('x','y','snr_db','source_index'))
    if (x.shape!=(len(y),1,1024,2) or len(y)==0 or snr.shape!=y.shape or ids.shape!=y.shape
            or not np.issubdtype(y.dtype,np.integer) or set(y.tolist())!=set(range(5))
            or not np.isfinite(x).all() or not np.isfinite(snr).all()):
        raise ValueError('Ungueltige Trainingsarrays')
    rows=[]; groups={}
    for i in range(len(y)):
        label=LABELS[int(y[i])]
        f=features(x[i,0,:,0]+1j*x[i,0,:,1])
        row=dict(source='radioml_train',reference_label=label,prediction='',
                 source_index=int(ids[i]),time_seconds_channel='',snr_db=float(snr[i]),**f)
        rows.append(row); groups.setdefault('radioml_train/'+label,[]).append(row)
        if snr[i]>=10: groups.setdefault('radioml_train_snr_ge10/'+label,[]).append(row)
    iq=np.memmap(source,dtype='<c8',mode='r'); start=channel_meta.get('statistic_start_sample',33)
    if not isinstance(start,int) or start<0: raise ValueError('Ungueltiger Startindex')
    count=(len(iq)-start)//1024
    if count<1: raise ValueError('Realer Kanal zu kurz')
    source_hash=file_hash(source)
    prediction_folder=folder/'coral-real-v1'; predictions=None
    if (prediction_folder/'CLASSIFICATION_COMPLETE').is_file():
        report=json.loads((prediction_folder/'report.json').read_text(encoding='utf-8'))
        if (report.get('source_channel_sha256')!=source_hash or report.get('blocks')!=count
                or report.get('input_start_sample')!=start):
            raise ValueError('Vorhersagebericht passt nicht zum Kanal')
        with (prediction_folder/'predictions.csv').open(newline='',encoding='utf-8') as stream:
            predictions=list(csv.DictReader(stream))
        if len(predictions)!=count: raise ValueError('Vorhersagezahl passt nicht')
    for k in range(count):
        position=start+k*1024; prediction=''
        if predictions is not None:
            r=predictions[k]; prediction=r['prediction']
            if (int(r['block'])!=k or prediction not in LABELS
                    or abs(float(r['time_seconds_channel'])-position/32000)>1e-9):
                raise ValueError('Vorhersagezeit/Reihenfolge passt nicht')
        row=dict(source='real',reference_label='FM (recording listening reference)',prediction=prediction,
                 source_index=k,time_seconds_channel=position/32000,snr_db='',**features(iq[position:position+1024]))
        rows.append(row); groups.setdefault('real/all',[]).append(row)
        if prediction: groups.setdefault('real/predicted_'+prediction,[]).append(row)
    summaries={name:dict(count=len(group),features=summary(group)) for name,group in groups.items()}
    output.mkdir(parents=True)
    with (output/'features.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result=dict(source_channel_sha256=source_hash,training_npz_sha256=file_hash(reference/'train.npz'),
        channel_metadata=channel_meta,training_metadata=meta,groups=summaries,
        limitations='Descriptive distributions only, not causal attribution or classifier. RadioML physical sample rate unknown. No resampling/training/model modification. Offset may include noise and modulation; not calibrated frequency error.')
    (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    names=['real/all','radioml_train_snr_ge10/FM','radioml_train_snr_ge10/QPSK']
    for name in names:
        if name not in summaries: continue
        print('\n'+name+' (n='+str(summaries[name]['count'])+')')
        for feature,values in summaries[name]['features'].items():
            print('{}: Median {:.6f}; p10 {:.6f}; p90 {:.6f}'.format(feature,values['median'],values['p10'],values['p90']))
    print('\nErgebnisse:',output.resolve())
    print('Deskriptive Diagnose; keine isolierte Ursache und keine neue Genauigkeitsmessung.')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('folder',type=Path)
    p.add_argument('--reference',type=Path,default=Path('data/radioml2018-iq-v1'))
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    try: diagnose(args.folder,args.reference,args.output or args.folder/'domain-diagnosis-v1')
    except (OSError,ValueError,KeyError) as error: p.exit(1,'I/Q-Diagnose fehlgeschlagen: '+str(error)+'\n')


if __name__=='__main__': main()
