"""Gesamte RTL-Aufnahme auf einen schmalen Kanal filtern und zeitlich auswerten."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from inspect_rtl_iq import decode_cu8
from radioml_holdout import file_hash


def lowpass(rate,cutoff=6250,taps=1025):
    t=np.arange(taps)-(taps-1)/2
    h=2*cutoff/rate*np.sinc(2*cutoff/rate*t)*np.hamming(taps)
    return h/h.sum()


def filtered_blocks(raw,rate,offset,decimation,h,block_size=65536):
    """Kausale FFT-Faltung mit Zustand; globale Dezimierungsphase bleibt erhalten."""
    history=np.zeros(len(h)-1,dtype=np.complex128)
    consumed=0
    for start in range(0,len(raw),2*block_size):
        iq=decode_cu8(raw[start:start+2*block_size])
        indices=np.arange(consumed,consumed+len(iq),dtype=np.float64)
        iq=iq*np.exp(-2j*np.pi*offset/rate*indices)
        combined=np.concatenate((history,iq))
        fft_size=1<<(len(combined)+len(h)-2).bit_length()
        convolution=np.fft.ifft(np.fft.fft(combined,fft_size)*np.fft.fft(h,fft_size))
        valid=convolution[len(h)-1:len(h)-1+len(iq)]
        yield valid[(-consumed)%decimation::decimation].astype(np.complex64)
        history=combined[-(len(h)-1):]
        consumed+=len(iq)


def analyze(folder,frequency,output):
    folder=Path(folder); output=Path(output)
    if output.exists(): raise ValueError('Ausgabeordner existiert bereits')
    meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    rawpath=folder/'capture.cu8'
    if (not (folder/'CAPTURE_COMPLETE').is_file() or not meta.get('complete')
            or meta.get('format')!='cu8' or meta.get('channel_order')!=['I','Q']
            or rawpath.stat().st_size!=2*meta['actual_samples']
            or file_hash(rawpath)!=meta['raw_sha256']):
        raise ValueError('Aufnahme/Hash ungueltig')
    rate=meta['sample_rate_hz_requested']
    if rate!=1024000:
        raise ValueError('Diese erste Kanalvariante erwartet nominell 1024000 Samples/s')
    offset=frequency-meta['center_frequency_hz_requested']
    if abs(offset)+10000>=rate/2: raise ValueError('Zielkanal ausserhalb der Aufnahme')
    raw=np.memmap(rawpath,dtype=np.uint8,mode='r')
    h=lowpass(rate); decimation=32; out_rate=rate//decimation
    output.mkdir(parents=True)
    destination=output/'channel.cf32'
    with destination.open('xb') as stream:
        for block in filtered_blocks(raw,rate,offset,decimation,h):
            block.astype('<c8').tofile(stream)
    channel=np.memmap(destination,dtype='<c8',mode='r')
    # Einschwingvorgang am Anfang nicht in der Statistik verwenden.
    skip=int(np.ceil(len(h)/decimation))
    frame=out_rate//50
    usable=(len(channel)-skip)//frame
    if usable<1: raise ValueError('Zu kurze Aufnahme fuer Kanalauswertung')
    frames=np.asarray(channel[skip:skip+usable*frame]).reshape(usable,frame)
    power=np.mean(np.abs(frames.astype(np.complex128))**2,axis=1)
    db=10*np.log10(np.maximum(power,1e-20))
    baseline=float(np.percentile(db,20)); threshold=baseline+6
    above=db>threshold
    with (output/'power_timeline.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream); writer.writerow(['time_seconds_nominal','power_db_relative','above_relative_threshold'])
        for k,value in enumerate(db): writer.writerow([(skip+k*frame)/out_rate,float(value),int(above[k])])
    fft_size=1024
    count=len(channel[skip:])//fft_size
    windows=np.asarray(channel[skip:skip+count*fft_size]).reshape(count,fft_size)
    window=np.hanning(fft_size)
    # Begrenzte Batches, alle vollstaendigen Fenster ueber die Aufnahme.
    psd=np.zeros(fft_size)
    for start in range(0,count,256):
        transforms=np.fft.fft(windows[start:start+256]*window,axis=1)/window.sum()
        psd+=np.sum(np.abs(transforms)**2,axis=0)
    psd=np.fft.fftshift(psd/max(count,1))
    freq=np.fft.fftshift(np.fft.fftfreq(fft_size,1/out_rate))
    np.savez_compressed(output/'channel_spectrum.npz',offset_hz=freq,power=psd)
    report=dict(source_directory=str(folder.resolve()),source_raw_sha256=meta['raw_sha256'],
        channel_frequency_hz_nominal=frequency,input_rate_hz_nominal=rate,
        output_rate_hz_nominal=out_rate,format='little-endian complex64 interleaved I,Q',
        samples=len(channel),duration_seconds_nominal=len(channel)/out_rate,
        filter=dict(type='Hamming-windowed sinc FIR',taps=len(h),cutoff_hz=6250,
                    group_delay_seconds=(len(h)-1)/2/rate,decimation=decimation),
        statistic_start_sample=skip,frame_seconds=.02,
        power_p10_db=float(np.percentile(db,10)),power_median_db=float(np.median(db)),
        power_p90_db=float(np.percentile(db,90)),power_max_db=float(db.max()),
        relative_baseline_db=baseline,relative_threshold_db=threshold,
        fraction_above_relative_threshold=float(np.mean(above)),
        spectrum_peak_offset_hz=float(freq[np.argmax(psd)]),
        center_dc_not_removed=True,normalization='none',
        limitation='Relative power heuristic is not voice/signal detection or SNR. Continuous transmission may have no quiet reference. No classification.')
    (output/'channel_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (output/'CHANNEL_COMPLETE').write_text('OK\n')
    print(json.dumps(report,indent=2))
    print('Kanal gespeichert:',destination.resolve())
    print('Keine Coral-Klassifikation. Pegelheuristik ist kein Sprach-/Sendesignalnachweis.')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('--frequency',type=int,required=True)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    try: analyze(args.folder,args.frequency,args.output or args.folder/'channel-analysis-v1')
    except (OSError,ValueError,KeyError) as error: p.exit(1,'Kanalanalyse fehlgeschlagen: '+str(error)+'\n')


if __name__=='__main__': main()
