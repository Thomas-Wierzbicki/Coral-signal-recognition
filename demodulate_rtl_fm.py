"""Gefilterten 32-kS/s-I/Q-Kanal als FM-Hoerprobe in eine WAV-Datei umwandeln."""
import argparse
import json
from pathlib import Path
import wave
import numpy as np
from radioml_holdout import file_hash


def fm_discriminator(iq,rate):
    iq=np.asarray(iq,dtype=np.complex128)
    if iq.ndim!=1 or len(iq)<2 or not np.isfinite(iq).all():
        raise ValueError('Mindestens zwei endliche komplexe Samples erforderlich')
    return np.angle(iq[1:]*np.conj(iq[:-1]))*rate/(2*np.pi)


def speech_filter(rate,taps=257):
    t=np.arange(taps)-(taps-1)/2
    high=2*3000/rate*np.sinc(2*3000/rate*t)
    low=2*300/rate*np.sinc(2*300/rate*t)
    h=(high-low)*np.hamming(taps)
    gain=abs(np.sum(h*np.exp(-2j*np.pi*1000/rate*np.arange(taps))))
    return h/gain


def demodulate(folder,output):
    folder=Path(folder); output=Path(output)
    if output.exists(): raise ValueError('Ausgabeordner existiert bereits; neuen --output verwenden')
    meta=json.loads((folder/'channel_report.json').read_text(encoding='utf-8'))
    source=folder/'channel.cf32'
    rate=meta.get('output_rate_hz_nominal')
    if (not (folder/'CHANNEL_COMPLETE').is_file() or rate!=32000
            or meta.get('format')!='little-endian complex64 interleaved I,Q'
            or source.stat().st_size!=8*meta['samples']):
        raise ValueError('Vollstaendiger 32-kS/s-complex64-Kanal erforderlich')
    iq=np.fromfile(source,dtype='<c8')
    skip=meta.get('statistic_start_sample',33)
    if not isinstance(skip,int) or skip<0 or skip>=len(iq)-258:
        raise ValueError('Kanal zu kurz oder ungueltiger Startindex')
    frequency=fm_discriminator(iq[skip:],rate)
    offset=float(np.median(frequency))
    # Langsamen/statischen Diskriminatoranteil entfernen; kein RF-DC-Abzug.
    frequency=frequency-offset
    h=speech_filter(rate)
    filtered=np.convolve(frequency,h,mode='valid')
    peak=float(np.max(np.abs(filtered)))
    gain=.9/peak if peak>1e-12 else 0.
    pcm=np.rint(filtered*gain*32767).astype('<i2')
    output.mkdir(parents=True)
    wav=output/'fm_audio.wav'
    with wave.open(str(wav),'wb') as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(rate)
        stream.writeframes(pcm.tobytes())
    report=dict(source_channel=str(source.resolve()),source_channel_sha256=file_hash(source),
        audio_file=wav.name,sample_rate_hz_nominal=rate,audio_samples=len(pcm),
        duration_seconds_nominal=len(pcm)/rate,
        discriminator='angle(z[n] * conjugate(z[n-1])) * sample_rate / (2*pi)',
        removed_discriminator_median_hz=offset,filter_hz=[300,3000],filter_taps=len(h),
        deemphasis='none',squelch='none',normalization='global peak to 0.9',
        audio_gain_from_hz=gain,input_skip_samples=skip,
        audio_first_sample_time_seconds_nominal=(skip+.5+(len(h)-1)/2)/rate,
        limitation='Listening check only. No modulation classification, calibrated deviation, SNR or frequency-error estimate. Noise may also produce loud audio.')
    (output/'audio_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    print('WAV:',wav.resolve())
    print('Bitte pruefen: Vortrag verstaendlich? Keine Coral-Klassifikation erfolgt.')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path); p.add_argument('--output',type=Path)
    args=p.parse_args()
    try: demodulate(args.folder,args.output or args.folder/'fm-listening-v1')
    except (OSError,ValueError,KeyError,wave.Error) as error: p.exit(1,'FM-Hoerprobe fehlgeschlagen: '+str(error)+'\n')


if __name__=='__main__': main()
