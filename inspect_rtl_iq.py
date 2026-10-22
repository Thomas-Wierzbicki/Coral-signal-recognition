"""Gespeicherte cu8-I/Q-Aufnahme pruefen; keine Coral-Klassifikation."""
import argparse
import json
from pathlib import Path
import numpy as np
from radioml_holdout import file_hash


def decode_cu8(raw):
    values=np.asarray(raw)
    if values.dtype!=np.uint8 or values.size%2:
        raise ValueError('Gerade Anzahl uint8-Werte erforderlich')
    pairs=(values.reshape(-1,2).astype(np.float32)-127.5)/127.5
    return pairs[:,0]+1j*pairs[:,1]


def inspect(folder):
    folder=Path(folder)
    meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    raw=folder/'capture.cu8'
    if (not (folder/'CAPTURE_COMPLETE').is_file() or not meta.get('complete')
            or meta.get('format')!='cu8' or meta.get('channel_order')!=['I','Q']
            or raw.stat().st_size!=2*meta['actual_samples']
            or file_hash(raw)!=meta['raw_sha256']):
        raise ValueError('Aufnahme unvollstaendig, falsches Format oder SHA256 passt nicht')
    values=np.memmap(raw,dtype=np.uint8,mode='r')
    if len(values)<2048: raise ValueError('Zu kurze Aufnahme')
    total=0j; energy=0.; rails=0
    for start in range(0,len(values),2*262144):
        block=values[start:start+2*262144]
        iq=decode_cu8(block).astype(np.complex128)
        total+=iq.sum(); energy+=float(np.sum(np.abs(iq)**2))
        rails+=int(np.sum((block==0)|(block==255)))
    n=len(values)//2
    # Nur eine begrenzte PSD-Vorschau, nach dem ersten Viertelsekundenabschnitt.
    rate=meta['sample_rate_hz_requested']; fft_size=1024
    offset=min(int(.25*rate),max(0,n-fft_size))
    available=min(64,(n-offset)//fft_size)
    iq=decode_cu8(values[2*offset:2*(offset+available*fft_size)]).reshape(available,fft_size)
    window=np.hanning(fft_size)
    psd=np.mean(np.abs(np.fft.fftshift(np.fft.fft(iq*window,axis=1),axes=1)/window.sum())**2,axis=0)
    frequencies=np.fft.fftshift(np.fft.fftfreq(fft_size,1/rate))
    masked=psd.copy(); masked[fft_size//2-2:fft_size//2+3]=0
    peak=int(np.argmax(masked))
    result=dict(samples=n,duration_seconds_nominal=n/rate,
        rms=float(np.sqrt(energy/n)),dc_i=float((total/n).real),dc_q=float((total/n).imag),
        adc_rail_fraction=rails/len(values),preview_fft_size=fft_size,preview_windows=available,
        preview_start_sample=offset,preview_peak_offset_hz=float(frequencies[peak]),
        nominal_peak_frequency_hz=float(meta['center_frequency_hz_requested']+frequencies[peak]),
        note='Nominal frequencies; strongest FFT bin outside central five bins, not a detected channel or modulation.')
    (folder/'quality.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez_compressed(folder/'spectrum_preview.npz',offset_hz=frequencies,power=psd)
    print(json.dumps(result,indent=2))
    print('Keine Klassifikation. Kanalwahl, Filterung und Zeit-/Frequenzskalierung noch zu pruefen.')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('folder',type=Path); args=p.parse_args()
    try: inspect(args.folder)
    except (OSError,ValueError,KeyError) as error: p.exit(1,'I/Q-Pruefung fehlgeschlagen: '+str(error)+'\n')


if __name__=='__main__': main()
