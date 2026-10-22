"""Begrenzte RTL-SDR-I/Q-Aufnahme mit Rohdaten, Log und JSON; keine Klassifikation."""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from radioml_holdout import file_hash


def resolve_recorder(value):
    found=shutil.which(value)
    if found: return str(Path(found).resolve())
    path=Path(value)
    if path.is_file(): return str(path.resolve())
    raise ValueError('rtl_sdr.exe nicht gefunden. --rtl-sdr "C:\\Pfad\\rtl_sdr.exe" angeben.')


def capture(exe,output,frequency,rate,seconds,gain,ppm,device,label):
    if (not all(math.isfinite(v) for v in (frequency,rate,seconds,gain))
            or frequency<=0 or frequency>=2**32 or rate<225001 or rate>3200000
            or seconds<=0 or int(rate*seconds)<1024 or int(rate*seconds)>=2**31):
        raise ValueError('Ungueltige Frequenz/Rate/Dauer; mindestens 1024 Samples und <2^31 Samples')
    output=output.resolve()
    if output.exists(): raise ValueError('Ausgabeordner existiert bereits')
    output.mkdir(parents=True)
    raw=output/'capture.cu8'
    count=int(rate*seconds)
    command=[exe,'-d',str(device),'-f',str(frequency),'-s',str(rate),'-g',str(gain),
             '-p',str(ppm),'-n',str(count),str(raw)]
    metadata=dict(schema=1,source='RTL-SDR Blog V3 (user-selected)',
        created_utc=datetime.now(timezone.utc).isoformat(),format='cu8',
        channel_order=['I','Q'],sample_rate_hz_requested=rate,center_frequency_hz_requested=frequency,
        gain_db_requested=gain,gain_mode_requested='automatic' if gain==0 else 'manual',
        ppm_requested=ppm,device_requested=device,requested_samples=count,
        reference_label=label,label_status='user description, not verified ground truth',
        command=command,recorder_sha256=file_hash(exe),complete=False,
        note='Requested settings; consult recorder.log for actual rate/gain and errors. No resampling or normalization.')
    meta=output/'metadata.json'
    meta.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('Aufnahme:',output,flush=True)
    try:
        with (output/'recorder.log').open('wb') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            try:
                code=process.wait(timeout=seconds+30)
            except (subprocess.TimeoutExpired,KeyboardInterrupt):
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                raise
        metadata['recorder_exit_code']=code
        size=raw.stat().st_size if raw.exists() else 0
        metadata.update(actual_bytes=size,actual_samples=size//2)
        if code!=0 or size!=2*count:
            raise ValueError('Aufnahme unvollstaendig oder Recorderfehler; recorder.log pruefen')
        metadata['raw_sha256']=file_hash(raw)
        metadata['complete']=True
        (output/'CAPTURE_COMPLETE').write_text('OK\n')
    finally:
        meta.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('Aufnahme vollstaendig:',count,'I/Q-Samples')
    print('Naechster Schritt: python inspect_rtl_iq.py "'+str(output)+'"')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rtl-sdr',default='rtl_sdr.exe')
    parser.add_argument('--check',action='store_true',help='Nur Recorderpfad pruefen, keine Hardware oeffnen')
    parser.add_argument('--frequency',type=int,help='Mittenfrequenz in Hz')
    parser.add_argument('--sample-rate',type=int,default=1024000)
    parser.add_argument('--seconds',type=float,default=5)
    parser.add_argument('--gain',type=float,default=20.7,help='Angeforderte dB; 0 bedeutet Auto')
    parser.add_argument('--ppm',type=int,default=0)
    parser.add_argument('--device',default='0')
    parser.add_argument('--label',default='unknown')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    try:
        exe=resolve_recorder(args.rtl_sdr)
        if args.check:
            print('Recorder gefunden:',exe)
            print('Nur Dateipfad geprueft; DLLs, Treiber und Hardware noch nicht getestet.')
            return 0
        if args.frequency is None: parser.error('--frequency in Hz erforderlich')
        output=args.output or Path('data/recordings')/datetime.now(timezone.utc).strftime('rtl-%Y%m%d-%H%M%S-%f')
        capture(exe,output,args.frequency,args.sample_rate,args.seconds,args.gain,args.ppm,args.device,args.label)
        return 0
    except (OSError,ValueError,subprocess.TimeoutExpired) as error:
        print('RTL-Aufnahme fehlgeschlagen: '+str(error),file=sys.stderr); return 1
    except KeyboardInterrupt:
        print('Abgebrochen; Aufnahme nicht vollstaendig.',file=sys.stderr); return 130


if __name__=='__main__': sys.exit(main())
