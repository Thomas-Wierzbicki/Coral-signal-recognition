"""Import externally recorded rtl_sdr cu8 bytes without changing the original."""
import argparse
from datetime import datetime,timezone
import hashlib
import math
from pathlib import Path
from classical_modulation_baseline import sha,write


def import_capture(source,output,frequency,rate=1024000,seconds=60.,gain=20.7):
    source,output=Path(source),Path(output)
    if not source.is_file():raise ValueError('Rohdatei fehlt: '+str(source))
    if not all(math.isfinite(v) for v in (frequency,rate,seconds,gain)) or frequency<=0 or rate<225001 or rate>3200000 or seconds<=0:
        raise ValueError('Ungueltige Aufnahmeparameter')
    samples=int(rate*seconds)
    if samples<1024 or source.stat().st_size!=2*samples:
        raise ValueError('Dateilaenge passt nicht zu angegebener Rate/Dauer; keine stille Teilaufnahme importieren')
    output.mkdir(parents=True,exist_ok=False)
    metadata=dict(schema=1,source='externally recorded RTL-SDR cu8 file, user-supplied settings',
        import_created_utc=datetime.now(timezone.utc).isoformat(),source_path=str(source.resolve()),
        format='cu8',channel_order=['I','Q'],sample_rate_hz_requested=int(rate),
        center_frequency_hz_requested=int(frequency),gain_db_requested=float(gain),
        gain_mode_requested='automatic' if gain==0 else 'manual',ppm_requested=0,
        requested_samples=samples,actual_samples=samples,actual_bytes=2*samples,
        reference_label='unknown',label_status='no listening/ground-truth label established',
        settings_status='user supplied command settings, actual tuner frequency/rate/gain and dropouts not independently verified; no recorder log available',
        complete=False)
    write(output/'metadata.json',metadata)
    try:
        before=source.stat();digest=hashlib.sha256()
        with source.open('rb') as original,(output/'capture.cu8').open('xb') as destination:
            for block in iter(lambda:original.read(1024*1024),b''):
                destination.write(block);digest.update(block)
        copied=digest.hexdigest();after=source.stat()
        if (before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns
                or sha(source)!=copied or sha(output/'capture.cu8')!=copied):
            raise ValueError('Original/Kopie veraendert oder SHA256 abweichend')
        metadata.update(raw_sha256=copied,source_raw_sha256=copied,complete=True)
        write(output/'metadata.json',metadata)
        (output/'CAPTURE_COMPLETE').write_text('External file import verified; device settings not independently verified.\n')
    except Exception as error:
        metadata['error']=str(error);metadata['complete']=False
        write(output/'metadata.json',metadata);raise
    print('Importiert:',output.resolve(),flush=True)
    print('Original und Kopie SHA256-geprueft:',copied,flush=True)
    return metadata


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('--frequency',type=int,required=True)
    p.add_argument('--sample-rate',type=int,default=1024000)
    p.add_argument('--seconds',type=float,default=60.);p.add_argument('--gain',type=float,default=20.7)
    p.add_argument('--output',type=Path,default=Path('data/recordings')/('rtl-import-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%fZ')))
    p.add_argument('--inspect',action='store_true',help='Nach Import vorhandene IQ-Qualitaetspruefung ausfuehren')
    args=p.parse_args()
    try:
        import_capture(args.source,args.output,args.frequency,args.sample_rate,args.seconds,args.gain)
        if args.inspect:
            from inspect_rtl_iq import inspect
            inspect(args.output)
    except (OSError,ValueError,KeyError,RuntimeError) as error:
        p.exit(1,'RTL-Import fehlgeschlagen: '+str(error)+'\n')
