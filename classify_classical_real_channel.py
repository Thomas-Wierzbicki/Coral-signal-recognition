"""Practical transfer check of frozen classical model on an existing real IQ channel."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import numpy as np
import classical_modulation_baseline as baseline
import evaluate_classical_holdout as frozen

PLAN=Path(__file__).with_name('CLASSICAL_REAL_CHANNEL_PLAN_V1.json')
RECORDING='rtl-20261022-171509-339614'
FILES=('channel.cf32','channel_report.json','CHANNEL_COMPLETE')


def locate_channel(explicit=None):
    if explicit is not None:return Path(explicit)
    suffix=Path('data/recordings')/RECORDING/'channel-analysis-v1'
    candidates=[Path.cwd()/suffix,
        Path.home()/'source/repos/Coral-signal-recognition'/suffix,
        Path('/mnt/c/Users')/Path.home().name/'source/repos/Coral-signal-recognition'/suffix]
    found={p.resolve() for p in candidates if all((p/f).is_file() for f in FILES)}
    if len(found)==1:return found.pop()
    if len(found)>1:raise ValueError('Mehrere Aufnahmen gefunden; --channel PFAD angeben')
    raise ValueError('FM-Aufnahme hier nicht gefunden. Vom Aufnahme-Rechner die Dateien '+
        ', '.join(FILES)+' aus '+str(suffix)+' in denselben relativen Ordner unter ~/rf-learning kopieren; dann erneut starten. Alternativ --channel PFAD.')


def read_channel(folder):
    meta=json.loads((folder/'channel_report.json').read_text(encoding='utf-8'))
    path=folder/'channel.cf32'
    if (not (folder/'CHANNEL_COMPLETE').is_file() or meta.get('output_rate_hz_nominal')!=32000
            or meta.get('format')!='little-endian complex64 interleaved I,Q'
            or not isinstance(meta.get('samples'),int) or meta['samples']<4096
            or path.stat().st_size!=8*meta['samples']):
        raise ValueError('Vollstaendiger 32-kS/s-complex64-Kanal erforderlich')
    start=meta.get('statistic_start_sample',33)
    if not isinstance(start,int) or start<0 or start+4096>meta['samples']:
        raise ValueError('Ungueltiger Einschwing-Startindex oder Aufnahme zu kurz')
    return meta,np.memmap(path,dtype='<c8',mode='r'),start


def classify_blocks(model,iq,start,expected,output_stream):
    count=(len(iq)-start)//4096
    if count<1:raise ValueError('Mindestens4096 verwendbare Samples erforderlich')
    writer=csv.writer(output_stream)
    writer.writerow(['block','start_seconds_nominal','input_rms','status','prediction','top_neighbor_votes']+
        ['votes_'+label for label in baseline.CLASSES])
    counts={label:0 for label in baseline.CLASSES};invalid={};valid=0
    for first in range(0,count,32):
        rows=[];features=[];valid_positions=[]
        for k in range(first,min(first+32,count)):
            offset=start+k*4096;z=np.asarray(iq[offset:offset+4096],dtype=np.complex128)
            if not np.isfinite(z).all():
                rows.append([k,offset/32000,'','invalid_nonfinite','',0]+[0]*12)
                invalid['nonfinite']=invalid.get('nonfinite',0)+1;continue
            rms=float(np.sqrt(np.mean(abs(z)**2)))
            if rms<=0:
                rows.append([k,offset/32000,rms,'invalid_zero_power','',0]+[0]*12)
                invalid['zero_power']=invalid.get('zero_power',0)+1;continue
            normalized=(.5*z/rms).astype(np.complex64)
            features.append(baseline.features(normalized));valid_positions.append(len(rows))
            rows.append([k,offset/32000,rms,'unvalidated_known_class_output','',0]+[0]*12)
        if features:
            x=np.stack(features)
            pred,votes=baseline.neighbors(model['train_scaled'],model['train_labels'],(x-model['mean'])/model['std'])
            for pos,cid,vote in zip(valid_positions,pred,votes):
                label=baseline.CLASSES[int(cid)];rows[pos][4:]=[label,int(vote.max())]+vote.tolist()
                counts[label]+=1;valid+=1
        writer.writerows(rows)
        print(min(first+32,count),'/',count,'reale Bloecke',flush=True)
    matches=counts[expected] if expected in counts else None
    return dict(blocks=count,valid_blocks=valid,invalid_blocks=sum(invalid.values()),invalid_reasons=invalid,
        predictions=counts,expected_reference_label=expected,
        reference_match_count=matches,
        reference_match_fraction_valid_blocks=matches/valid if matches is not None and valid else None,
        reference_status='recording-level user listening reference, not per-block truth; match fraction is not accuracy',
        block_samples=4096,block_seconds_nominal=.128,start_sample=start,
        tail_samples_omitted=len(iq)-start-count*4096,
        output_status='experimental forced known-class output; no validated unknown detection',
        neighbor_vote_interpretation='integer0..5, not calibrated probability')


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(complete=False,fitting_performed=False,plan=json.loads(PLAN.read_text()),
        plan_sha256=baseline.sha(PLAN))
    try:
        source_report,meta,model,_,_=frozen.freeze_source(args.data,args.source)
        if source_report['numpy_version']!=np.__version__:raise ValueError('NumPy-Version weicht vom eingefrorenen Klassifikator ab')
        channel_meta,iq,start=read_channel(args.channel)
        model_digest=frozen.array_digest(model);classifier_sha=baseline.sha(args.source/'classifier.npz')
        input_sha=baseline.sha(args.channel/'channel.cf32');metadata_sha=baseline.sha(args.channel/'channel_report.json')
        for array in model.values():array.flags.writeable=False
        protocol=dict(plan=report['plan'],plan_sha256=report['plan_sha256'],numpy_version=np.__version__,
            classifier_sha256=classifier_sha,classifier_report_sha256=baseline.sha(args.source/'report.json'),
            fitted_arrays_sha256=model_digest,channel_sha256=input_sha,channel_metadata_sha256=metadata_sha,
            channel_directory=str(args.channel.resolve()),channel_metadata=channel_meta,
            preprocessing='original32kS/s4096-sample blocks, transient skipped, RMS0.5; no DC/offset/resampling correction',
            sources={Path(p).name:baseline.sha(p) for p in (__file__,baseline.__file__,frozen.__file__)})
        baseline.write(args.output/'protocol.json',protocol);report['protocol']=protocol
        before=time.perf_counter()
        with (args.output/'predictions.csv').open('x',newline='',encoding='utf-8') as stream:
            report['real_channel']=classify_blocks(model,iq,start,args.expected,stream)
        if report['real_channel']['valid_blocks']==0:raise ValueError('Keine verwendbaren Signalbloecke; Aufnahme pruefen')
        report['processing_seconds']=time.perf_counter()-before
        if (frozen.array_digest(model)!=model_digest or baseline.sha(args.source/'classifier.npz')!=classifier_sha
                or baseline.sha(args.channel/'channel.cf32')!=input_sha
                or baseline.sha(args.channel/'channel_report.json')!=metadata_sha):
            raise ValueError('Klassifikator oder Quellaufnahme waehrend des Laufs veraendert')
        report['classifier_and_channel_unchanged']=True
        report['predictions_sha256']=baseline.sha(args.output/'predictions.csv')
        report['complete']=True
        for label,count in report['real_channel']['predictions'].items():print(label,count,flush=True)
        fraction=report['real_channel']['reference_match_fraction_valid_blocks']
        if fraction is not None:print('Anteil passend zur Hoerreferenz:',f'{fraction:.1%}','(keine Testgenauigkeit)',flush=True)
        print('Bekannte Klassen erzwungen; keine garantierte Unbekannt-Erkennung.',flush=True)
    except Exception as error:
        report['error']=str(error);raise
    finally:
        baseline.write(args.output/'report.json',report);print('Bericht:',args.output/'report.json',flush=True)
    (args.output/'CLASSICAL_REAL_CHANNEL_COMPLETE').write_text('OK\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--channel',type=Path)
    p.add_argument('--data',type=Path,default=Path('data/modulation12-v1'))
    p.add_argument('--source',type=Path)
    p.add_argument('--expected',choices=baseline.CLASSES+['unlabeled'],default='fm')
    p.add_argument('--output',type=Path,default=Path('models')/('classical-real-channel-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%fZ')))
    args=p.parse_args()
    try:
        args.channel=locate_channel(args.channel)
        if args.source is None:
            sources=[p.parent for p in Path('models').glob('classical-modulation-*/CLASSICAL_BASELINE_COMPLETE')]
            if len(sources)!=1:raise ValueError('Genau eine klassische Baseline erwartet; --source PFAD angeben')
            args.source=sources[0]
        run(args)
    except (OSError,ValueError,KeyError,RuntimeError,AssertionError) as error:
        p.exit(1,'Realer Kanaltest fehlgeschlagen: '+str(error)+'\n')
