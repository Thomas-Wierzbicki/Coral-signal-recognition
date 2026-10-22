"""Experimenteller RadioML-Coral-Test auf gespeichertem realem 32-kS/s-I/Q-Kanal."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from signal_preprocessing import prepare_iq
from radioml_holdout import file_hash
from evaluate_radioml_iq_coral import quantize

LABELS=['AM-DSB-WC','FM','BPSK','QPSK','16QAM']


def prepare_block(iq):
    iq=np.asarray(iq)
    if iq.shape!=(1024,) or not np.isfinite(iq).all():
        raise ValueError('1024 endliche komplexe Samples erforderlich')
    rms=float(np.sqrt(np.mean(np.abs(iq.astype(np.complex128))**2)))
    if rms==0: raise ValueError('Nullblock kann nicht sinnvoll normalisiert werden')
    normalized=prepare_iq(iq,'rms-v1')
    return np.stack((normalized.real,normalized.imag),axis=-1)[None].astype(np.float32),rms


def evaluate(interpreter,iq,start,rate,expected,stream):
    interpreter.allocate_tensors()
    inp=interpreter.get_input_details()[0]; out=interpreter.get_output_details()[0]
    if (tuple(inp['shape'])!=(1,1,1024,2) or tuple(out['shape'])!=(1,5)
            or out['dtype']!=np.uint8 or out['quantization'][0]<=0):
        raise ValueError('Modelltensoren passen nicht zu I/Q-Fuenfklassenmodell')
    count=(len(iq)-start)//1024
    if count<1: raise ValueError('Kanal zu kurz')
    first,_=prepare_block(iq[start:start+1024])
    interpreter.set_tensor(inp['index'],quantize(first,inp)); interpreter.invoke()
    writer=csv.writer(stream)
    writer.writerow(['block','time_seconds_channel','input_rms','prediction','top_score',
                     'invoke_ms','input_clipped_fraction']+['score_'+name for name in LABELS])
    counts=np.zeros(5,dtype=np.int64); timings=[]; scores=[]; clipped=[]
    for k in range(count):
        offset=start+k*1024
        sample,rms=prepare_block(iq[offset:offset+1024])
        scale,zero=inp['quantization']
        if inp['dtype']!=np.uint8 or scale<=0:
            raise ValueError('uint8-Eingang mit positiver Skala erforderlich')
        unbounded=np.rint(sample/scale+zero)
        fraction=float(np.mean((unbounded<0)|(unbounded>255)))
        interpreter.set_tensor(inp['index'],quantize(sample,inp))
        before=time.perf_counter(); interpreter.invoke()
        elapsed=(time.perf_counter()-before)*1000
        raw=interpreter.get_tensor(out['index'])[0]
        values=(raw.astype(np.float64)-out['quantization'][1])*out['quantization'][0]
        prediction=int(np.argmax(values)); counts[prediction]+=1
        timings.append(elapsed); scores.append(float(values[prediction])); clipped.append(fraction)
        writer.writerow([k,offset/rate,rms,LABELS[prediction],values[prediction],elapsed,fraction]+values.tolist())
        if (k+1)%500==0: print(k+1,'/',count,'Bloecke',flush=True)
    return dict(blocks=count,block_samples=1024,block_seconds_nominal=1024/rate,
        input_start_sample=start,tail_samples_omitted=len(iq)-start-count*1024,
        predictions={label:int(counts[k]) for k,label in enumerate(LABELS)},
        expected_reference_label=expected,
        fraction_matching_expected_reference=float(counts[LABELS.index(expected)]/count),
        invoke_median_ms=float(np.median(timings)),invoke_p95_ms=float(np.percentile(timings,95)),
        top_score_median=float(np.median(scores)),mean_input_clipped_fraction=float(np.mean(clipped)),
        max_input_clipped_fraction=float(np.max(clipped)))


def run(folder,model,output,expected='FM',interpreter_factory=None):
    folder=Path(folder); model=Path(model); output=Path(output)
    if output.exists(): raise ValueError('Ausgabeordner existiert bereits; neuen --output verwenden')
    metadata=json.loads((folder/'channel_report.json').read_text(encoding='utf-8'))
    source=folder/'channel.cf32'
    if (not (folder/'CHANNEL_COMPLETE').is_file() or metadata.get('output_rate_hz_nominal')!=32000
            or metadata.get('format')!='little-endian complex64 interleaved I,Q'
            or source.stat().st_size!=8*metadata['samples']):
        raise ValueError('Vollstaendiger 32-kS/s-Kanal erforderlich')
    manifest=json.loads((model.parent.parent/'preprocessing.json').read_text(encoding='utf-8'))
    if (manifest.get('representation')!='iq-rms-v1' or manifest.get('input_shape')!=[1,1024,2]
            or manifest.get('labels')!=LABELS or manifest.get('channel_order')!=['I','Q']
            or manifest.get('target_rms')!=0.5 or manifest.get('mode')!='rms-v1'):
        raise ValueError('Modellmanifest passt nicht')
    iq=np.memmap(source,dtype='<c8',mode='r')
    start=metadata.get('statistic_start_sample',33)
    if not isinstance(start,int) or start<0 or start>=len(iq): raise ValueError('Ungueltiger Startindex')
    if expected not in LABELS: raise ValueError('Unbekannte Referenzklasse')
    model_hash=file_hash(model); source_hash=file_hash(source)
    if interpreter_factory is None:
        from pycoral.utils.edgetpu import make_interpreter,list_edge_tpus
        print('Coral-Geraete:',list_edge_tpus(),flush=True)
        interpreter_factory=make_interpreter
    interpreter=interpreter_factory(str(model))
    output.mkdir(parents=True)
    print('Experimenteller Real-I/Q-Test; keine unabhaengige Genauigkeitsmessung.',flush=True)
    with (output/'predictions.csv').open('x',newline='',encoding='utf-8') as stream:
        report=evaluate(interpreter,iq,start,32000,expected,stream)
    report.update(model=str(model.resolve()),model_sha256=model_hash,
        source_channel_sha256=source_hash,channel_metadata=metadata,labels=LABELS,
        preprocessing='Block-RMS 0.5, I/Q channels, no further resampling/DC/frequency correction',
        expected_reference_status='User listening reference for recording, not individual block truth',
        limitations='RadioML synthetic/real domain and time scale differ. No noise/unknown class. Scores are not calibrated probabilities. No accuracy claim. Adjacent blocks are dependent.')
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (output/'CLASSIFICATION_COMPLETE').write_text('OK\n')
    for label,n in report['predictions'].items():
        print('{}: {}/{} ({:.1%})'.format(label,n,report['blocks'],n/report['blocks']))
    print('Anteil mit erwarteter Referenz {}: {:.1%} (keine Testgenauigkeit)'.format(expected,report['fraction_matching_expected_reference']))
    print('invoke Median {:.3f} ms; p95 {:.3f} ms'.format(report['invoke_median_ms'],report['invoke_p95_ms']))
    print('Quantisierungs-Clipping: Mittel {:.3%}; Maximum pro Block {:.3%}'.format(report['mean_input_clipped_fraction'],report['max_input_clipped_fraction']))
    print('Protokoll:',(output/'report.json').resolve())
    print('Modellbewertungen sind keine garantierten Wahrscheinlichkeiten.')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('folder',type=Path)
    p.add_argument('--model',type=Path,default=Path('models/radioml2018-iq-v1/compiled/radioml_int8_edgetpu.tflite'))
    p.add_argument('--output',type=Path); p.add_argument('--expected',choices=LABELS,default='FM')
    args=p.parse_args()
    try: run(args.folder,args.model,args.output or args.folder/'coral-real-v1',args.expected)
    except (OSError,ValueError,KeyError,ImportError,RuntimeError) as error:
        p.exit(1,'Real-I/Q-Test fehlgeschlagen: '+str(error)+'\n')


if __name__=='__main__': main()
