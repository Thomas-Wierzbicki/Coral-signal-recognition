"""Gepaarter Coral-Test: Frequenzlage bei identischen realen I/Q-Bloecken variieren."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from classify_rtl_channel import LABELS,evaluate
from radioml_holdout import file_hash

TARGET=0.031295


def phase_offset(iq):
    z=np.asarray(iq,dtype=np.complex128)
    if z.shape!=(1024,) or not np.isfinite(z).all() or np.any(np.abs(z)==0):
        raise ValueError('1024 endliche, von null verschiedene komplexe Samples erforderlich')
    vector=np.mean(np.exp(1j*np.angle(z[1:]*np.conj(z[:-1]))))
    if abs(vector)<1e-6: raise ValueError('Phasenmittel nicht stabil bestimmbar')
    return float(np.angle(vector)/(2*np.pi))


def shift_block(iq,offset,target):
    return (np.asarray(iq,dtype=np.complex128)*
            np.exp(2j*np.pi*(target-offset)*np.arange(1024))).astype(np.complex64)


def transitions(baseline,pred):
    matrix=np.zeros((5,5),dtype=np.int64)
    lookup={label:k for k,label in enumerate(LABELS)}
    for a,b in zip(baseline,pred): matrix[lookup[a],lookup[b]]+=1
    return dict(changed=int(np.sum(np.asarray(baseline)!=np.asarray(pred))),
                matrix=matrix.tolist(),rows='original prediction',columns='variant prediction')


def run(folder,model,output,interpreter_factory=None):
    folder=Path(folder); model=Path(model); output=Path(output)
    if output.exists(): raise ValueError('Ausgabeordner existiert bereits')
    metadata=json.loads((folder/'channel_report.json').read_text(encoding='utf-8'))
    source=folder/'channel.cf32'
    if (not (folder/'CHANNEL_COMPLETE').is_file() or metadata.get('output_rate_hz_nominal')!=32000
            or metadata.get('format')!='little-endian complex64 interleaved I,Q'
            or source.stat().st_size!=metadata['samples']*8):
        raise ValueError('Vollstaendiger 32-kS/s-Kanal erforderlich')
    manifest=json.loads((model.parent.parent/'preprocessing.json').read_text(encoding='utf-8'))
    if (manifest.get('representation')!='iq-rms-v1' or manifest.get('mode')!='rms-v1'
            or manifest.get('target_rms')!=.5 or manifest.get('input_shape')!=[1,1024,2]
            or manifest.get('channel_order')!=['I','Q'] or manifest.get('labels')!=LABELS):
        raise ValueError('Modellmanifest passt nicht')
    iq=np.memmap(source,dtype='<c8',mode='r'); start=metadata.get('statistic_start_sample',33)
    if not isinstance(start,int) or start<0: raise ValueError('Ungueltiger Startindex')
    count=(len(iq)-start)//1024
    if count<1: raise ValueError('Kanal zu kurz')
    offsets=np.array([phase_offset(iq[start+k*1024:start+(k+1)*1024]) for k in range(count)])
    if interpreter_factory is None:
        from pycoral.utils.edgetpu import make_interpreter,list_edge_tpus
        print('Coral-Geraete:',list_edge_tpus(),flush=True)
        interpreter_factory=make_interpreter
    interpreter=interpreter_factory(str(model))
    output.mkdir(parents=True)
    variants={}; predictions={}
    for name,target in [('original',None),('centered',0.),('radioml_fm_offset',TARGET)]:
        print('\nVariante:',name,flush=True)
        if target is None: transformed=iq
        else:
            transformed=np.array(iq,copy=True)
            for k,offset in enumerate(offsets):
                position=start+k*1024
                transformed[position:position+1024]=shift_block(iq[position:position+1024],offset,target)
        path=output/(name+'.csv')
        with path.open('x',newline='',encoding='utf-8') as stream:
            stats=evaluate(interpreter,transformed,start,32000,'FM',stream)
        with path.open(newline='',encoding='utf-8') as stream:
            predictions[name]=[row['prediction'] for row in csv.DictReader(stream)]
        stats['target_cycles_per_sample']=target
        variants[name]=stats
        for label,n in stats['predictions'].items():
            print('{}: {}/{} ({:.1%})'.format(label,n,count,n/count))
        print('FM-Anteil: {:.1%}; Quantisierungs-Clipping Mittel {:.3%}'.format(
            stats['fraction_matching_expected_reference'],stats['mean_input_clipped_fraction']))
    with (output/'shifts.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream)
        writer.writerow(['block','time_seconds_channel','estimated_cycles_per_sample','centered_shift','radioml_fm_shift'])
        for k,value in enumerate(offsets): writer.writerow([k,(start+k*1024)/32000,value,-value,TARGET-value])
    report=dict(model_sha256=file_hash(model),source_channel_sha256=file_hash(source),
        labels=LABELS,blocks=count,channel_metadata=metadata,variants=variants,
        transitions={name:transitions(predictions['original'],predictions[name])
                     for name in ('centered','radioml_fm_offset')},
        estimated_offset_median=float(np.median(offsets)),
        protocol='Three fixed variants; blockwise circular phase-step mean; local oscillator phase starts at 0 per block; no resampling, DC subtraction or retraining.',
        target_provenance='0.031295 cycles/sample from previously reported RadioML training FM SNR>=10 median, rounded. Fixed before test.',
        limitation='Paired diagnostic experiment on previously viewed real recording, not independent accuracy. Centering may remove slow modulation as well as frequency offset. Quantization can change coordinate effects. No unknown/noise class.')
    prior=folder/'coral-real-v1'
    if (prior/'CLASSIFICATION_COMPLETE').is_file():
        old=json.loads((prior/'report.json').read_text(encoding='utf-8'))
        if (old.get('source_channel_sha256')!=report['source_channel_sha256']
                or old.get('model_sha256')!=report['model_sha256']
                or old.get('input_start_sample')!=start):
            raise ValueError('Frueherer Realtest passt nicht zum Modell/Kanal/Startindex')
        with (prior/'predictions.csv').open(newline='',encoding='utf-8') as stream:
            old_predictions=[r['prediction'] for r in csv.DictReader(stream)]
        if len(old_predictions)!=count: raise ValueError('Frueherer Realtest hat andere Blockzahl')
        agreement=float(np.mean(np.array(old_predictions)==np.array(predictions['original'])))
        report['original_agreement_with_prior']=agreement
        print('\nUnveraenderte Referenz vs. frueherer Lauf: {:.1%}'.format(agreement))
    (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (output/'FREQUENCY_TEST_COMPLETE').write_text('OK\n')
    for name,stats in report['transitions'].items(): print(name+': Klassenwechsel '+str(stats['changed'])+'/'+str(count))
    print('Protokoll:',(output/'summary.json').resolve())
    print('Diagnose, kein Nachweis verbesserter allgemeiner Erkennung.')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('folder',type=Path)
    p.add_argument('--model',type=Path,default=Path('models/radioml2018-iq-v1/compiled/radioml_int8_edgetpu.tflite'))
    p.add_argument('--output',type=Path); args=p.parse_args()
    try: run(args.folder,args.model,args.output or args.folder/'frequency-test-v1')
    except (OSError,ValueError,KeyError,ImportError,RuntimeError) as error:
        p.exit(1,'Frequenztest fehlgeschlagen: '+str(error)+'\n')


if __name__=='__main__': main()
