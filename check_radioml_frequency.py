"""Frequenz-Gegencheck auf RadioML-Training: drei feste Varianten fuer alle Klassen."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from classify_rtl_channel import LABELS,prepare_block
from evaluate_radioml_iq_coral import quantize
from sweep_rtl_frequency import phase_offset,shift_block,TARGET
from train_radioml_iq import score
from radioml_holdout import file_hash


def load_training(folder):
    folder=Path(folder)
    meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    if (not (folder/'IMPORT_COMPLETE').is_file() or meta.get('source')!='RadioML2018.01A'
            or meta.get('representation')!='iq-rms-v1' or meta.get('labels')!=LABELS
            or meta.get('input_shape')!=[1,1024,2] or meta.get('target_rms')!=.5):
        raise ValueError('RadioML-I/Q-Training passt nicht')
    with np.load(folder/'train.npz',allow_pickle=False) as data:
        x,y,snr,ids=(data[k] for k in ('x','y','snr_db','source_index'))
    if (x.shape!=(len(y),1,1024,2) or len(y)==0 or snr.shape!=y.shape or ids.shape!=y.shape
            or not np.issubdtype(y.dtype,np.integer) or not np.issubdtype(ids.dtype,np.integer)
            or not np.isfinite(x).all() or not np.isfinite(snr).all()
            or set(y.tolist())!=set(range(5)) or len(np.unique(ids))!=len(ids) or np.any(ids<0)):
        raise ValueError('Ungueltige Trainingsarrays')
    return meta,x,y,snr,ids


def group_stats(y,pred,snr,clipped):
    result=score(y,pred,LABELS)
    result['fm_prediction_fraction_by_true_class']={label:float(np.mean(pred[y==k]==1))
                                                   for k,label in enumerate(LABELS)}
    result['mean_input_clipped_fraction']=float(np.mean(clipped))
    result['by_snr']={str(float(level)):score(y[snr==level],pred[snr==level],LABELS)
                      for level in np.unique(snr)}
    high=snr>=10
    if np.any(high):
        result['snr_ge10']=score(y[high],pred[high],LABELS)
        result['snr_ge10']['fm_prediction_fraction_by_true_class']={label:
            (float(np.mean(pred[high&(y==k)]==1)) if np.any(high&(y==k)) else None)
            for k,label in enumerate(LABELS)}
    return result


def run(data,model,output,interpreter_factory=None):
    data=Path(data); model=Path(model); output=Path(output)
    if output.exists(): raise ValueError('Ausgabeordner existiert bereits')
    metadata,x,y,snr,ids=load_training(data)
    manifest=json.loads((model.parent.parent/'preprocessing.json').read_text(encoding='utf-8'))
    if (manifest.get('representation')!='iq-rms-v1' or manifest.get('mode')!='rms-v1'
            or manifest.get('labels')!=LABELS or manifest.get('input_shape')!=[1,1024,2]
            or manifest.get('channel_order')!=['I','Q'] or manifest.get('target_rms')!=.5):
        raise ValueError('Modellmanifest passt nicht')
    if interpreter_factory is None:
        from pycoral.utils.edgetpu import make_interpreter,list_edge_tpus
        print('Coral-Geraete:',list_edge_tpus(),flush=True)
        interpreter_factory=make_interpreter
    interpreter=interpreter_factory(str(model)); interpreter.allocate_tensors()
    inp=interpreter.get_input_details()[0]; out=interpreter.get_output_details()[0]
    if (tuple(inp['shape'])!=(1,1,1024,2) or tuple(out['shape'])!=(1,5)
            or inp['dtype']!=np.uint8 or inp['quantization'][0]<=0
            or out['dtype']!=np.uint8 or out['quantization'][0]<=0):
        raise ValueError('Modelltensoren passen nicht')
    z=x[:,0,:,0]+1j*x[:,0,:,1]
    offsets=np.array([phase_offset(block) for block in z])
    # Ein Warmup; keine Laufzeitvergleiche zwischen Varianten in diesem Test.
    interpreter.set_tensor(inp['index'],quantize(x[0],inp)); interpreter.invoke()
    output.mkdir(parents=True); variants={}; all_predictions={}
    with (output/'cases.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream)
        writer.writerow(['variant','source_index','true_label','snr_db','offset_estimate_cycles_per_sample',
                         'applied_shift_cycles_per_sample','prediction','input_clipped_fraction']+
                        ['score_'+label for label in LABELS])
        for name,target in [('original',None),('centered',0.),('radioml_fm_offset',TARGET)]:
            print('\nVariante:',name,flush=True); predictions=[]; clipped=[]
            for i,block in enumerate(z):
                # Original exakt wie gespeicherter Trainingseingang, ohne erneute Normierung.
                sample=x[i] if target is None else prepare_block(shift_block(block,offsets[i],target))[0]
                scale,zero=inp['quantization']; q=np.rint(sample/scale+zero)
                fraction=float(np.mean((q<0)|(q>255)))
                interpreter.set_tensor(inp['index'],quantize(sample,inp)); interpreter.invoke()
                values=(interpreter.get_tensor(out['index'])[0].astype(float)-out['quantization'][1])*out['quantization'][0]
                pred=int(np.argmax(values)); predictions.append(pred); clipped.append(fraction)
                writer.writerow([name,int(ids[i]),LABELS[int(y[i])],float(snr[i]),float(offsets[i]),
                                 0. if target is None else float(target-offsets[i]),LABELS[pred],fraction]+values.tolist())
                if (i+1)%1000==0: print(i+1,'/',len(y),flush=True)
            pred=np.asarray(predictions); all_predictions[name]=pred
            stats=group_stats(y,pred,snr,np.asarray(clipped)); variants[name]=stats
            print('Matrix: Zeilen wahre Klasse, Spalten Vorhersage',LABELS)
            print(np.asarray(stats['confusion_matrix']))
            print('FM-Ausgabeanteil je wahrer Klasse:')
            for label,value in stats['fm_prediction_fraction_by_true_class'].items(): print('{}: {:.1%}'.format(label,value))
            print('Trainings-Uebereinstimmung: {:.1%}; Clipping {:.3%}'.format(stats['accuracy'],stats['mean_input_clipped_fraction']))
    paired={}
    for name in ('centered','radioml_fm_offset'):
        before=all_predictions['original']; after=all_predictions[name]
        paired[name]=dict(changed=int(np.sum(before!=after)),
            newly_fm_by_true_class={label:int(np.sum((y==k)&(before!=1)&(after==1))) for k,label in enumerate(LABELS)},
            lost_fm_by_true_class={label:int(np.sum((y==k)&(before==1)&(after!=1))) for k,label in enumerate(LABELS)})
    report=dict(model_sha256=file_hash(model),training_npz_sha256=file_hash(data/'train.npz'),
        dataset_metadata=metadata,labels=LABELS,variants=variants,paired=paired,
        target_cycles_per_sample=TARGET,protocol='Fixed three frequency variants, training split only; no retraining or resampling.',
        limitation='Diagnostic on seen training data, not test accuracy. Frequency changes can interact with quantization. Center estimate can include slow modulation. No independent generalization claim.')
    (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (output/'FREQUENCY_CHECK_COMPLETE').write_text('OK\n')
    print('Protokoll:',(output/'summary.json').resolve())
    print('Diagnose auf Trainingsdaten, keine neue Testgenauigkeit.')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/radioml2018-iq-v1'))
    p.add_argument('--model',type=Path,default=Path('models/radioml2018-iq-v1/compiled/radioml_int8_edgetpu.tflite'))
    p.add_argument('--output',type=Path,default=Path('models/radioml2018-iq-v1/frequency-check-train-v1'))
    args=p.parse_args()
    try: run(args.data,args.model,args.output)
    except (OSError,ValueError,KeyError,ImportError,RuntimeError) as error:
        p.exit(1,'RadioML-Frequenzcheck fehlgeschlagen: '+str(error)+'\n')


if __name__=='__main__': main()
