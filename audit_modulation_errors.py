"""Read-only checkpoint/parameter audit; no training or test-split access."""
import argparse
from collections import Counter,defaultdict
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from build_modulation_dataset import sha
from check_modulation_learning import state_digest
from diagnose_training_collapse import write_json
from train_modulation12 import metrics
from se_gain_experiment import restore_gain_checkpoint
import continue_se_gain as continuation

CLASSES=continuation.development.CLASSES
FOCUS={'fm','2fsk','2gfsk','2gmsk'}
BINS={'bandwidth_bin':('requested_bandwidth_hz',[1500,4000,8000,12000]),
      'offset_bin':('absolute_offset_hz',[0,1600,3200,4800]),
      'clean_width_bin':('clean_width90_hz',[0,2000,4000,8000,16000,32000])}


def bucket(value,edges):
    if value is None:return 'not_applicable'
    if not np.isfinite(value):raise ValueError('Nonfinite bin value')
    for index,(low,high) in enumerate(zip(edges,edges[1:])):
        if low<=value<high or (index==len(edges)-2 and value==high):return f'{low}:{high}'
    return 'outside_range'


def read_records(path,expected_hash,ids,labels,conditions,snr,x):
    if sha(path)!=expected_hash:raise ValueError('Parameter JSONL hash mismatch')
    wanted=set(ids.tolist());records={};seen=set()
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            record=json.loads(line);identity=record['source_id']
            if identity in seen:raise ValueError('Duplicate parameter source ID')
            seen.add(identity)
            if identity in wanted:records[identity]=record
    if set(records)!=wanted or len(ids)!=len(wanted):raise ValueError('Missing/duplicate selected source IDs')
    rows=[]
    for index,identity in enumerate(ids.tolist()):
        record=records[identity];label=CLASSES[int(labels[index])];params=record.get('parameters')
        if record['label']!=label or record['condition_id']!=conditions[index]:raise ValueError('Parameter label/condition join mismatch')
        raw=np.empty(x.shape[-1],dtype=np.complex64)
        raw.real=x[index,0,0].numpy();raw.imag=x[index,1,0].numpy()
        if hashlib.sha256(raw.tobytes()).hexdigest()!=record['iq_sha256']:raise ValueError('Parameter/IQ hash mismatch')
        if label=='noise':
            if params is not None or not np.isnan(snr[index]):raise ValueError('Noise has signal parameters')
            snr_value=bandwidth=offset=rolloff=None
        else:
            if not isinstance(params,dict) or float(params['snr_db'])!=float(snr[index]):raise ValueError('Parameter SNR mismatch')
            snr_value=float(snr[index]);bandwidth=float(params['bandwidth_hz']);offset=float(params['offset_hz'])
            if (not np.isfinite([snr_value,bandwidth,offset]).all() or not 1500<=bandwidth<=12000
                or abs(offset)>4800):raise ValueError('Invalid recorded signal parameters')
            rolloff=float(params['rolloff']) if label in {'bpsk','qpsk','16qam','64qam','ook'} else None
        clean=record.get('clean_metrics') or {};received=record.get('received_metrics') or {}
        row=dict(source_id=identity,condition_id=str(conditions[index]),true_label=label,snr_db=snr_value,
            requested_bandwidth_hz=bandwidth,offset_hz=offset,absolute_offset_hz=abs(offset) if offset is not None else None,
            rolloff_used_by_generator=rolloff,clean_width90_hz=clean.get('power90_width_hz'),
            received_width90_hz=received.get('power90_width_hz'),clean_envelope_cv=clean.get('envelope_cv'),
            generator=record.get('generator'),fm_crop_rms_deviation_hz=record.get('crop_rms_deviation_hz'),
            fm_crop_peak_deviation_hz=record.get('crop_peak_deviation_hz'),
            fsk_modulation_index_recorded=record.get('modulation_index',record.get('h')),
            fsk_bt_recorded=record.get('bt',record.get('BT')),
            signal_seed=record.get('signal_seed'),noise_seed=record.get('noise_seed'))
        for name,(field,edges) in BINS.items():row[name]=bucket(row[field],edges)
        rows.append(row)
    return rows


def predict_rows(model,x,labels,metadata,device):
    model.eval();predictions=[];losses=[];scores=[]
    with torch.inference_mode():
        for start in range(0,len(labels),32):
            logits=model(x[start:start+32].to(device));target=torch.from_numpy(labels[start:start+32]).to(device)
            if not torch.isfinite(logits).all():raise ValueError('Nonfinite logits')
            predictions.extend(logits.argmax(1).cpu().tolist())
            scores.extend(logits.softmax(1).max(1).values.cpu().tolist())
            losses.extend(nn.functional.cross_entropy(logits,target,reduction='none').cpu().tolist())
    rows=[]
    for record,prediction,loss,score in zip(metadata,predictions,losses,scores):
        rows.append(dict(record,predicted_label=CLASSES[prediction],correct=record['true_label']==CLASSES[prediction],
                         model_score=score,cross_entropy=loss))
    result=metrics(labels,np.asarray(predictions));result['loss']=float(np.mean(losses))
    return rows,result


def table(rows,keys):
    grouped=defaultdict(list)
    for row in rows:grouped[tuple(row[key] for key in keys)].append(row)
    cells=[]
    for key,items in sorted(grouped.items(),key=lambda pair:repr(pair[0])):
        count=len(items);correct=sum(r['correct'] for r in items)
        cells.append(dict(zip(keys,key),count=count,correct=correct,accuracy=correct/count,
            unique_conditions=len({r['condition_id'] for r in items}),sparse=count<20,
            predicted_counts=dict(Counter(r['predicted_label'] for r in items)),
            errors_score_ge08=sum(not r['correct'] and r['model_score']>=.8 for r in items)))
    return cells


def summarize_rows(rows):
    focus=[r for r in rows if r['true_label'] in FOCUS]
    high=[r for r in rows if r['snr_db'] is not None and r['snr_db']>=15]
    return dict(by_class_snr=table(rows,['true_label','snr_db']),
        by_class_requested_bandwidth=table(rows,['true_label','bandwidth_bin']),
        by_class_absolute_offset=table(rows,['true_label','offset_bin']),
        by_class_clean_width90=table(rows,['true_label','clean_width_bin']),
        focus_class_snr_requested_bandwidth=table(focus,['true_label','snr_db','bandwidth_bin']),
        high_snr_ge15_by_class=table(high,['true_label']),
        focus_family=dict(count=len(focus),errors=sum(not r['correct'] for r in focus),
            errors_within_family=sum(not r['correct'] and r['predicted_label'] in FOCUS for r in focus)),
        generator_coverage=dict(generators=dict(Counter(r['generator'] for r in rows)),
            fsk_count=sum(r['true_label'] in {'2fsk','2gfsk','2gmsk'} for r in rows),
            fsk_modulation_index_recorded=sum(r['true_label'] in {'2fsk','2gfsk','2gmsk'} and r['fsk_modulation_index_recorded'] is not None for r in rows),
            fsk_bt_recorded=sum(r['true_label'] in {'2fsk','2gfsk','2gmsk'} and r['fsk_bt_recorded'] is not None for r in rows)),
        high_score_error_examples=[dict(source_id=r['source_id'],true_label=r['true_label'],predicted_label=r['predicted_label'],
            snr_db=r['snr_db'],model_score=r['model_score'],requested_bandwidth_hz=r['requested_bandwidth_hz'],offset_hz=r['offset_hz'])
            for label in CLASSES for r in sorted([r for r in rows if r['true_label']==label and not r['correct']],
                                                  key=lambda r:(-r['model_score'],r['source_id']))[:3]])


def choose_run(path):
    if path is not None:return Path(path)
    candidates=[p for p in Path('models').glob('se-gain-continuation-*') if (p/'CONTINUATION_COMPLETE').is_file()]
    if len(candidates)!=1:raise ValueError('Specify --run models/se-gain-continuation-TIMESTAMP')
    return candidates[0]


def run(args):
    source=choose_run(args.run)
    if not (source/'CONTINUATION_COMPLETE').is_file():raise ValueError('Continuation is incomplete')
    report=json.loads((source/'report.json').read_text());protocol=report['protocol']
    continuation_plan=Path(__file__).with_name('SE_GAIN_CONTINUATION_PLAN_V1.json')
    if (not protocol.get('default_design') or protocol.get('epochs')!=50 or not report['summary'].get('complete')
        or protocol['plan_sha256']!=sha(continuation_plan) or protocol['plan']!=json.loads(continuation_plan.read_text())
        or set(report['runs'])!={'2081','2082'}):raise ValueError('Unexpected continuation design')
    if json.loads((source/'protocol.json').read_text())!=protocol:raise ValueError('Continuation protocol/report mismatch')
    for name,expected in protocol['sources'].items():
        if sha(Path(__file__).with_name(name))!=expected:raise ValueError('Source code mismatch: '+name)
    development_source=args.development_source or Path(protocol['source'])
    if sha(development_source/'report.json')!=protocol['source_report_sha256']:raise ValueError('Development source report mismatch')
    prepared=continuation.prepare_source(development_source,args.data)
    if prepared['checkpoint_sha256']!=protocol['source_checkpoints']:raise ValueError('Source chain mismatch')
    meta=continuation.development.load_metadata(args.data)
    with np.load(development_source/'training_selection.npz',allow_pickle=False) as d:
        train_ids,train_labels,train_conditions,train_snr=(d[k] for k in ['source_id','y','condition_id','snr_db'])
    with np.load(args.data/'validation.npz',allow_pickle=False) as d:
        val_ids,val_labels,val_conditions,val_snr=(d[k] for k in ['source_id','y','condition_id','snr_db'])
    inputs={'train':(prepared['x'],train_labels,train_ids,train_conditions,train_snr),
            'validation':(prepared['vx'],val_labels,val_ids,val_conditions,val_snr)}
    metadata={};parameter_hashes={}
    for split,(x,labels,ids,conditions,snr) in inputs.items():
        path=args.data/(split+'_parameters.jsonl');expected=meta['splits'][split]['parameters_sha256']
        metadata[split]=read_records(path,expected,ids,labels,conditions,snr,x);parameter_hashes[split]=expected
    if args.device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    if args.output is None:args.output=source/('error-audit-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%fZ'))
    if args.output.exists():raise ValueError('Output exists; choose a new folder')
    args.output.mkdir(parents=True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    plan=Path(__file__).with_name('MODULATION_ERROR_AUDIT_PLAN_V1.json')
    provenance=dict(plan=json.loads(plan.read_text()),plan_sha256=sha(plan),source=str(source.resolve()),
        source_report_sha256=sha(source/'report.json'),development_source=str(development_source.resolve()),
        parameter_jsonl_sha256=parameter_hashes,test_opened=False,training_performed=False,
        torch_version=torch.__version__,numpy_version=np.__version__,device=args.device,cuda=torch.version.cuda,
        device_name=torch.cuda.get_device_name(0) if args.device=='cuda' else 'CPU',
        sources={name:sha(Path(__file__).with_name(name)) for name in [Path(__file__).name,'continue_se_gain.py',
            'train_se_gain_development.py','diagnose_training_lr.py','diagnose_training_collapse.py',
            'modulation_reference.py','se_gain_experiment.py','train_modulation12.py','check_modulation_learning.py',
            'build_modulation_dataset.py','torchsig_probe.py']})
    write_json(args.output/'protocol.json',provenance);cases={}
    for seed in ['2081','2082']:
        if set(report['runs'][seed])!={'keep_lr','lower_lr'}:raise ValueError('Missing continuation branch')
        for branch in ['keep_lr','lower_lr']:
            expected=report['runs'][seed][branch];folder=source/('seed'+seed)/branch
            path=folder/'best_checkpoint.pt'
            if not (folder/'CASE_COMPLETE').is_file() or sha(path)!=expected['best_checkpoint_sha256']:
                raise ValueError('Selected checkpoint integrity mismatch')
            if sha(folder/'history.json')!=expected['history_sha256']:raise ValueError('History hash mismatch')
            history=json.loads((folder/'history.json').read_text())
            if (len(history)!=51 or any(row['epoch']!=100+i or row['steps']!=(100+i)*53 for i,row in enumerate(history))
                or history[-1]['validation']!=expected['final_validation']):raise ValueError('History/report mismatch')
            saved=torch.load(path,map_location='cpu',weights_only=True)
            if (saved['model_seed']!=int(seed) or saved['se_gain']!=2. or saved['activation']!='tf-compatible'
                or saved['epoch']!=expected['best_epoch'] or saved['steps']!=saved['epoch']*53
                or not 100<=saved['epoch']<=150 or saved['protocol_sha256']!=sha(source/'protocol.json')
                or saved['selection_sha256']!=prepared['report']['protocol']['selection_sha256']):
                raise ValueError('Selected checkpoint provenance mismatch')
            model=restore_gain_checkpoint(saved).to(args.device);before=state_digest(model.state_dict());case={}
            for split,(x,labels,ids,conditions,snr) in inputs.items():
                rows,overall=predict_rows(model,x,labels,metadata[split],args.device)
                prior=expected['selected_training' if split=='train' else 'selected_validation']
                continuation.require_baseline(overall,prior)
                filename=f'seed{seed}-{branch}-{split}-predictions.csv'
                with (args.output/filename).open('w',newline='',encoding='utf-8') as stream:
                    fields=list(rows[0]);writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
                    for row in rows:
                        writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})
                case[split]=dict(overall=overall,prediction_matrix_and_loss_reproduced=True,
                    prediction_csv=filename,prediction_csv_sha256=sha(args.output/filename),tables=summarize_rows(rows))
            if state_digest(model.state_dict())!=before or any(p.grad is not None for p in model.parameters()):
                raise ValueError('Model state was modified')
            if sha(path)!=expected['best_checkpoint_sha256']:raise ValueError('Source checkpoint changed')
            case.update(checkpoint_sha256=sha(path),state_sha256=before,selected_epoch=saved['epoch'])
            cases[seed+'-'+branch]=case
            write_json(args.output/'report.json',dict(protocol=provenance,cases=cases,complete=len(cases)==4))
            print('Audit:',seed,branch,'| train und validation reproduziert',flush=True);del model
    (args.output/'ERROR_AUDIT_COMPLETE').write_text('OK\n')
    print('Ergebnisse:',args.output.resolve())
    print('Bitte report.json hochladen. Deskriptive Entwicklungsauswertung, keine isolierte Ursache.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path)
    p.add_argument('--development-source',type=Path)
    p.add_argument('--data',type=Path,default=Path('data/modulation12-v1'))
    p.add_argument('--output',type=Path)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    try:run(p.parse_args())
    except (OSError,ValueError,RuntimeError,KeyError,AssertionError) as error:
        p.exit(1,'Fehleraudit fehlgeschlagen: '+str(error)+'\n')
