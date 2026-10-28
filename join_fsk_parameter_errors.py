"""Join exact FSK replay parameters to existing predictions; standard library only."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

FSK = {'2fsk','2gfsk','2gmsk'}
CLASSES = ['am-dsb','fm','bpsk','qpsk','16qam','64qam','2fsk','2gfsk','2gmsk','ofdm-64','ook','noise']
CASES = {'2081-keep_lr','2081-lower_lr','2082-keep_lr','2082-lower_lr'}
PLAN_PATH = Path(__file__).with_name('FSK_PARAMETER_JOIN_PLAN_V1.json')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def checked_path(folder,name,expected):
    path=Path(folder)/name
    if path.resolve().parent != Path(folder).resolve() or sha(path)!=expected:
        raise ValueError('Path/hash mismatch: '+str(path))
    return path


def index_records(path):
    rows={}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        row=json.loads(line);sid=row['source_id']
        if sid in rows:
            raise ValueError('Duplicate source ID: '+sid)
        rows[sid]=row
    return rows


def bucket(value,edges):
    if value is None:
        return 'not_applicable'
    value=float(value)
    if not math.isfinite(value):
        raise ValueError('Nonfinite parameter')
    for i,(lo,hi) in enumerate(zip(edges,edges[1:])):
        if lo<=value<hi or i==len(edges)-2 and value==hi:
            return f'{lo}:{hi}'
    raise ValueError('Parameter outside registered bins: '+str(value))


def summarize(rows,keys):
    groups=defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    result=[]
    for key,items in sorted(groups.items(),key=lambda pair:repr(pair[0])):
        correct=sum(r['true_label']==r['predicted_label'] for r in items)
        result.append(dict(zip(keys,key),count=len(items),correct=correct,accuracy=correct/len(items),
            sparse=len(items)<20,unique_conditions=len({r['condition_id'] for r in items}),
            predicted_counts=dict(Counter(r['predicted_label'] for r in items))))
    return result


def join_rows(predictions,replayed,original,expected_matrix,plan):
    matrix=[[0]*12 for _ in CLASSES];seen=set();joined=[]
    for row in predictions:
        sid=row['source_id']
        if sid in seen or sid not in replayed or sid not in original:
            raise ValueError('Duplicate/missing prediction source ID')
        seen.add(sid);rep=replayed[sid];src=original[sid]
        label=row['true_label'];pred=row['predicted_label']
        if label not in CLASSES or pred not in CLASSES:
            raise ValueError('Unknown label')
        if label!=rep['label'] or label!=src['label'] or row['condition_id']!=src['condition_id']:
            raise ValueError('Label/condition join mismatch')
        if (rep['exact'] is not True or rep['stored_sha256']!=src['iq_sha256'] or
                rep['replay_sha256']!=src['iq_sha256']):
            raise ValueError('IQ replay join mismatch')
        matrix[CLASSES.index(label)][CLASSES.index(pred)]+=1
        if row['correct']!=str(label==pred):
            raise ValueError('Prediction correctness field mismatch')
        if label not in FSK:
            continue
        if rep['instrumented_sha256']!=src['iq_sha256'] or len(rep['actual_fsk_parameters'])!=1:
            raise ValueError('FSK instrumentation mismatch')
        params=rep['actual_fsk_parameters'][0]
        snr=float(row['snr_db'])
        if snr!=src['parameters']['snr_db']:
            raise ValueError('SNR join mismatch')
        span=params['gaussian_span']
        if label=='2fsk':
            if params['bt'] is not None or span is not None:
                raise ValueError('Unexpected FSK Gaussian parameters')
        elif params['bt'] is None or span not in (1,2,3,4):
            raise ValueError('Missing/invalid Gaussian parameters')
        joined.append(dict(source_id=sid,condition_id=row['condition_id'],true_label=label,
            predicted_label=pred,snr_db=snr,h_bin=bucket(params['h'],plan['h_edges']),
            bt_bin=bucket(params['bt'],plan['bt_edges']),span='not_applicable' if span is None else str(int(span))))
    if matrix!=expected_matrix:
        raise ValueError('CSV confusion matrix does not reproduce audit')
    return joined,seen


def unique_folder(root,marker,explicit):
    if explicit is not None:
        result=Path(explicit)
        if not (result/marker).is_file():
            raise ValueError('Missing completion marker: '+str(result))
        return result
    matches=sorted({p.parent for p in Path(root).rglob(marker)})
    if len(matches)!=1:
        raise ValueError(f'Expected one {marker}, found {len(matches)}; specify --audit/--replay explicitly')
    return matches[0]


def run(data,audit,replay,output):
    data,audit,replay,output=map(Path,(data,audit,replay,output))
    output.mkdir(parents=True,exist_ok=False)
    report=dict(complete=False,training_performed=False,test_opened=False,inference_performed=False,
        plan=read(PLAN_PATH),plan_sha256=sha(PLAN_PATH),cases={},
        inputs=dict(audit_report_sha256=sha(audit/'report.json'),replay_report_sha256=sha(replay/'report.json'),
                    metadata_sha256=sha(data/'metadata.json')),script_sha256=sha(__file__))
    try:
        a=read(audit/'report.json');r=read(replay/'report.json');meta=read(data/'metadata.json')
        if not all((p/m).is_file() for p,m in ((audit,'ERROR_AUDIT_COMPLETE'),(replay,'REPLAY_COMPLETE'),(data,'DATASET_COMPLETE'))):
            raise ValueError('Missing completion marker')
        if a['complete'] is not True or r['complete'] is not True or r['parameter_interpretation_allowed'] is not True:
            raise ValueError('Source audit/replay incomplete or parameter interpretation blocked')
        if set(a['cases'])!=CASES or meta['classes']!=CLASSES or sha(data/'metadata.json')!=r['metadata_sha256']:
            raise ValueError('Source cases/classes/metadata mismatch')
        originals={};replays={}
        for split in ('train','validation'):
            params=checked_path(data,split+'_parameters.jsonl',meta['splits'][split]['parameters_sha256'])
            if sha(params)!=a['protocol']['parameter_jsonl_sha256'][split]:
                raise ValueError('Audit parameter source differs from replay dataset')
            originals[split]=index_records(params)
            rp=checked_path(replay,split+'_replay.jsonl',r['splits'][split]['replay_jsonl_sha256'])
            replays[split]=index_records(rp)
            if set(originals[split])!=set(replays[split]) or len(replays[split])!=r['splits'][split]['examples']:
                raise ValueError('Incomplete replay ID coverage')
            if r['splits'][split]['mismatches']!=0 or r['splits'][split]['exact']!=len(replays[split]):
                raise ValueError('Replay not fully exact')
        if set(originals['train']) & set(originals['validation']):
            raise ValueError('Train/validation source overlap')
        if ({x['condition_id'] for x in originals['train'].values()} &
                {x['condition_id'] for x in originals['validation'].values()}):
            raise ValueError('Train/validation condition overlap')
        reference_ids={}
        for case in sorted(CASES):
            result={};report['cases'][case]=result
            for split in ('train','validation'):
                source=a['cases'][case][split]
                path=checked_path(audit,source['prediction_csv'],source['prediction_csv_sha256'])
                with path.open(newline='',encoding='utf-8') as stream:
                    predictions=list(csv.DictReader(stream))
                rows,ids=join_rows(predictions,replays[split],originals[split],source['overall']['confusion_matrix'],report['plan'])
                if len(ids)!=source['overall']['count'] or source['prediction_matrix_and_loss_reproduced'] is not True:
                    raise ValueError('Audit count/reproduction mismatch')
                if split=='validation' and ids!=set(replays[split]):
                    raise ValueError('Validation coverage incomplete')
                if split in reference_ids and reference_ids[split]!=ids:
                    raise ValueError('Models use different example sets')
                reference_ids[split]=ids
                tables={}
                for scope,subset in [('all_snr',rows),('snr_ge15',[x for x in rows if x['snr_db']>=15])]:
                    tables[scope]=dict(by_h=summarize(subset,['true_label','h_bin']),
                        by_bt=summarize(subset,['true_label','bt_bin']),
                        by_span=summarize(subset,['true_label','span']),
                        by_h_bt=summarize(subset,['true_label','h_bin','bt_bin']))
                result[split]=dict(fsk_examples=len(rows),all_class_examples=len(ids),
                    csv_matrix_reproduced=True,prediction_csv_sha256=sha(path),tables=tables)
            print(case,'Training und Validierung verknuepft',flush=True)
        report['complete']=True
    except Exception as error:
        report['error']=str(error)
        raise
    finally:
        (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('Bericht:',output/'report.json',flush=True)
    (output/'FSK_PARAMETER_JOIN_COMPLETE').write_text('OK\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/modulation12-v1'))
    p.add_argument('--audit',type=Path);p.add_argument('--replay',type=Path)
    p.add_argument('--output',type=Path,default=Path('data')/('fsk-parameter-errors-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%fZ')))
    args=p.parse_args()
    try:
        audit=unique_folder('models','ERROR_AUDIT_COMPLETE',args.audit)
        replay=unique_folder('data','REPLAY_COMPLETE',args.replay)
        run(args.data,audit,replay,args.output)
    except (OSError,ValueError,KeyError,TypeError) as error:
        p.exit(1,'FSK-Verknuepfung fehlgeschlagen: '+str(error)+'\n')


if __name__=='__main__':
    main()
