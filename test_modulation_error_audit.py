import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import audit_modulation_errors as audit
from se_gain_experiment import make_gain_model
from check_modulation_learning import state_digest
from train_modulation12 import as_tensor


def make_split(root,name):
    rng=np.random.default_rng(12 if name=='train' else 13)
    labels=np.tile(np.arange(12),2).astype(np.int64)
    iq=(rng.normal(size=(24,64))+1j*rng.normal(size=(24,64))).astype(np.complex64)
    x=as_tensor(iq);ids=np.asarray([name+':'+str(i) for i in range(24)])
    conditions=np.asarray([name+':group'+str(i//12) for i in range(24)])
    snr=np.full(24,20.);snr[labels==11]=np.nan
    records=[]
    for i,label_id in enumerate(labels):
        label=audit.CLASSES[int(label_id)]
        records.append(dict(source_id=ids[i],condition_id=conditions[i],label=label,
            parameters=None if label=='noise' else dict(snr_db=20.,bandwidth_hz=4000.,offset_hz=-1600.,rolloff=.3),
            clean_metrics=None if label=='noise' else dict(power90_width_hz=3000.,envelope_cv=.1),
            received_metrics=dict(power90_width_hz=5000.),generator='fixture',signal_seed=[i,1],noise_seed=[i,2],
            iq_sha256=hashlib.sha256(iq[i].tobytes()).hexdigest()))
    path=root/(name+'_parameters.jsonl')
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    return dict(x=x,labels=labels,ids=ids,conditions=conditions,snr=snr,path=path,records=records)


def joined(split):
    return audit.read_records(split['path'],audit.sha(split['path']),split['ids'],split['labels'],
                              split['conditions'],split['snr'],split['x'])


class ErrorAuditTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_bins_and_complete_metadata_join_detect_corruption(self):
        self.assertEqual(audit.bucket(4000,[1500,4000,8000,12000]),'4000:8000')
        self.assertEqual(audit.bucket(12000,[1500,4000,8000,12000]),'8000:12000')
        self.assertEqual(audit.bucket(None,[0,1]),'not_applicable')
        with tempfile.TemporaryDirectory() as temporary:
            split=make_split(Path(temporary),'train');rows=joined(split)
            self.assertEqual(len(rows),24)
            self.assertIsNone(rows[11]['snr_db']);self.assertIsNone(rows[6]['rolloff_used_by_generator'])
            self.assertEqual(rows[4]['rolloff_used_by_generator'],.3)
            changed=copy.deepcopy(split);changed['x'][0,0,0,0]+=1
            with self.assertRaisesRegex(ValueError,'IQ hash'):joined(changed)
            bad=copy.deepcopy(split);bad['labels'][0]=1
            with self.assertRaisesRegex(ValueError,'label/condition'):joined(bad)
            split['path'].write_text(split['path'].read_text()+json.dumps(split['records'][0])+'\n')
            with self.assertRaisesRegex(ValueError,'Duplicate'):joined(split)

    def test_fixed_tables_preserve_counts_noise_and_missing_fsk_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            split=make_split(Path(temporary),'train');rows=joined(split)
            for row in rows:
                row.update(predicted_label='fm',correct=row['true_label']=='fm',model_score=.9)
            summary=audit.summarize_rows(rows)
            self.assertEqual(sum(c['count'] for c in summary['by_class_snr']),24)
            self.assertEqual(sum(c['count'] for c in summary['high_snr_ge15_by_class']),22)
            self.assertEqual(summary['focus_family']['count'],8)
            self.assertEqual(summary['focus_family']['errors_within_family'],6)
            self.assertEqual(summary['generator_coverage']['fsk_count'],6)
            self.assertEqual(summary['generator_coverage']['fsk_bt_recorded'],0)
            self.assertEqual(summary['generator_coverage']['fsk_modulation_index_recorded'],0)
            self.assertTrue(all(c['sparse'] for c in summary['by_class_snr']))

    def test_four_checkpoint_file_workflow_is_read_only_and_reproduces_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);data=root/'data';data.mkdir();source=root/'continuation';source.mkdir()
            development=root/'development';development.mkdir()
            train=make_split(data,'train');val=make_split(data,'validation')
            np.savez(development/'training_selection.npz',source_id=train['ids'],y=train['labels'],
                     condition_id=train['conditions'],snr_db=train['snr'])
            np.savez(data/'validation.npz',source_id=val['ids'],y=val['labels'],condition_id=val['conditions'],snr_db=val['snr'])
            original={'protocol':{'selection_sha256':audit.sha(development/'training_selection.npz')}}
            audit.write_json(development/'report.json',original)
            plan_path=Path(audit.__file__).with_name('SE_GAIN_CONTINUATION_PLAN_V1.json')
            protocol=dict(default_design=True,epochs=50,plan=json.loads(plan_path.read_text()),plan_sha256=audit.sha(plan_path),
                source=str(development),source_report_sha256=audit.sha(development/'report.json'),source_checkpoints={'2081':'fixture1','2082':'fixture2'},
                sources={name:audit.sha(Path(audit.__file__).with_name(name)) for name in ['continue_se_gain.py','modulation_reference.py','se_gain_experiment.py']})
            audit.write_json(source/'protocol.json',protocol);(source/'CONTINUATION_COMPLETE').write_text('OK')
            report=dict(protocol=protocol,summary={'complete':True},runs={})
            checkpoint_hashes={}
            for seed in ['2081','2082']:
                report['runs'][seed]={}
                torch.manual_seed(int(seed));model=make_gain_model(2.)
                for branch in ['keep_lr','lower_lr']:
                    folder=source/('seed'+seed)/branch;folder.mkdir(parents=True);(folder/'CASE_COMPLETE').write_text('OK')
                    saved=dict(model=copy.deepcopy(model.state_dict()),se_gain=2.,activation='tf-compatible',
                        architecture='Harper2023-experimental-se-gain',epoch=144,steps=144*53,model_seed=int(seed),
                        protocol_sha256=audit.sha(source/'protocol.json'),selection_sha256=original['protocol']['selection_sha256'])
                    torch.save(saved,folder/'best_checkpoint.pt')
                    a=audit.continuation.score(model,train['x'],torch.from_numpy(train['labels']),'cpu')
                    b=audit.continuation.score(model,val['x'],torch.from_numpy(val['labels']),'cpu')
                    history=[dict(epoch=100+i,steps=(100+i)*53,validation=b) for i in range(51)]
                    audit.write_json(folder/'history.json',history)
                    report['runs'][seed][branch]=dict(best_checkpoint_sha256=audit.sha(folder/'best_checkpoint.pt'),
                        history_sha256=audit.sha(folder/'history.json'),best_epoch=144,selected_training=a,selected_validation=b,final_validation=b)
                    checkpoint_hashes[str(folder)]=audit.sha(folder/'best_checkpoint.pt')
            audit.write_json(source/'report.json',report)
            prepared=dict(report=original,checkpoint_sha256=protocol['source_checkpoints'],x=train['x'],vx=val['x'])
            metadata={'splits':{name:{'parameters_sha256':audit.sha(data/(name+'_parameters.jsonl'))} for name in ['train','validation']}}
            args=argparse.Namespace(run=source,development_source=None,data=data,output=root/'audit',device='cpu')
            with patch.object(audit.continuation,'prepare_source',return_value=prepared),patch.object(
                audit.continuation.development,'load_metadata',return_value=metadata):audit.run(args)
            result=json.loads((args.output/'report.json').read_text())
            self.assertTrue(result['complete']);self.assertEqual(len(result['cases']),4)
            self.assertFalse(result['protocol']['training_performed']);self.assertFalse(result['protocol']['test_opened'])
            for case in result['cases'].values():
                for split in ['train','validation']:
                    self.assertTrue(case[split]['prediction_matrix_and_loss_reproduced'])
                    self.assertEqual(case[split]['overall']['count'],24)
                    self.assertTrue((args.output/case[split]['prediction_csv']).is_file())
            self.assertTrue((args.output/'ERROR_AUDIT_COMPLETE').is_file())
            for folder,expected in checkpoint_hashes.items():self.assertEqual(audit.sha(Path(folder)/'best_checkpoint.pt'),expected)
            self.assertFalse((data/'test.npz').exists())
            with patch.object(audit.continuation,'prepare_source',return_value=prepared),patch.object(
                audit.continuation.development,'load_metadata',return_value=metadata):
                with self.assertRaisesRegex(ValueError,'Output exists'):audit.run(args)


if __name__=='__main__':unittest.main()
