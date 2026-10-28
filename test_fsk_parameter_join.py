import csv
import json
from pathlib import Path
import tempfile
import unittest
import join_fsk_parameter_errors as join


class JoinTests(unittest.TestCase):
    def test_bins_preserve_gmsk_boundary(self):
        edges=json.loads(join.PLAN_PATH.read_text())['h_edges']
        self.assertEqual(join.bucket(.499,edges),'0.45:0.5')
        self.assertEqual(join.bucket(.5,edges),'0.5:0.6')
        self.assertEqual(join.bucket(None,edges),'not_applicable')
        with self.assertRaises(ValueError):join.bucket(float('nan'),edges)

    def fixture(self,root):
        data=root/'data';audit=root/'audit';replay=root/'replay'
        for path,marker in ((data,'DATASET_COMPLETE'),(audit,'ERROR_AUDIT_COMPLETE'),(replay,'REPLAY_COMPLETE')):
            path.mkdir();(path/marker).touch()
        meta=dict(classes=join.CLASSES,splits={})
        r=dict(complete=True,parameter_interpretation_allowed=True,splits={})
        a=dict(complete=True,protocol=dict(parameter_jsonl_sha256={}),cases={k:{} for k in join.CASES})
        for split in ('train','validation'):
            originals=[];reconstructed=[];predictions=[];matrix=[[0]*12 for _ in range(12)]
            for i,label in enumerate(join.CLASSES):
                sid=f'{split}:{i}';h=.5 if label=='2gmsk' else (.49 if label=='2gfsk' else 1.)
                params=dict(h=h,bt=None if label=='2fsk' else .3,gaussian_span=None if label=='2fsk' else 3.)
                originals.append(dict(source_id=sid,label=label,condition_id=split,iq_sha256=sid,parameters=dict(snr_db=20.)))
                reconstructed.append(dict(source_id=sid,label=label,exact=True,stored_sha256=sid,replay_sha256=sid,
                    instrumented_sha256=sid if label in join.FSK else None,actual_fsk_parameters=[params] if label in join.FSK else []))
                pred='fm' if label=='2gfsk' else label
                predictions.append(dict(source_id=sid,true_label=label,predicted_label=pred,condition_id=split,
                    correct=str(label==pred),snr_db=20.))
                matrix[i][join.CLASSES.index(pred)]+=1
            param_path=data/(split+'_parameters.jsonl')
            param_path.write_text(''.join(json.dumps(x)+'\n' for x in originals))
            rp=replay/(split+'_replay.jsonl');rp.write_text(''.join(json.dumps(x)+'\n' for x in reconstructed))
            meta['splits'][split]=dict(parameters_sha256=join.sha(param_path))
            a['protocol']['parameter_jsonl_sha256'][split]=join.sha(param_path)
            r['splits'][split]=dict(replay_jsonl_sha256=join.sha(rp),examples=12,exact=12,mismatches=0)
            for case in join.CASES:
                path=audit/(case+'-'+split+'.csv')
                with path.open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
                a['cases'][case][split]=dict(prediction_csv=path.name,prediction_csv_sha256=join.sha(path),
                    prediction_matrix_and_loss_reproduced=True,overall=dict(count=12,confusion_matrix=matrix))
        (data/'metadata.json').write_text(json.dumps(meta));r['metadata_sha256']=join.sha(data/'metadata.json')
        (replay/'report.json').write_text(json.dumps(r));(audit/'report.json').write_text(json.dumps(a))
        return data,audit,replay

    def test_full_join_all_four_models_without_test_or_torch(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);data,audit,replay=self.fixture(root)
            report=join.run(data,audit,replay,root/'out')
            self.assertTrue(report['complete']);self.assertEqual(set(report['cases']),join.CASES)
            for case in report['cases'].values():
                cells=case['validation']['tables']['snr_ge15']['by_h']
                gfsk=next(x for x in cells if x['true_label']=='2gfsk')
                self.assertEqual(gfsk['predicted_counts'],{'fm':1})
                self.assertTrue(gfsk['sparse'])
            self.assertTrue((root/'out/FSK_PARAMETER_JOIN_COMPLETE').exists())

    def test_tampered_csv_retains_failure(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);data,audit,replay=self.fixture(root)
            path=next(audit.glob('*.csv'));path.write_text(path.read_text()+'tamper')
            with self.assertRaises(ValueError):join.run(data,audit,replay,root/'out')
            self.assertFalse(json.loads((root/'out/report.json').read_text())['complete'])
            self.assertFalse((root/'out/FSK_PARAMETER_JOIN_COMPLETE').exists())

    def test_label_mismatch_rejected_after_updated_hash(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);data,audit,replay=self.fixture(root)
            path=replay/'validation_replay.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
            rows[0]['label']='fm';path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
            report=json.loads((replay/'report.json').read_text());report['splits']['validation']['replay_jsonl_sha256']=join.sha(path)
            (replay/'report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'Label/condition'):join.run(data,audit,replay,root/'out')


if __name__=='__main__':unittest.main()
