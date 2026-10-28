import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
import continue_se_gain as continuation
from se_gain_experiment import make_gain_model
from check_modulation_learning import state_digest
from test_se_gain_development import fixture


def optimize_epoch(model,optimizer,generator,x,y):
    model.train()
    for batch in torch.randperm(len(y),generator=generator).split(32):
        optimizer.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(x[batch]),y[batch]).backward();optimizer.step()


def saved_state(model,optimizer,generator,epoch,steps,seed=2081):
    return dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(optimizer.state_dict()),
                generator_state=generator.get_state(),epoch=epoch,steps=steps,model_seed=seed,
                architecture='Harper2023-experimental-se-gain',activation='tf-compatible',se_gain=2.,
                selection_sha256='fixture')


class ContinuationTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_restored_adam_and_rng_match_uninterrupted_training_exactly(self):
        torch.manual_seed(10);model=make_gain_model(2.)
        x=torch.randn(36,2,1,64);y=torch.arange(36)%12
        optimizer=torch.optim.Adam(model.parameters(),lr=.0001,eps=1e-8)
        generator=torch.Generator().manual_seed(2080)
        optimize_epoch(model,optimizer,generator,x,y)
        saved=saved_state(model,optimizer,generator,1,2)
        resumed,adam,rng=continuation.restore_training(saved,'cpu',.0001)
        lower,lower_adam,lower_rng=continuation.restore_training(saved,'cpu',.00001)
        self.assertEqual(state_digest(resumed.state_dict()),state_digest(lower.state_dict()))
        self.assertEqual(continuation.adam_digest(adam.state_dict(),True),continuation.adam_digest(lower_adam.state_dict(),True))
        self.assertTrue(torch.equal(rng.get_state(),lower_rng.get_state()))
        optimize_epoch(model,optimizer,generator,x,y)
        optimize_epoch(resumed,adam,rng,x,y)
        self.assertEqual(state_digest(model.state_dict()),state_digest(resumed.state_dict()))
        self.assertEqual(continuation.adam_digest(optimizer.state_dict()),continuation.adam_digest(adam.state_dict()))
        self.assertTrue(torch.equal(generator.get_state(),rng.get_state()))
        with self.assertRaisesRegex(ValueError,'epoch100'):continuation.validate_snapshot(saved)
        altered=copy.deepcopy(saved);altered['epoch']=100;altered['steps']=5300
        with self.assertRaisesRegex(ValueError,'step/moments'):continuation.validate_snapshot(altered)

    def test_baseline_mismatch_is_not_silently_accepted(self):
        reference={'confusion_matrix':[[1]],'loss':1.}
        continuation.require_baseline(reference,reference)
        with self.assertRaisesRegex(ValueError,'does not reproduce'):
            continuation.require_baseline({'confusion_matrix':[[1]],'loss':1.1},reference)
        with self.assertRaisesRegex(ValueError,'does not reproduce'):
            continuation.require_baseline({'confusion_matrix':[[0]],'loss':1.},reference)

    def test_complete_preflight_and_four_short_continuations(self):
        development=continuation.development
        rng=np.random.default_rng(11)
        ty,tc,ts=fixture('train',20)
        vy,vc,vs=fixture('val',13)
        # Full validation size1080: ninety whole groups, retain all twelve classes.
        vy,vc,vs=vy[:1080],vc[:1080],vs[:1080]
        def iq(n):return (rng.normal(size=(n,64))+1j*rng.normal(size=(n,64))).astype(np.complex64)
        train=(iq(len(ty)),ty,np.asarray(['t'+str(i) for i in range(len(ty))]),ts)
        val=(iq(len(vy)),vy,np.asarray(['v'+str(i) for i in range(len(vy))]),vs)
        registered=Path(continuation.__file__).with_name('SE_GAIN_DEVELOPMENT_PLAN_V1.json')
        plan=json.loads(registered.read_text());real_sha=continuation.sha
        # Synthetic I/O fixtures register their actual tensors; production uses the untouched source plan.
        for seed in [2081,2082]:
            torch.manual_seed(seed)
            plan['controls']['initial_sha256_by_seed'][str(seed)]=state_digest(make_gain_model(2.).state_dict())
        def hashes(path):
            if path.name=='metadata.json':return plan['controls']['metadata_sha256']
            if path.name=='train.npz':return plan['controls']['train_sha256']
            return real_sha(path)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source';source.mkdir();data=root/'data';data.mkdir()
            (source/'DEVELOPMENT_COMPLETE').write_text('OK');(data/'metadata.json').write_text('{}')
            fixture_plan=root/'fixture-plan.json';fixture_plan.write_text(json.dumps(plan))
            registered_hash=real_sha(fixture_plan)
            np.savez(data/'train.npz',condition_id=tc);np.savez(data/'validation.npz',condition_id=vc)
            indices,groups=development.select_condition_groups(ty,tc,ts,20,2080)
            np.savez(source/'training_selection.npz',source_id=train[2][indices],condition_id=tc[indices],y=ty[indices],snr_db=ts[indices])
            protocol=dict(default_design=True,epochs=100,train_examples=1680,validation_examples=1080,
                plan=plan,plan_sha256=registered_hash,classes=development.CLASSES,
                dataset_metadata_sha256=plan['controls']['metadata_sha256'],train_sha256=plan['controls']['train_sha256'],
                validation_sha256=real_sha(data/'validation.npz'),selection_sha256=real_sha(source/'training_selection.npz'),
                selected_condition_ids=groups,sources={name:real_sha(Path(continuation.__file__).with_name(name)) for name in
                    ['train_se_gain_development.py','modulation_reference.py','se_gain_experiment.py']})
            continuation.write_json(source/'protocol.json',protocol)
            runs={};inputs=development.as_tensor(train[0][indices]);targets=torch.from_numpy(ty[indices])
            for seed in [2081,2082]:
                torch.manual_seed(seed);model=make_gain_model(2.)
                parent=source/('seed'+str(seed));folder=parent/'gain2';folder.mkdir(parents=True)
                initial=copy.deepcopy(model.state_dict());torch.save(initial,parent/'initial_state.pt')
                optimizer=torch.optim.Adam(model.parameters(),lr=.0001,eps=1e-8)
                batch_rng=torch.Generator().manual_seed(2080)
                # Short synthetic warmup supplies real moments; emulate source step metadata for I/O test.
                optimizer.zero_grad(set_to_none=True)
                nn.functional.cross_entropy(model(inputs[:32]),targets[:32]).backward();optimizer.step()
                for state in optimizer.state.values():state['step'].fill_(5300)
                for _ in range(100):torch.randperm(1680,generator=batch_rng)
                saved=saved_state(model,optimizer,batch_rng,100,5300,seed)
                saved['selection_sha256']=protocol['selection_sha256'];saved['protocol_sha256']=real_sha(source/'protocol.json')
                torch.save(saved,folder/'last_checkpoint.pt')
                baseline=continuation.score(model,development.as_tensor(val[0]),torch.from_numpy(vy),'cpu')
                history=[dict(epoch=i,steps=53*i,validation=baseline) for i in range(1,101)]
                continuation.write_json(folder/'history.json',history)
                runs[str(seed)]={'gain2':dict(epochs=100,steps=5300,se_gain=2.,initial_sha256=state_digest(initial),
                    final_validation=baseline,last_checkpoint_sha256=real_sha(folder/'last_checkpoint.pt'),history_sha256=real_sha(folder/'history.json'))}
            continuation.write_json(source/'report.json',dict(protocol=protocol,runs=runs))
            checkpoints={seed:real_sha(source/('seed'+str(seed))/'gain2/last_checkpoint.pt') for seed in [2081,2082]}
            args=argparse.Namespace(source=source,data=data,output=root/'output',device='cpu',epochs=1)
            def split(folder,name,metadata,seen):
                self.assertIn(name,['train','validation']);return train if name=='train' else val
            with patch.object(development,'load_metadata',return_value={}),patch.object(development,'load_split',side_effect=split) as loader,patch.object(continuation,'sha',side_effect=hashes),patch.object(continuation,'registered_source_plan',return_value=(plan,registered_hash)):
                continuation.run(args)
            self.assertEqual([c.args[1] for c in loader.call_args_list],['train','validation'])
            report=json.loads((args.output/'report.json').read_text())
            self.assertTrue(report['summary']['complete']);self.assertFalse(report['protocol']['default_design'])
            self.assertFalse(report['protocol']['test_opened'])
            for seed,cases in report['runs'].items():
                self.assertEqual(cases['keep_lr']['starting_checks'],cases['lower_lr']['starting_checks'])
                for name,result in cases.items():
                    self.assertEqual(result['additional_steps'],53);self.assertEqual(result['end_epoch'],101)
                    self.assertTrue(result['starting_checks']['source_validation_reproduced'])
                    self.assertEqual(real_sha(source/('seed'+seed)/'gain2/last_checkpoint.pt'),checkpoints[int(seed)])
            self.assertTrue((args.output/'CONTINUATION_COMPLETE').is_file())
            with self.assertRaisesRegex(ValueError,'Output exists'):continuation.run(args)
            # Tampering must be caught before reading data or starting updates.
            with (source/'seed2081/gain2/last_checkpoint.pt').open('ab') as f:f.write(b'tamper')
            with patch.object(development,'load_metadata',return_value={}),patch.object(development,'load_split',side_effect=split),patch.object(continuation,'sha',side_effect=hashes),patch.object(continuation,'registered_source_plan',return_value=(plan,registered_hash)):
                with self.assertRaisesRegex(ValueError,'checkpoint/history hash'):continuation.prepare_source(source,data)


if __name__=='__main__':unittest.main()
