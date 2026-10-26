"""Paired full-batch learning-rate diagnostic; training split only."""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from build_modulation_dataset import sha
from check_modulation_learning import select_examples, state_digest
from modulation_reference import Harper2023
from train_modulation12 import as_tensor, load_metadata, load_split, metrics
from torchsig_probe import CLASSES


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def score(model, x, y, device):
    model.eval()
    losses, predictions = [], []
    with torch.inference_mode():
        for start in range(0, len(y), 32):
            logits = model(x[start:start+32].to(device))
            target = y[start:start+32].to(device)
            losses.extend(nn.functional.cross_entropy(logits, target, reduction='none').cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    labels = y.numpy()
    result = metrics(labels, np.asarray(predictions))
    result['loss'] = float(np.mean(losses))
    result['loss_by_class'] = {name: float(np.asarray(losses)[labels == cid].mean())
                               for cid, name in enumerate(CLASSES)}
    result['predicted_counts'] = np.bincount(predictions, minlength=len(CLASSES)).tolist()
    return result


def run(args):
    if args.epochs < 1 or args.per_class < 1:
        raise ValueError('Positive epochs/per-class required')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA nicht verfuegbar')
    if args.output is None:
        args.output = Path('models') / ('training-lr-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    if args.output.exists():
        raise ValueError('Ausgabe existiert; neuen --output waehlen')
    meta = load_metadata(args.data)
    iq, labels, ids, snr = load_split(args.data, 'train', meta, set())
    iq, labels, ids, snr = select_examples(iq, labels, ids, snr, args.per_class, 2070)
    random.seed(2070); np.random.seed(2070); torch.manual_seed(2070)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2070)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = Harper2023(tensorflow_selu=args.activation == 'tf-compatible')
    initial = copy.deepcopy(model.state_dict())
    expected = json.loads(Path(__file__).with_name('TRAINING_LR_PLAN_V1.json').read_text())
    if sha(args.data/'metadata.json') != expected['controls']['metadata_sha256']:
        raise ValueError('Datensatz stimmt nicht mit dem festgelegten Vergleich ueberein')
    if sha(args.data/'train.npz') != expected['controls']['train_sha256']:
        raise ValueError('Trainingsdatei stimmt nicht mit dem Vergleich ueberein')
    if state_digest(initial) != expected['controls']['initial_sha256']:
        raise ValueError('Anfangsgewichte stimmen nicht mit dem Vergleich ueberein')
    args.output.mkdir(parents=True)
    torch.save(initial, args.output/'initial_state.pt')
    np.savez_compressed(args.output/'selection.npz', iq=iq, y=labels, source_id=ids, snr_db=snr)
    plan_path = Path(__file__).with_name('TRAINING_LR_PLAN_V1.json')
    protocol = dict(plan=json.loads(plan_path.read_text()), plan_sha256=sha(plan_path),
        activation=args.activation, epochs=args.epochs, per_class=args.per_class,
        default_design=(args.epochs == 400 and args.per_class == 10 and args.activation == 'tf-compatible'),
        torch_version=torch.__version__, numpy_version=np.__version__, cuda=torch.version.cuda,
        device=args.device, device_name=torch.cuda.get_device_name(0) if args.device == 'cuda' else 'CPU',
        metadata_sha256=sha(args.data/'metadata.json'), train_sha256=sha(args.data/'train.npz'),
        selection_sha256=sha(args.output/'selection.npz'), initial_sha256=state_digest(initial),
        sources={name: sha(Path(__file__).with_name(name)) for name in
                 ['diagnose_training_lr.py', 'modulation_reference.py',
                  'check_modulation_learning.py', 'train_modulation12.py']},
        qualification='training-only diagnostic; not accuracy or historical reproduction')
    write_json(args.output/'protocol.json', protocol)
    x, y = as_tensor(iq), torch.from_numpy(labels)
    results = {}
    for case, learning_rate in [('lr0.001', .001), ('lr0.0001', .0001)]:
        batch_size = len(y)
        folder = args.output/case; folder.mkdir()
        model.load_state_dict(initial); model.to(args.device)
        assert state_digest(model.state_dict()) == protocol['initial_sha256']
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-8)
        generator = torch.Generator().manual_seed(2070)
        history = [dict(epoch=0, steps=0, examples_processed=0, training=score(model,x,y,args.device))]
        steps = 0
        for epoch in range(1,args.epochs+1):
            model.train()
            order = torch.randperm(len(y), generator=generator)
            batch_records = []
            for indices in order.split(batch_size):
                optimizer.zero_grad(set_to_none=True)
                logits = model(x[indices].to(args.device))
                loss = nn.functional.cross_entropy(logits,y[indices].to(args.device))
                if not torch.isfinite(loss): raise ValueError('Nichtendlicher Verlust')
                loss.backward()
                parameters = {}
                before = {}
                # Detailed norms for first batch each epoch; class counts for every batch.
                if not batch_records:
                    for name,p in model.named_parameters():
                        if p.grad is None or not torch.isfinite(p.grad).all():
                            raise ValueError('Fehlender/nichtendlicher Gradient: '+name)
                        before[name] = p.detach().clone()
                        parameters[name] = dict(weight_l2=float(p.detach().norm()), gradient_l2=float(p.grad.norm()))
                else:
                    for p in model.parameters():
                        if p.grad is None or not torch.isfinite(p.grad).all():
                            raise ValueError('Fehlender/nichtendlicher Gradient')
                optimizer.step(); steps += 1
                for name,p in model.named_parameters():
                    if name in before:
                        parameters[name]['update_l2'] = float((p.detach()-before[name]).norm())
                batch_records.append(dict(class_counts=torch.bincount(y[indices],minlength=12).tolist(),
                                          loss=float(loss.detach()), parameters=parameters))
            result = score(model,x,y,args.device)
            history.append(dict(epoch=epoch,steps=steps,examples_processed=epoch*len(y),
                                training=result,batches=batch_records))
            write_json(folder/'history.json',history)
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                            generator_state=generator.get_state()),folder/'last_checkpoint.pt.tmp')
            (folder/'last_checkpoint.pt.tmp').replace(folder/'last_checkpoint.pt')
            print(f'{case} Epoche {epoch}/{args.epochs}, Schritte {steps}: '
                  f'Verlust {result["loss"]:.4f}, Trainings-Wiedererkennung {result["accuracy"]:.1%}',flush=True)
        best_loss = min(history, key=lambda row: row['training']['loss'])
        best_accuracy = max(history, key=lambda row: row['training']['accuracy'])
        jumps = [(history[i]['training']['loss'] - history[i-1]['training']['loss'], i)
                 for i in range(1, len(history))]
        largest_jump, jump_epoch = max(jumps)
        trajectory = dict(minimum_loss=best_loss['training']['loss'], minimum_loss_epoch=best_loss['epoch'],
            maximum_accuracy=best_accuracy['training']['accuracy'], maximum_accuracy_epoch=best_accuracy['epoch'],
            largest_loss_increase=max(0., largest_jump),
            largest_loss_increase_epoch=jump_epoch if largest_jump > 0 else None,
            final_loss_minus_minimum=history[-1]['training']['loss']-best_loss['training']['loss'],
            maximum_accuracy_minus_final=best_accuracy['training']['accuracy']-history[-1]['training']['accuracy'])
        results[case] = dict(learning_rate=learning_rate, trajectory=trajectory,
                             initial_sha256=protocol['initial_sha256'], final=history[-1],final_sha256=state_digest(model.state_dict()),
                             checkpoint_sha256=sha(folder/'last_checkpoint.pt'),
                             history_sha256=sha(folder/'history.json'))
        write_json(args.output/'report.json',dict(protocol=protocol,cases=results))
        model.cpu()
    (args.output/'LR_DIAGNOSTIC_COMPLETE').write_text('OK\n')
    print('Ergebnisse:',args.output.resolve())
    print('Nur Trainings-Wiedererkennung; keine Testgenauigkeit. Kein automatisches Resume.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data/modulation12-v1'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--epochs',type=int,default=400)
    parser.add_argument('--per-class',type=int,default=10)
    parser.add_argument('--activation',choices=['tf-compatible','historical'],default='tf-compatible')
    try: run(parser.parse_args())
    except (OSError,ValueError,RuntimeError,KeyError,AssertionError) as error:
        parser.exit(1,'Diagnose fehlgeschlagen: '+str(error)+'\n')
