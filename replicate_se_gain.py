"""Run the two preregistered paired initialization-seed replications."""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
from build_modulation_dataset import sha
import diagnose_training_se_gain as engine


def summarize(runs):
    comparisons={}
    for seed,report in runs.items():
        a=report['cases']['gain1']['final']['training']
        b=report['cases']['gain2']['final']['training']
        comparisons[seed]=dict(accuracy_gain2_minus_gain1=b['accuracy']-a['accuracy'],
                               loss_gain2_minus_gain1=b['loss']-a['loss'],
                               both_primary_metrics_improve=b['accuracy']>a['accuracy'] and b['loss']<a['loss'])
    complete=set(runs)=={'2071','2072'}
    return dict(comparisons=comparisons,complete=complete,
                directional_replication=(all(v['both_primary_metrics_improve'] for v in comparisons.values())
                                         if complete else None),
                interpretation='two initialization seeds on the same training examples; not statistical significance or generalization')


def run(args):
    if args.epochs<1 or args.per_class<1: raise ValueError('Positive training parameters required')
    if args.output is None:
        args.output=Path('models')/('se-gain-replication-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    if args.output.exists(): raise ValueError('Ausgabe existiert; neuen --output waehlen')
    plan_path=Path(__file__).with_name('SE_GAIN_REPLICATION_PLAN_V1.json')
    plan=json.loads(plan_path.read_text())
    args.output.mkdir(parents=True)
    protocol=dict(plan=plan,plan_sha256=sha(plan_path),script_sha256=sha(Path(__file__)),
                  training_engine_sha256=sha(Path(engine.__file__)),epochs=args.epochs,per_class=args.per_class,
                  default_design=(args.epochs==400 and args.per_class==10),
                  historical_seed2070_excluded=True)
    engine.write_json(args.output/'protocol.json',protocol)
    runs={}; selection_hash=None
    for seed in plan['model_seeds']:
        print('\nModell-Seed:',seed,'| SE-Faktor 1 und 2, gleiche 120 Trainingsbeispiele',flush=True)
        child=copy.copy(args); child.output=args.output/('seed'+str(seed)); child.activation='tf-compatible'
        engine.run(child,model_seed=seed,plan_name=plan_path.name)
        report=json.loads((child.output/'report.json').read_text())
        digest=report['protocol']['selection_sha256']
        if selection_hash is not None and digest!=selection_hash:
            raise ValueError('Selection differs between seeds')
        selection_hash=digest
        if report['cases']['gain1']['initial_sha256']!=report['cases']['gain2']['initial_sha256']:
            raise ValueError('Unpaired initial weights')
        runs[str(seed)]=report
        engine.write_json(args.output/'report.json',dict(protocol=protocol,runs=runs,summary=summarize(runs)))
    (args.output/'SE_GAIN_REPLICATION_COMPLETE').write_text('OK\n')
    print('Replikation abgeschlossen:',args.output.resolve())
    print('Bitte report.json aus diesem Hauptordner auswerten; keine Testgenauigkeit.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data/modulation12-v1'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--epochs',type=int,default=400)
    parser.add_argument('--per-class',type=int,default=10)
    try: run(parser.parse_args())
    except (OSError,ValueError,RuntimeError,KeyError,AssertionError) as error:
        parser.exit(1,'SE-Replikation fehlgeschlagen: '+str(error)+'\n')
