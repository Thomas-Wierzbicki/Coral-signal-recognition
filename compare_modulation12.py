"""Fixed 12-class architecture comparison; development results, no test-based selection."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from build_modulation_dataset import sha
from train_modulation12 import load_metadata, train

ARCHITECTURES=('legacy','harper2023')


def summarize(runs):
    summary={}
    for architecture in ARCHITECTURES:
        selected=[row for row in runs if row['architecture']==architecture]
        values=np.array([row['test_accuracy'] for row in selected])
        summary[architecture]=dict(runs=len(selected),mean_accuracy=float(values.mean()),
            sample_std_accuracy=float(values.std(ddof=1)) if len(values)>1 else None,
            mean_macro_f1=float(np.mean([row['macro_f1'] for row in selected])),
            mean_recall={label:float(np.mean([row['recall'][label] for row in selected]))
                for label in selected[0]['recall']},
            mean_accuracy_by_snr={snr:float(np.mean([row['by_snr'][snr]['accuracy'] for row in selected]))
                for snr in selected[0]['by_snr']},
            mean_frequency_accuracy={str(hz):float(np.mean([row['frequency_tests'][str(hz)]['accuracy']
                for row in selected])) for hz in (-1600,1600)})
    return summary


def compare(data,output,seeds=(2051,2052,2053),epochs=40,batch_size=32,patience=8,device='cuda'):
    data=Path(data); output=Path(output)
    if output.exists(): raise ValueError('Ausgabe existiert; einen neuen --output waehlen')
    if not seeds or len(set(seeds))!=len(seeds) or any(s<0 for s in seeds):
        raise ValueError('Verschiedene nichtnegative Seeds erforderlich')
    if min(epochs,batch_size,patience)<1: raise ValueError('Positive Trainingsparameter erforderlich')
    import torch
    if device=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA nicht verfuegbar')
    metadata=load_metadata(data)
    for name,spec in metadata['splits'].items():
        if sha(data/(name+'.npz'))!=spec['sha256']: raise ValueError('Datensatz-Hash falsch: '+name)
    output.mkdir(parents=True)
    protocol=dict(architectures=list(ARCHITECTURES),seeds=list(seeds),epochs=epochs,
        batch_size=batch_size,patience=patience,device=device,dataset_metadata_sha256=sha(data/'metadata.json'),
        selection='minimum validation loss within each run; no choice based on test results',
        training='same Adam lr=0.001, no scheduler or extra augmentation, same batches per seed',
        limitation='Architecture package comparison on existing synthetic development split. Not isolated proof about pooling, representation, or real-world performance.',
        source_sha256={p:sha(Path(__file__).parent/p) for p in
            ('train_modulation12.py','modulation_reference.py','compare_modulation12.py')})
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    environment=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout
    (output/'environment-freeze.txt').write_text(environment,encoding='utf-8')
    runs=[]
    for seed in seeds:
        for architecture in ARCHITECTURES:
            print('\nVergleich:',architecture,'Seed',seed,flush=True)
            report=train(data,output/f'{architecture}-seed{seed}',epochs,batch_size,seed,device,patience,architecture)
            runs.append(dict(architecture=architecture,seed=seed,best_epoch=report['best_epoch'],
                test_accuracy=report['test']['accuracy'],macro_f1=report['test']['macro_f1'],
                recall=report['test']['recall'],by_snr=report['by_snr'],frequency_tests=report['frequency_tests']))
            (output/'runs.json').write_text(json.dumps(runs,indent=2),encoding='utf-8')
    summary=summarize(runs)
    (output/'comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (output/'COMPARISON_COMPLETE').write_text('OK\n')
    for architecture,values in summary.items():
        print('{}: Mittel {:.1%}; Streuung {}; Macro-F1 {:.3f}'.format(architecture,values['mean_accuracy'],
            '{:.1%}'.format(values['sample_std_accuracy']) if values['sample_std_accuracy'] is not None else 'nur ein Lauf',values['mean_macro_f1']))
        for label,recall in values['mean_recall'].items(): print('  {}: {:.1%}'.format(label,recall))
        for hz,accuracy in values['mean_frequency_accuracy'].items(): print('  Verschiebung {} Hz: {:.1%}'.format(hz,accuracy))
    print('Vergleich fertig:',output.resolve())
    print('Alle zwoelf Klassen. Entwicklungsvergleich, keine unabhaengige Abschlussbewertung.')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/modulation12-v1'))
    p.add_argument('--output',type=Path,default=Path('models/modulation12-reference-v1'))
    p.add_argument('--seeds',type=int,nargs='+',default=[2051,2052,2053])
    p.add_argument('--epochs',type=int,default=40); p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--patience',type=int,default=8); p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=p.parse_args()
    try: compare(args.data,args.output,args.seeds,args.epochs,args.batch_size,args.patience,args.device)
    except (OSError,ValueError,KeyError,RuntimeError,AssertionError,subprocess.CalledProcessError) as error:
        p.exit(1,'Vergleich fehlgeschlagen: '+str(error)+'\n')
