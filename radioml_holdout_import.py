"""Zusaetzlichen RadioML-Test importieren; bisherige Train/Val/Test-Zeilen ausschliessen."""
import argparse
import json
from pathlib import Path
import numpy as np
from radioml_holdout import reference_ids,file_hash
from radioml_iq_import import iq_features


def choose_new_rows(handle,columns,levels,excluded,cap,seed,scan_func=None):
    if scan_func is None:
        from radioml_import import scan as scan_func
    rng=np.random.default_rng(seed)
    pools={}
    for start,classes,snrs in scan_func(handle):
        indices=np.arange(start,start+len(classes),dtype=np.int64)
        eligible=~np.isin(indices,excluded)
        for label,cid in enumerate(columns):
            for snr in levels:
                found=indices[eligible&(classes==cid)&(snrs==snr)]
                if not len(found): continue
                keys=rng.random(len(found))
                key=(label,float(snr))
                old=pools.get(key)
                if old is not None:
                    found=np.concatenate((old[0],found)); keys=np.concatenate((old[1],keys))
                keep=np.argsort(keys)[:cap]
                pools[key]=(found[keep],keys[keep])
    rows=[]
    for label in range(len(columns)):
        for snr in levels:
            selected=pools.get((label,float(snr)))
            if selected is None or len(selected[0])!=cap:
                raise ValueError('Nicht genug unbenutzte Zeilen je Klasse/SNR')
            rows.extend((int(i),label,float(snr)) for i in selected[0])
    return [rows[i] for i in rng.permutation(len(rows))]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--reference',type=Path,default=Path('data/radioml2018-iq-v1'))
    parser.add_argument('--model',type=Path,default=Path('models/radioml2018-iq-v1/compiled/radioml_int8_edgetpu.tflite'))
    parser.add_argument('--output',type=Path,default=Path('data/radioml2018-iq-holdout-v1'))
    parser.add_argument('--per-bucket',type=int,default=200)
    parser.add_argument('--seed',type=int,default=2032)
    args=parser.parse_args()
    try:
        if args.per_bucket<1 or args.output.exists():
            raise ValueError('Positive Bucketgroesse und neuer Ausgabeordner erforderlich')
        meta,excluded=reference_ids(args.reference)
        if args.source.stat().st_size!=meta['source_bytes']:
            raise ValueError('HDF5-Dateigroesse weicht von Referenz ab')
        frozen=file_hash(args.model)
        import h5py
        from radioml_import import validate_file
        print('Ausgeschlossene bisherige Quellzeilen:',len(excluded),flush=True)
        with h5py.File(args.source,'r') as handle:
            validate_file(handle)
            rows=choose_new_rows(handle,meta['selected_y_columns'],meta['snr_values'],excluded,args.per_bucket,args.seed)
            print('Neue Testzeilen ausgewaehlt:',len(rows),flush=True)
            features=[]
            for n,(index,_,_) in enumerate(rows,1):
                features.append(iq_features(handle['X'][index]))
                if n%1000==0: print(n,'/',len(rows),'importiert',flush=True)
        x=np.stack(features)
        ids=np.array([r[0] for r in rows],dtype=np.int64)
        y=np.array([r[1] for r in rows],dtype=np.int64)
        snr=np.array([r[2] for r in rows],dtype=np.float32)
        if np.intersect1d(ids,excluded).size: raise ValueError('Interner Ausschlussfehler')
        meta.update(purpose='holdout-only',seed=args.seed,per_class_per_snr=args.per_bucket,
            excluded_source_rows=len(excluded),reference_directory=str(args.reference.resolve()),
            frozen_model_sha256=frozen,source_file=str(args.source.resolve()),
            split='test only; all reference train/validation/test source rows excluded',
            limitations='Disjoint source rows do not prove independent recordings; synthetic data')
        args.output.mkdir(parents=True)
        np.savez_compressed(args.output/'test.npz',x=x,y=y,snr_db=snr,source_index=ids)
        (args.output/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        (args.output/'IMPORT_COMPLETE').write_text('OK\n')
        print('Holdout fertig:',args.output.resolve(),x.shape)
        print('Kein Training, keine Modellanpassung. Erstes Ergebnis als Holdout-Ergebnis festhalten.')
    except (OSError,ValueError,KeyError,ImportError) as error:
        parser.exit(1,'Holdout-Import fehlgeschlagen: '+str(error)+'\n')


if __name__=='__main__': main()
