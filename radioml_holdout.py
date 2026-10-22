"""Gemeinsame Pruefungen fuer einen zusaetzlichen RadioML-I/Q-Testsatz."""
import hashlib
import json
from pathlib import Path
import numpy as np
from train_radioml_iq import load_dataset


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def reference_ids(folder):
    meta,_=load_dataset(folder)
    arrays=[]
    for name in ('train','validation','test'):
        with np.load(Path(folder)/(name+'.npz'),allow_pickle=False) as data:
            arrays.append(data['source_index'])
    return meta,np.concatenate(arrays)


def load_holdout(folder,reference,model):
    folder=Path(folder)
    if not (folder/'IMPORT_COMPLETE').is_file():
        raise ValueError('Holdout-Import nicht abgeschlossen')
    meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    ref,excluded=reference_ids(reference)
    if (meta.get('purpose')!='holdout-only' or meta.get('representation')!='iq-rms-v1'
            or meta.get('input_shape')!=[1,1024,2] or meta.get('target_rms')!=0.5
            or meta.get('labels')!=ref['labels']
            or meta.get('selected_y_columns')!=ref['selected_y_columns']
            or meta.get('frozen_model_sha256')!=file_hash(model)):
        raise ValueError('Holdout-Metadaten/Modell stimmen nicht mit Referenz ueberein')
    with np.load(folder/'test.npz',allow_pickle=False) as data:
        x,y,snr,ids=(data[k] for k in ('x','y','snr_db','source_index'))
    if (x.shape!=(len(y),1,1024,2) or len(y)==0 or y.shape!=(len(x),)
            or snr.shape!=y.shape or ids.shape!=y.shape
            or not np.issubdtype(y.dtype,np.integer) or not np.issubdtype(ids.dtype,np.integer)
            or not np.isfinite(x).all() or not np.isfinite(snr).all()
            or set(y.tolist())!=set(range(5)) or np.any(ids<0)
            or len(np.unique(ids))!=len(ids) or np.intersect1d(ids,excluded).size):
        raise ValueError('Ungueltiger oder ueberlappender Holdout-Testsatz')
    levels=meta['snr_values']; cap=meta['per_class_per_snr']
    if (len(x)!=5*len(levels)*cap or set(snr.tolist())!=set(levels)
            or any(np.sum((y==c)&(snr==s))!=cap for c in range(5) for s in levels)):
        raise ValueError('Holdout-Stratifizierung stimmt nicht')
    return meta,x,y,snr,ids
