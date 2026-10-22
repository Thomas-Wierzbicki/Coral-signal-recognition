"""RadioML 2018.01A pruefen und einen begrenzten, stratifizierten Datensatz importieren."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from signal_preprocessing import prepare_iq

DEFAULT_CLASSES = ("AM-DSB-WC","FM","BPSK","QPSK","16QAM")


def load_classes(path):
    text=Path(path).read_text(encoding="utf-8-sig").strip()
    if text.startswith("["):
        try:
            labels=json.loads(text)
        except json.JSONDecodeError:
            labels=ast.literal_eval(text)
    else:
        labels=[line.strip().strip(",").strip('"').strip("'")
                for line in text.splitlines() if line.strip()]
    if (not isinstance(labels,list) or len(labels)!=24
            or not all(isinstance(x,str) and x.strip() for x in labels)
            or len(set(labels))!=24):
        raise ValueError("Klassenliste muss 24 eindeutige Namen in Y-Spaltenreihenfolge enthalten")
    return labels


def validate_file(handle):
    if not all(k in handle for k in ("X","Y","Z")):
        raise ValueError("HDF5 muss X, Y und Z enthalten")
    x,y,z=(handle[k] for k in ("X","Y","Z"))
    if (len(x.shape)!=3 or x.shape[1:]!=(1024,2) or y.shape!=(len(x),24)
            or z.shape not in ((len(x),),(len(x),1)) or len(x)==0):
        raise ValueError("Erwartet X=(N,1024,2), Y=(N,24), Z=(N,1) oder (N,)")
    if any(not np.issubdtype(d.dtype,np.number) or np.issubdtype(d.dtype,np.complexfloating)
           for d in (x,y,z)):
        raise ValueError("Numerische reelle X/Y/Z erwartet")
    return len(x)


def scan(handle, chunk_size=16384):
    count=validate_file(handle)
    for start in range(0,count,chunk_size):
        y=np.asarray(handle["Y"][start:start+chunk_size])
        z=np.asarray(handle["Z"][start:start+chunk_size]).reshape(-1)
        if (not np.isfinite(y).all() or not np.isfinite(z).all()
                or not np.all((y==0)|(y==1)) or not np.all(y.sum(axis=1)==1)):
            raise ValueError("Ungueltige One-Hot-Labels oder SNR ab Zeile "+str(start))
        yield start,np.argmax(y,axis=1),z


def inspect(handle, labels=None):
    counts=Counter()
    for _,classes,snrs in scan(handle):
        counts.update((int(c),float(s)) for c,s in zip(classes,snrs))
    print("HDF5-Struktur:")
    for name in ("X","Y","Z"):
        print("  {}: {} {}".format(name,handle[name].shape,handle[name].dtype))
    print("SNR-Stufen:",sorted({s for _,s in counts}))
    print("Klassenbelegung:")
    for index in range(24):
        name=labels[index] if labels else "Name noch nicht zugeordnet"
        print("  {:2d} {:20s} {}".format(index,name,sum(v for (c,_),v in counts.items() if c==index)))
    print("Labels/SNR komplett gelesen; IQ-Inhalte werden erst beim Import der Auswahl geprueft.")
    if labels is None:
        print("Klassenliste fehlt: Zuordnung der 24 Y-Spalten noch nicht bestaetigt.")


def choose_rows(handle, labels, selected, low, high, cap, seed):
    if len(set(selected))!=len(selected) or any(c not in labels for c in selected):
        raise ValueError("Auswahl enthaelt doppelte oder nicht vorhandene Klassennamen")
    ids=[labels.index(name) for name in selected]
    rng=np.random.default_rng(seed)
    pools={}
    available=Counter()
    for start,classes,snrs in scan(handle):
        mask=np.isin(classes,ids)&(snrs>=low)&(snrs<=high)
        for cid in ids:
            for snr in np.unique(snrs[mask&(classes==cid)]):
                positions=np.flatnonzero(mask&(classes==cid)&(snrs==snr))
                key=(cid,float(snr))
                available[key]+=len(positions)
                indices=positions.astype(np.int64)+start
                keys=rng.random(len(indices))
                old=pools.get(key)
                if old is not None:
                    indices=np.concatenate((old[0],indices))
                    keys=np.concatenate((old[1],keys))
                keep=np.argsort(keys)[:cap]
                pools[key]=(indices[keep],keys[keep])
    snr_values=sorted({s for _,s in pools})
    if not snr_values:
        raise ValueError("Keine passenden SNR-Stufen gefunden")
    for cid in ids:
        for snr in snr_values:
            if available[(cid,snr)]<cap:
                raise ValueError("Zu wenige Beispiele fuer {} bei {} dB: {} statt {}".format(
                    labels[cid],snr,available[(cid,snr)],cap))
    return pools,ids,snr_values


def radio_features(iq_pairs):
    if iq_pairs.shape!=(1024,2) or not np.isfinite(iq_pairs).all():
        raise ValueError("IQ-Beispiel muss 1024 endliche I/Q-Paare enthalten")
    iq=prepare_iq(iq_pairs[:,0]+1j*iq_pairs[:,1],"rms-v1")
    # 128-Punkt-FFT, Hop 32: 29 Fenster, alle innerhalb desselben Originalbeispiels.
    blocks=np.stack([iq[start:start+128] for start in range(0,897,32)])
    window=np.hanning(128)
    fft=np.fft.fftshift(np.fft.fft(blocks*window,axis=1),axes=1)/window.sum()
    db=10*np.log10(np.maximum(np.abs(fft)**2,1e-12))
    return np.clip((db+80)/80,0,1).astype(np.float32)[...,None]


def prepare(handle, source, class_map, selected, low, high, cap, seed, output):
    if output.exists():
        raise ValueError("Ausgabeordner existiert bereits; neuen Ordner waehlen")
    labels=load_classes(class_map)
    pools,ids,snrs=choose_rows(handle,labels,selected,low,high,cap,seed)
    rng=np.random.default_rng(seed+1)
    split_rows={"train":[],"validation":[],"test":[]}
    for cid in ids:
        for snr in snrs:
            indices=rng.permutation(pools[(cid,snr)][0])
            n_train=int(0.7*len(indices))
            n_val=int(0.15*len(indices))
            groups=(indices[:n_train],indices[n_train:n_train+n_val],indices[n_train+n_val:])
            for split,group in zip(split_rows,groups):
                split_rows[split].extend((int(i),ids.index(cid),snr) for i in group)
    output.mkdir(parents=True)
    try:
        for split,rows in split_rows.items():
            rows=[rows[i] for i in rng.permutation(len(rows))]
            x=np.stack([radio_features(np.asarray(handle["X"][i],dtype=np.float32)) for i,_,_ in rows])
            y=np.array([label for _,label,_ in rows],dtype=np.int64)
            snr=np.array([s for _,_,s in rows],dtype=np.float32)
            indices=np.array([i for i,_,_ in rows],dtype=np.int64)
            np.savez_compressed(output/(split+".npz"),x=x,y=y,snr_db=snr,source_index=indices)
            print("{}: {} Beispiele, Form {}".format(split,len(rows),x.shape),flush=True)
        metadata=dict(source="RadioML2018.01A",source_file=str(source.resolve()),
            source_bytes=source.stat().st_size,source_mtime_ns=source.stat().st_mtime_ns,
            class_map_sha256=hashlib.sha256(Path(class_map).read_bytes()).hexdigest(),
            original_class_order=labels,labels=selected,selected_y_columns=ids,
            snr_values=snrs,per_class_per_snr=cap,seed=seed,
            preprocessing="rms-v1",target_rms=0.5,iq_samples=1024,
            fft_size=128,hop=32,input_shape=[29,128,1],scale_dbfs=[-80,0],
            split="70/15/remainder percent per class/SNR, disjoint source rows",
            sampling_rate_hz=None,
            limitations="Row split does not prove independent recordings; source full SHA256 not computed",
            license_url="https://www.deepsig.ai/datasets/")
        (output/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
        (output/"IMPORT_COMPLETE").write_text("OK\n")
    except Exception:
        print("Import unvollstaendig; kein IMPORT_COMPLETE. Neuen Ausgabeordner verwenden.")
        raise
    print("Import fertig:",output.resolve())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path,help="Entpackte HDF5-Datei")
    parser.add_argument("--inspect",action="store_true")
    parser.add_argument("--class-map",type=Path)
    parser.add_argument("--classes",nargs="+",default=list(DEFAULT_CLASSES))
    parser.add_argument("--snr-min",type=float,default=0)
    parser.add_argument("--snr-max",type=float,default=20)
    parser.add_argument("--per-bucket",type=int,default=100)
    parser.add_argument("--seed",type=int,default=2030)
    parser.add_argument("--output",type=Path,default=Path("data/radioml2018-v1"))
    args=parser.parse_args()
    if not np.isfinite([args.snr_min,args.snr_max]).all() or args.snr_min>args.snr_max:
        parser.error("Ungueltiger SNR-Bereich")
    if args.per_bucket<20:
        parser.error("--per-bucket muss mindestens 20 sein")
    if not args.inspect and args.class_map is None:
        parser.error("Import erfordert --class-map; zuvor Klassenreihenfolge pruefen")
    try:
        with h5py.File(args.source,"r") as handle:
            validate_file(handle)
            if args.inspect:
                inspect(handle,load_classes(args.class_map) if args.class_map else None)
            else:
                prepare(handle,args.source,args.class_map,args.classes,args.snr_min,args.snr_max,
                        args.per_bucket,args.seed,args.output)
    except (OSError,ValueError,KeyError,SyntaxError) as error:
        parser.exit(1,"RadioML-Import fehlgeschlagen: "+str(error)+"\n")


if __name__=="__main__":
    main()
