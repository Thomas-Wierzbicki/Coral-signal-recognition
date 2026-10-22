"""Direkte I/Q-Daten mit exakt den bestehenden RadioML-Splitzeilen importieren."""
import argparse
import json
from pathlib import Path
import numpy as np
from signal_preprocessing import prepare_iq
from train_radioml import load_dataset


def iq_features(pairs):
    pairs=np.asarray(pairs,dtype=np.float32)
    if pairs.shape!=(1024,2) or not np.isfinite(pairs).all():
        raise ValueError("1024 endliche I/Q-Paare erforderlich")
    iq=prepare_iq(pairs[:,0]+1j*pairs[:,1],"rms-v1")
    return np.stack((iq.real,iq.imag),axis=-1)[None].astype(np.float32)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("--reference",type=Path,default=Path("data/radioml2018-v1"))
    parser.add_argument("--output",type=Path,default=Path("data/radioml2018-iq-v1"))
    args=parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Ausgabe existiert bereits; neuen --output-Ordner verwenden")
        metadata,_=load_dataset(args.reference)
        if args.source.stat().st_size!=metadata["source_bytes"]:
            raise ValueError("Quelldateigroesse weicht vom Referenzimport ab")
        import h5py
        from radioml_import import validate_file
        with h5py.File(args.source,"r") as handle:
            count=validate_file(handle)
            args.output.mkdir(parents=True)
            for split in ("train","validation","test"):
                with np.load(args.reference/(split+".npz"),allow_pickle=False) as data:
                    indices=data["source_index"]
                    y=data["y"]
                    snr=data["snr_db"]
                features=[]
                for index,label,expected_snr in zip(indices,y,snr):
                    if index>=count:
                        raise ValueError("Quellindex ausserhalb HDF5")
                    source_y=np.asarray(handle["Y"][int(index)])
                    source_snr=float(np.asarray(handle["Z"][int(index)]).reshape(-1)[0])
                    if (not np.all((source_y==0)|(source_y==1)) or source_y.sum()!=1
                            or int(source_y.argmax())!=metadata["selected_y_columns"][int(label)]
                            or source_snr!=float(expected_snr)):
                        raise ValueError("Label/SNR stimmt nicht mit Referenz ueberein")
                    features.append(iq_features(handle["X"][int(index)]))
                x=np.stack(features)
                np.savez_compressed(args.output/(split+".npz"),x=x,y=y,snr_db=snr,source_index=indices)
                print(split,len(y),x.shape,flush=True)
        metadata.update(representation="iq-rms-v1",input_shape=[1,1024,2],
                        reference_directory=str(args.reference.resolve()),channel_order=["I","Q"],
                        source_file=str(args.source.resolve()))
        for key in ("fft_size","hop","scale_dbfs"):
            metadata.pop(key,None)
        (args.output/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
        (args.output/"IMPORT_COMPLETE").write_text("OK\n")
        print("I/Q-Import fertig:",args.output.resolve())
        print("Identische Quellzeilen/Labels/SNR und Reihenfolge; keine Aufnahme-Unabhaengigkeit bewiesen.")
    except (OSError,ValueError,KeyError,ImportError) as error:
        parser.exit(1,"I/Q-Import fehlgeschlagen: "+str(error)+"\n")


if __name__=="__main__":
    main()
