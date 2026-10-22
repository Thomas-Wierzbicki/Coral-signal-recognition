"""Eigener RadioML-Trainer fuer importierte 1024-IQ-Spektrogramme."""
import argparse
import json
import os
from pathlib import Path
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
SHAPE=(29,128,1)


def load_dataset(folder):
    if not (folder/"IMPORT_COMPLETE").is_file():
        raise ValueError("IMPORT_COMPLETE fehlt: Import nicht abgeschlossen")
    meta=json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
    labels=meta.get("labels")
    if (meta.get("source")!="RadioML2018.01A" or meta.get("preprocessing")!="rms-v1"
            or meta.get("target_rms")!=0.5 or meta.get("input_shape")!=list(SHAPE)
            or meta.get("fft_size")!=128 or meta.get("hop")!=32
            or not isinstance(labels,list) or len(labels)!=5 or len(set(labels))!=5
            or not all(isinstance(s,str) and s for s in labels)):
        raise ValueError("Metadaten passen nicht zur RadioML-Fuenfklassenvariante")
    splits={}
    seen=set()
    for name in ("train","validation","test"):
        with np.load(folder/(name+".npz"),allow_pickle=False) as data:
            x,y,snr,indices=(data[k] for k in ("x","y","snr_db","source_index"))
        if (x.ndim!=4 or x.shape[1:]!=SHAPE or len(x)==0 or y.shape!=(len(x),)
                or snr.shape!=(len(x),) or indices.shape!=(len(x),)
                or not np.issubdtype(y.dtype,np.integer)
                or not np.issubdtype(indices.dtype,np.integer)
                or not np.isfinite(x).all() or not np.isfinite(snr).all()
                or x.min()<0 or x.max()>1 or set(y.tolist())!=set(range(5))
                or np.any(indices<0)):
            raise ValueError("Ungueltige Arrays in "+name)
        ids=set(indices.tolist())
        if len(ids)!=len(indices) or seen.intersection(ids):
            raise ValueError("Doppelte Quellzeilen innerhalb/zwischen den Aufteilungen")
        seen.update(ids)
        splits[name]=(x.astype(np.float32),y.astype(np.int64),snr)
    return meta,splits


def score(y,pred,labels):
    matrix=np.zeros((len(labels),len(labels)),dtype=np.int64)
    np.add.at(matrix,(y,pred),1)
    return dict(count=len(y),accuracy=float(np.mean(y==pred)),
                confusion_matrix=matrix.tolist(),
                recall={label:(float(matrix[k,k]/matrix[k].sum()) if matrix[k].sum() else None)
                        for k,label in enumerate(labels)})


def predict_integer(interpreter,x):
    interpreter.allocate_tensors()
    inp=interpreter.get_input_details()[0]
    out=interpreter.get_output_details()[0]
    scale,zero=inp["quantization"]
    if inp["dtype"]!=np.uint8 or out["dtype"]!=np.uint8 or scale<=0:
        raise ValueError("Export erwartet uint8-Tensoren und positive Eingangsskala")
    pred=[]
    for sample in x:
        q=np.clip(np.rint(sample/scale+zero),0,255).astype(np.uint8)
        interpreter.set_tensor(inp["index"],q[None])
        interpreter.invoke()
        pred.append(int(np.argmax(interpreter.get_tensor(out["index"])[0])))
    return np.asarray(pred)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("data/radioml2018-v1"))
    parser.add_argument("--output",type=Path,default=Path("models/radioml2018-v1"))
    parser.add_argument("--epochs",type=int,default=50)
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--seed",type=int,default=2031)
    args=parser.parse_args()
    if args.epochs<1 or args.batch_size<1:
        parser.error("Epochen und Batchgroesse muessen positiv sein")
    if args.output.exists():
        parser.error("Ausgabeordner existiert bereits; neuen --output-Ordner verwenden")
    try:
        metadata,splits=load_dataset(args.data)
        import tensorflow as tf
        labels=metadata["labels"]
        train_x,train_y,_=splits["train"]
        val_x,val_y,_=splits["validation"]
        test_x,test_y,test_snr=splits["test"]
        print("Klassen:",labels)
        print("Training/Validierung/Test:",len(train_y),len(val_y),len(test_y))
        print("RadioML-Spektrogrammvariante; kein Vergleich mit der Drei-Klassen-Genauigkeit.")
        tf.keras.utils.set_random_seed(args.seed)
        model=tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=SHAPE),
            tf.keras.layers.Conv2D(16,3,strides=2,padding="same",activation="relu"),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(32,3,strides=2,padding="same",activation="relu"),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32,activation="relu"),
            tf.keras.layers.Dense(len(labels),activation="softmax")])
        model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
        model.fit(train_x,train_y,validation_data=(val_x,val_y),
                  epochs=args.epochs,batch_size=args.batch_size,verbose=2,
                  callbacks=[tf.keras.callbacks.EarlyStopping(
                      monitor="val_loss",patience=8,restore_best_weights=True)])
        converter=tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations=[tf.lite.Optimize.DEFAULT]
        rng=np.random.default_rng(args.seed)
        calibration=np.concatenate([rng.choice(np.flatnonzero(train_y==k),
                                    min(100,int(np.sum(train_y==k))),replace=False)
                                    for k in range(len(labels))])
        def representative():
            for i in calibration:
                yield [train_x[i:i+1]]
        converter.representative_dataset=representative
        converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type=tf.uint8
        converter.inference_output_type=tf.uint8
        quantized=converter.convert()
        float_pred=np.argmax(model.predict(test_x,batch_size=1,verbose=0),axis=1)
        integer_pred=predict_integer(tf.lite.Interpreter(model_content=quantized),test_x)
        report=dict(labels=labels,seed=args.seed,tensorflow_version=tf.__version__,
                    dataset_metadata=metadata,
                    float_model=score(test_y,float_pred,labels),
                    quantized_model=score(test_y,integer_pred,labels),
                    prediction_agreement=float(np.mean(float_pred==integer_pred)),
                    by_snr={str(float(s)):score(test_y[test_snr==s],
                               integer_pred[test_snr==s],labels) for s in np.unique(test_snr)})
        args.output.mkdir(parents=True,exist_ok=False)
        model.save(str(args.output/"radioml_float.h5"),include_optimizer=False)
        (args.output/"radioml_int8.tflite").write_bytes(quantized)
        (args.output/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        np.savez_compressed(args.output/"test_predictions.npz",true=test_y,
                            float_pred=float_pred,int8_pred=integer_pred,snr_db=test_snr)
        manifest=dict(schema=1,task="radioml2018",mode="rms-v1",target_rms=0.5,
                      input_shape=list(SHAPE),labels=labels,iq_samples=1024,fft_size=128,hop=32)
        (args.output/"preprocessing.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        (args.output/"labels.txt").write_text("\n".join(
            "{} {}".format(i,label) for i,label in enumerate(labels))+"\n",encoding="utf-8")
        print("\nTestgenauigkeit Float: {:.1%}".format(report["float_model"]["accuracy"]))
        print("Testgenauigkeit INT8:  {:.1%}".format(report["quantized_model"]["accuracy"]))
        print("INT8-Matrix: Zeilen Soll, Spalten Vorhersage",labels)
        print(np.asarray(report["quantized_model"]["confusion_matrix"]))
        for label,recall in report["quantized_model"]["recall"].items():
            print("{}: {:.1%}".format(label,recall))
        for snr,stats in report["by_snr"].items():
            print("SNR {} dB: {:.1%} (n={})".format(snr,stats["accuracy"],stats["count"]))
        print("Export:",args.output.resolve())
        print("Noch nicht fuer die Edge TPU kompiliert oder auf Coral getestet.")
    except (OSError,ValueError,KeyError,ImportError) as error:
        parser.exit(1,"RadioML-Training fehlgeschlagen: "+str(error)+"\n")


if __name__=="__main__":
    main()
