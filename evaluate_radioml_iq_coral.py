"""Prueft das RadioML-I/Q-Modell auf dem Coral."""
import argparse
import json
import time
from pathlib import Path
import numpy as np

from train_radioml_iq import load_dataset, score


def quantize(sample, detail):
    scale, zero = detail["quantization"]
    if detail["dtype"] != np.uint8 or scale <= 0:
        raise ValueError("Erwartet uint8-Eingang mit positiver Quantisierungsskala")
    return np.clip(np.rint(sample / scale + zero), 0, 255).astype(np.uint8)[None, ...]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=Path("models/radioml2018-iq-v1/compiled/radioml_int8_edgetpu.tflite"))
    parser.add_argument("--data", type=Path, default=Path("data/radioml2018-iq-v1"))
    parser.add_argument('--holdout',action='store_true',help='Nur neuen Holdout-Testsatz auswerten')
    parser.add_argument('--reference',type=Path,default=Path('data/radioml2018-iq-v1'))
    parser.add_argument('--output',type=Path,help='Neuer Ergebnisordner; bei Holdout erforderlich')
    args = parser.parse_args()
    try:
        if not args.model.is_file():
            raise ValueError("Kompiliertes Modell fehlt: " + str(args.model))
        if args.holdout:
            if args.output is None or args.output.exists():
                raise ValueError('Holdout braucht einen neuen --output-Ordner')
            from radioml_holdout import load_holdout
            metadata,x,y,snr,source_ids=load_holdout(args.data,args.reference,args.model)
        else:
            if args.output is not None and args.output.exists():
                raise ValueError('Ausgabeordner existiert bereits')
            metadata,splits=load_dataset(args.data)
            x,y,snr=splits['test']
            with np.load(args.data/'test.npz',allow_pickle=False) as saved:
                source_ids=saved['source_index']
        LABELS=metadata['labels']
        manifest=json.loads((args.model.parent.parent/"preprocessing.json").read_text(encoding="utf-8"))
        if (manifest.get("representation")!="iq-rms-v1" or manifest.get("labels")!=LABELS
                or manifest.get("input_shape")!=[1,1024,2]
                or manifest.get("target_rms")!=0.5 or manifest.get("channel_order")!=["I","Q"]):
            raise ValueError("I/Q-Modellmanifest passt nicht zum Datensatz")
        from pycoral.utils.edgetpu import make_interpreter, list_edge_tpus
        print("Coral-Geraete:", list_edge_tpus(), flush=True)
        interpreter = make_interpreter(str(args.model))
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        if tuple(inp["shape"]) != (1, 1, 1024, 2) or tuple(out["shape"]) != (1, 5):
            raise ValueError("Modellform passt nicht zu Testdaten/Klassen")
        if out["dtype"]!=np.uint8 or out["quantization"][0]<=0:
            raise ValueError("uint8-Ausgang mit positiver Skala erforderlich")
        # Einmal oeffnen und aufwaermen, dann alle Beispiele ueber dieselbe Verbindung.
        interpreter.set_tensor(inp["index"], quantize(x[0], inp))
        interpreter.invoke()
        predictions, timings = [], []
        for number, sample in enumerate(x, 1):
            interpreter.set_tensor(inp["index"], quantize(sample, inp))
            start = time.perf_counter()
            interpreter.invoke()
            timings.append((time.perf_counter() - start) * 1000)
            predictions.append(int(np.argmax(interpreter.get_tensor(out["index"])[0])))
            if number % 100 == 0:
                print(str(number) + "/" + str(len(x)) + " berechnet", flush=True)
        predictions = np.array(predictions)
        matrix = np.zeros((5, 5), dtype=np.int64)
        np.add.at(matrix, (y, predictions), 1)
        accuracy = float(np.mean(y == predictions))
        print("\nCoral-Testgenauigkeit: {:.1%}".format(accuracy))
        print("Konfusionsmatrix: Zeilen wahr, Spalten vorhergesagt")
        print(matrix)
        print("Klassenreihenfolge:", LABELS)
        for k, label in enumerate(LABELS):
            print("{}: {}/{} richtig".format(label, matrix[k, k], matrix[k].sum()))
        print("Inferenz inklusive Python/USB, ohne Vorverarbeitung:")
        print("Median {:.3f} ms; p95 {:.3f} ms".format(
            np.median(timings), np.percentile(timings, 95)))
        report=dict(model=str(args.model.resolve()),dataset_metadata=metadata,
                    coral=score(y,predictions,LABELS),
                    invoke_median_ms=float(np.median(timings)),
                    invoke_p95_ms=float(np.percentile(timings,95)),
                    by_snr={str(float(s)):score(y[snr==s],predictions[snr==s],LABELS)
                            for s in np.unique(snr)})
        reference=args.model.parent.parent/"test_predictions.npz"
        if not args.holdout and reference.is_file():
            with np.load(reference,allow_pickle=False) as saved:
                if not np.array_equal(saved["true"],y) or not np.array_equal(saved["snr_db"],snr):
                    raise ValueError("CPU-Referenz passt nicht zur Testreihenfolge")
                cpu=saved["int8_pred"]
            if cpu.shape!=y.shape:
                raise ValueError("CPU-Referenz hat falsche Form")
            report["cpu_int8_agreement"]=float(np.mean(cpu==predictions))
            print("Uebereinstimmung mit CPU-INT8: {:.1%}".format(report["cpu_int8_agreement"]))
        for level,stats in report["by_snr"].items():
            print("SNR {} dB: {:.1%}".format(level,stats["accuracy"]))
        from radioml_holdout import file_hash
        report['model_sha256']=file_hash(args.model)
        report['holdout']=args.holdout
        target=args.output if args.output is not None else args.model.parent
        if args.output is not None:
            target.mkdir(parents=True,exist_ok=False)
        result=target/"coral_report.json"
        result.write_text(json.dumps(report,indent=2),encoding="utf-8")
        np.savez_compressed(target/"coral_predictions.npz",true=y,pred=predictions,snr_db=snr,source_index=source_ids)
        print("Protokoll:",result.resolve())
        print("Nur synthetische Testdaten; keine Bewertung echter Empfangssignale.")
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as error:
        parser.exit(1, "Coral-Auswertung fehlgeschlagen: " + str(error) + "\n")


if __name__ == "__main__":
    main()

