"""Diagnose des bestehenden Coral-Modells mit protokollierten Signalparametern."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from signal_lab import LABELS, generate_iq
from coral_worker import CoralClassifier, DEFAULT_MODEL

SNR_VALUES = (-5, 0, 5, 10, 15, 20, 25)
FIELDS = ("id", "actual", "predicted", "correct", "score", "snr_db", "offset_hz",
          "amplitude", "deviation_hz", "audio_hz", "modulation_index",
          "score_noise", "score_carrier", "score_fm", "invoke_ms")


def summary(rows):
    matrix = np.zeros((3,3), dtype=np.int64)
    for row in rows:
        matrix[LABELS.index(row["actual"]), LABELS.index(row["predicted"])] += 1
    recall = {label: (float(matrix[k,k]/matrix[k].sum()) if matrix[k].sum() else None)
              for k,label in enumerate(LABELS)}
    return dict(count=len(rows), accuracy=float(np.trace(matrix)/len(rows)) if rows else None,
                recall=recall, confusion_matrix=matrix.tolist())


def breakdown(rows, field, ranges):
    result = {}
    for name, low, high in ranges:
        selected = [r for r in rows if r[field] is not None and low <= r[field] < high]
        result[name] = summary(selected)
    return result


def run_diagnosis(classifier, per_class, seed, write_row):
    rng = np.random.default_rng(seed)
    rows = []
    for snr in SNR_VALUES:
        group = []
        for label in LABELS:
            for _ in range(per_class):
                iq, params = generate_iq(label, rng, return_metadata=True, snr_db=snr)
                result = classifier.classify(iq)
                predicted = result["label"]
                row = dict(id=len(rows)+1, actual=label, predicted=predicted,
                           correct=int(label==predicted), score=result["scores"][predicted],
                           snr_db=snr, offset_hz=params["offset_hz"],
                           amplitude=params["amplitude"], deviation_hz=params["deviation_hz"],
                           audio_hz=params["audio_hz"],
                           modulation_index=(params["deviation_hz"]/params["audio_hz"]
                                             if label=="fm" else None),
                           score_noise=result["scores"]["noise"],
                           score_carrier=result["scores"]["carrier"],
                           score_fm=result["scores"]["fm"], invoke_ms=result["invoke_ms"])
                write_row(row)
                rows.append(row)
                group.append(row)
        stats = summary(group)
        recalls = stats["recall"]
        print("SNR {:+3d} dB | noise {:5.1%} | carrier {:5.1%} | fm {:5.1%}".format(
            snr,recalls["noise"],recalls["carrier"],recalls["fm"]), flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--per-class", type=int, default=50,
                        help="Beispiele je Klasse UND je SNR-Stufe (Standard 50)")
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--output", type=Path, default=Path("models/signal/diagnostics"))
    args=parser.parse_args()
    if args.per_class<1:
        parser.error("--per-class muss positiv sein")
    if (args.output/"cases.csv").exists() or (args.output/"summary.json").exists():
        parser.error("Ausgabedateien existieren bereits; anderen --output-Ordner waehlen")
    try:
        classifier=CoralClassifier(args.model)
        args.output.mkdir(parents=True,exist_ok=True)
        print("Diagnose ohne Training: {} neue synthetische Beispiele".format(
            args.per_class*len(SNR_VALUES)*len(LABELS)), flush=True)
        print("noise-SNR ist nur der Parameter der Rauschskalierung; kein Nutzsignal vorhanden.")
        with (args.output/"cases.csv").open("w",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=FIELDS)
            writer.writeheader()
            rows=run_diagnosis(classifier,args.per_class,args.seed,writer.writerow)
        report=dict(
            source="synthetic_diagnostic_not_final_test",seed=args.seed,
            per_class_per_snr=args.per_class,snr_values=SNR_VALUES,labels=LABELS,
            model=str(args.model.resolve()),
            model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
            overall=summary(rows),
            by_snr={str(snr):summary([r for r in rows if r["snr_db"]==snr]) for snr in SNR_VALUES},
            by_amplitude=breakdown(rows,"amplitude",[
                ("0.2_to_0.4",0.2,0.4),("0.4_to_0.6",0.4,0.6),("0.6_to_0.9",0.6,0.900001)]),
            by_offset=breakdown(rows,"offset_hz",[
                ("minus60_to_minus20_kHz",-60000,-20000),
                ("minus20_to_plus20_kHz",-20000,20000),
                ("plus20_to_plus60_kHz",20000,60001)]),
            fm_by_modulation_index=breakdown(rows,"modulation_index",[
                ("below_1",0,1),("1_to_3",1,3),("3_and_above",3,float("inf"))]),
            high_score_errors=[r for r in rows if not r["correct"] and r["score"]>=0.8])
        (args.output/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False),
                                               encoding="utf-8")
        print("\nGesamtmatrix (Zeilen Soll, Spalten erkannt):",LABELS)
        print(np.array(report["overall"]["confusion_matrix"]))
        print("Diagnosegenauigkeit: {:.1%}".format(report["overall"]["accuracy"]))
        print("Fehler mit Modellbewertung >=80%:",len(report["high_score_errors"]))
        print("Ergebnisse:",args.output.resolve())
        print("Kein direkter Vergleich zur bisherigen Testgenauigkeit: anderes SNR-Raster.")
    except (OSError,ValueError,ImportError,RuntimeError) as error:
        parser.exit(1,"Diagnose fehlgeschlagen: "+str(error)+"\n")


if __name__=="__main__":
    main()
