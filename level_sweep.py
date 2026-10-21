"""Kontrollierter Pegeltest: gleiche IQ-Bloecke bei acht Amplituden, konstantes SNR."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from signal_lab import generate_iq
from coral_worker import CoralClassifier, DEFAULT_MODEL

AMPLITUDES = (0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)
SNR_VALUES = (0,10,25)
CLASSES = ("carrier","fm")


def rescale_iq(iq, source_amplitude, target_amplitude):
    if source_amplitude <= 0 or target_amplitude <= 0:
        raise ValueError("Amplituden muessen positiv sein")
    # Nutzsignal UND bereits erzeugtes Rauschen gemeinsam skalieren.
    return (iq * (target_amplitude/source_amplitude)).astype(np.complex64)


def run_sweep(classifier, per_class, seed, write_row):
    rng=np.random.default_rng(seed)
    rows, pairs = [], []
    for snr in SNR_VALUES:
        for label in CLASSES:
            for number in range(per_class):
                original,params=generate_iq(label,rng,return_metadata=True,snr_db=snr)
                pair_id="{}_{}_{}".format(snr,label,number+1)
                pair_rows=[]
                for amplitude in AMPLITUDES:
                    iq=rescale_iq(original,params["amplitude"],amplitude)
                    result=classifier.classify(iq)
                    row=dict(pair_id=pair_id,actual=label,snr_db=snr,
                             target_amplitude=amplitude,source_amplitude=params["amplitude"],
                             offset_hz=params["offset_hz"],deviation_hz=params["deviation_hz"],
                             audio_hz=params["audio_hz"],predicted=result["label"],
                             correct=int(result["label"]==label),
                             score_noise=result["scores"]["noise"],
                             score_carrier=result["scores"]["carrier"],
                             score_fm=result["scores"]["fm"])
                    rows.append(row)
                    pair_rows.append(row)
                    write_row(row)
                predicted=[r["predicted"] for r in pair_rows]
                true_scores=[r["score_"+label] for r in pair_rows]
                pairs.append(dict(pair_id=pair_id,actual=label,snr_db=snr,
                                  labels=predicted,label_changed=len(set(predicted))>1,
                                  true_score_range=max(true_scores)-min(true_scores)))
        print("\nSNR {:+d} dB, {} verschiedene Grundsignale:".format(snr,2*per_class),flush=True)
        for amplitude in AMPLITUDES:
            selected=[r for r in rows if r["snr_db"]==snr and r["target_amplitude"]==amplitude]
            recalls={label:sum(r["correct"] for r in selected if r["actual"]==label)/per_class
                     for label in CLASSES}
            print("Amplitude {:.1f} | carrier {:5.1%} | fm {:5.1%}".format(
                amplitude,recalls["carrier"],recalls["fm"]),flush=True)
    return rows,pairs


def build_report(rows,pairs):
    by_condition=[]
    for snr in SNR_VALUES:
        for amplitude in AMPLITUDES:
            selected=[r for r in rows if r["snr_db"]==snr and r["target_amplitude"]==amplitude]
            matrix=np.zeros((3,3),dtype=np.int64)
            labels=("noise","carrier","fm")
            for row in selected:
                matrix[labels.index(row["actual"]),labels.index(row["predicted"])]+=1
            by_condition.append(dict(snr_db=snr,amplitude=amplitude,
                                     count=len(selected),confusion_matrix=matrix.tolist()))
    return dict(count=len(rows),base_signals=len(pairs),
                pairs_with_label_changes=sum(p["label_changed"] for p in pairs),
                by_condition=by_condition,pairs=pairs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,default=DEFAULT_MODEL)
    parser.add_argument("--per-class",type=int,default=20,
                        help="Grundsignale je Klasse UND SNR-Stufe; jedes wird achtfach skaliert")
    parser.add_argument("--seed",type=int,default=2029)
    parser.add_argument("--output",type=Path,default=Path("models/signal/level-sweep"))
    args=parser.parse_args()
    if args.per_class<1:
        parser.error("--per-class muss positiv sein")
    if (args.output/"cases.csv").exists() or (args.output/"summary.json").exists():
        parser.error("Ergebnisse existieren bereits; neuen --output-Ordner verwenden")
    try:
        model_hash=hashlib.sha256(args.model.read_bytes()).hexdigest()
        classifier=CoralClassifier(args.model)
        args.output.mkdir(parents=True,exist_ok=True)
        print("Kontrollierter Pegeltest ohne Training.")
        print("Pro Grundsignal bleiben Rauschen, SNR, Frequenz, Phase und Modulation gleich.")
        fields=("pair_id","actual","snr_db","target_amplitude","source_amplitude",
                "offset_hz","deviation_hz","audio_hz","predicted","correct",
                "score_noise","score_carrier","score_fm")
        with (args.output/"cases.csv").open("w",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader()
            rows,pairs=run_sweep(classifier,args.per_class,args.seed,writer.writerow)
        report=build_report(rows,pairs)
        report.update(seed=args.seed,snr_values=SNR_VALUES,amplitudes=AMPLITUDES,
                      model_sha256=model_hash,model=str(args.model.resolve()),
                      source="paired_synthetic_amplitude_sweep",
                      matrix_labels=["noise","carrier","fm"])
        (args.output/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False),
                                               encoding="utf-8")
        print("\n{} Grundsignale, {} Klassifikationen.".format(len(pairs),len(rows)))
        print("Klassenwechsel allein durch Skalierung: {}/{} Grundsignale".format(
            report["pairs_with_label_changes"],len(pairs)))
        for snr in SNR_VALUES:
            for label in CLASSES:
                subset=[p for p in pairs if p["snr_db"]==snr and p["actual"]==label]
                print("SNR {:+d} dB | {}: {}/{} mit Klassenwechsel".format(
                    snr,label,sum(p["label_changed"] for p in subset),len(subset)))
        print("Ergebnisse:",args.output.resolve())
        print("Skalierte Varianten sind gepaart, keine unabhaengigen Testbeispiele.")
        print("Ein Klassenwechsel belegt Pegelabhaengigkeit dieser Verarbeitungskette,")
        print("aber noch nicht deren Ursache oder die Wirksamkeit einer Normalisierung.")
    except (OSError,ValueError,ImportError,RuntimeError) as error:
        parser.exit(1,"Pegeltest fehlgeschlagen: "+str(error)+"\n")


if __name__=="__main__":
    main()
