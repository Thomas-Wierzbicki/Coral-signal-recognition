"""Sendet wechselnde synthetische Signale an einen einzigen Coral-Unterprozess."""
import argparse
import base64
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import numpy as np
from signal_lab import LABELS, SAMPLE_RATE, generate_iq
from coral_worker import DEFAULT_MODEL


def read_responses(stream, responses):
    try:
        for line in stream:
            responses.put(json.loads(line))
    except Exception as error:
        responses.put(dict(type="error", message=str(error)))
    finally:
        responses.put(dict(type="eof"))


def receive(responses, timeout):
    try:
        message = responses.get(timeout=timeout)
    except queue.Empty as error:
        raise RuntimeError("Keine Antwort des Coral-Prozesses innerhalb des Zeitlimits") from error
    if message.get("type") in ("error", "eof"):
        raise RuntimeError(message.get("message", "Coral-Prozess hat sich beendet"))
    return message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--preprocessing", choices=("raw", "rms-v1"), default=None)
    args = parser.parse_args()
    if args.per_class < 1 or not np.isfinite(args.interval) or args.interval < 0:
        parser.error("Positive Beispielzahl und endliches, nichtnegatives Intervall erforderlich")
    worker = Path(__file__).resolve().with_name("coral_worker.py")
    command = [sys.executable, "-u", str(worker), "--model", str(args.model.resolve())]
    if args.preprocessing is not None:
        command.extend(["--preprocessing", args.preprocessing])
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        encoding="utf-8", bufsize=1)
    responses = queue.Queue()
    reader = threading.Thread(target=read_responses, args=(process.stdout, responses), daemon=True)
    reader.start()
    matrix = np.zeros((3, 3), dtype=np.int64)
    elapsed = []
    rng = np.random.default_rng(args.seed)
    try:
        ready = receive(responses, 60)
        if ready.get("type") != "ready" or ready.get("protocol") != 1:
            raise RuntimeError("Unbekanntes Startprotokoll")
        if args.preprocessing is not None and ready.get("preprocessing") != args.preprocessing:
            raise RuntimeError("Worker verwendet nicht die erwartete Vorverarbeitung")
        print("Vorverarbeitung:", ready.get("preprocessing", "nicht gemeldet"), flush=True)
        print("Coral bereit; das Modell bleibt fuer alle Signale geladen.", flush=True)
        print("Synthetischer Schnittstellentest, kein Live-Empfang und kein unabhaengiger Benchmark.")
        request_id = 0
        for actual_index, actual in enumerate(LABELS):
            print("\nQuelle: " + actual, flush=True)
            for _ in range(args.per_class):
                request_id += 1
                iq = generate_iq(actual, rng).astype("<c8")
                request = dict(type="classify", id=request_id, sample_rate=SAMPLE_RATE,
                               iq_base64=base64.b64encode(iq.tobytes()).decode("ascii"))
                start = time.perf_counter()
                process.stdin.write(json.dumps(request) + "\n")
                process.stdin.flush()
                result = receive(responses, 10)
                total_ms = (time.perf_counter() - start) * 1000
                if result.get("type") != "result" or result.get("id") != request_id:
                    raise RuntimeError("Antwort passt nicht zum Auftrag")
                predicted = result["label"]
                matrix[actual_index, LABELS.index(predicted)] += 1
                elapsed.append(total_ms)
                print("Soll {:7s} | erkannt {:7s} | Bewertung {:5.1f}% | TPU-Aufruf {:.3f} ms | Anfrage {:.2f} ms".format(
                    actual, predicted, 100*result["scores"][predicted], result["invoke_ms"], total_ms),
                    flush=True)
                time.sleep(args.interval)
        print("\nKonfusionsmatrix: Zeilen Soll, Spalten Vorhersage", LABELS)
        print(matrix)
        print("Treffer: {}/{}".format(int(np.trace(matrix)), int(matrix.sum())))
        print("Anfrage-Median inklusive Uebertragung/Vorverarbeitung: {:.2f} ms".format(np.median(elapsed)))
        print("Modellbewertungen sind keine garantierten Wahrscheinlichkeiten.")
        process.stdin.write(json.dumps(dict(type="shutdown")) + "\n")
        process.stdin.flush()
        if receive(responses, 10).get("type") != "bye":
            raise RuntimeError("Bestaetigung zum Beenden fehlt")
        process.wait(timeout=10)
        return 0
    except KeyboardInterrupt:
        print("\nTest abgebrochen.")
        return 130
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
        print("Test fehlgeschlagen: " + str(error), file=sys.stderr)
        return 1
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        process.stdin.close()
        process.stdout.close()
        reader.join(timeout=1)


if __name__ == "__main__":
    sys.exit(main())
