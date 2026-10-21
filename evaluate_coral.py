"""Prueft das kompilierte Funkmodell auf dem Coral gegen data/test.npz."""
import argparse
import json
import time
from pathlib import Path
import numpy as np

LABELS = ["noise", "carrier", "fm"]


def quantize(sample, detail):
    scale, zero = detail["quantization"]
    if detail["dtype"] != np.uint8 or scale <= 0:
        raise ValueError("Erwartet uint8-Eingang mit positiver Quantisierungsskala")
    return np.clip(np.rint(sample / scale + zero), 0, 255).astype(np.uint8)[None, ...]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=Path("models/signal/compiled/signal_int8_edgetpu.tflite"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    try:
        if not args.model.is_file():
            raise ValueError("Kompiliertes Modell fehlt: " + str(args.model))
        metadata = json.loads((args.data / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("labels") != LABELS:
            raise ValueError("Klassenreihenfolge in metadata.json passt nicht")
        with np.load(args.data / "test.npz", allow_pickle=False) as data:
            x, y = data["x"], data["y"]
        if (x.ndim != 4 or x.shape[1:] != (64, 256, 1) or len(x) == 0
                or y.shape != (len(x),) or not np.issubdtype(y.dtype, np.integer)
                or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1
                or set(y.tolist()) != {0, 1, 2}):
            raise ValueError("Ungueltige Testdaten")
        from pycoral.utils.edgetpu import make_interpreter, list_edge_tpus
        print("Coral-Geraete:", list_edge_tpus(), flush=True)
        interpreter = make_interpreter(str(args.model))
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        if tuple(inp["shape"]) != (1, 64, 256, 1) or tuple(out["shape"]) != (1, 3):
            raise ValueError("Modellform passt nicht zu Testdaten/Klassen")
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
        matrix = np.zeros((3, 3), dtype=np.int64)
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
        print("Nur synthetische Testdaten; keine Bewertung echter Empfangssignale.")
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as error:
        parser.exit(1, "Coral-Auswertung fehlgeschlagen: " + str(error) + "\n")


if __name__ == "__main__":
    main()
