"""Coral-Geraetetest: ein Modell einmal laden, wiederholt ausfuehren."""
import argparse
import time
import urllib.request
from pathlib import Path
import numpy as np

MODEL_URL = ("https://raw.githubusercontent.com/google-coral/test_data/master/"
             "mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--model", type=Path, default=Path("models/bird_edgetpu.tflite"))
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs muss positiv sein")
    try:
        from pycoral.utils.edgetpu import make_interpreter, list_edge_tpus, get_runtime_version
        print("Geraete:", list_edge_tpus())
        print("Laufzeit:", get_runtime_version())
        if not args.model.exists():
            args.model.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.model.with_suffix(".download")
            print("Lade offizielles Vogel-Testmodell herunter ...")
            urllib.request.urlretrieve(MODEL_URL, temporary)
            temporary.replace(args.model)
        interpreter = make_interpreter(str(args.model))
        interpreter.allocate_tensors()
        info = interpreter.get_input_details()[0]
        interpreter.set_tensor(info["index"], np.zeros(info["shape"], dtype=info["dtype"]))
        interpreter.invoke()  # Initialisierung und Modelltransfer nicht mitmessen.
        timings = []
        for _ in range(args.runs):
            start = time.perf_counter()
            interpreter.invoke()
            timings.append((time.perf_counter() - start) * 1000)
        print(str(args.runs) + " Berechnungen erfolgreich")
        print("Laufzeit inklusive Python/USB: Median {:.2f} ms, p95 {:.2f} ms".format(
            np.median(timings), np.percentile(timings, 95)))
        print("Hardwaretest bestanden. Dieses Modell erkennt KEINE Funksignale.")
    except Exception as error:
        parser.exit(1, "Coral-Test fehlgeschlagen: " + str(error) + "\n"
                    "Anaconda Prompt: conda activate coral; USB und Edge-TPU-Laufzeit pruefen.\n")


if __name__ == "__main__":
    main()
