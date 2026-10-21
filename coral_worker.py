"""Persistenter Coral-Prozess: JSON-Zeilen auf stdin/stdout, IQ als Base64."""
import argparse
import base64
import binascii
import json
import sys
import time
from pathlib import Path
import numpy as np
from signal_lab import SAMPLE_RATE, SAMPLES, spectrogram

LABELS = ("noise", "carrier", "fm")
MAX_LINE = 256 * 1024
DEFAULT_MODEL = Path(__file__).resolve().parent / "models/signal/compiled/signal_int8_edgetpu.tflite"


def decode_iq(request):
    if request.get("sample_rate") != SAMPLE_RATE:
        raise ValueError("sample_rate muss 240000 sein; Resampling erfolgt nicht automatisch")
    encoded = request.get("iq_base64")
    if not isinstance(encoded, str):
        raise ValueError("iq_base64 fehlt")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Ungueltige Base64-Daten") from error
    if len(raw) != SAMPLES * 8:
        raise ValueError("Erwartet genau 16384 komplexe float32-Samples")
    iq = np.frombuffer(raw, dtype="<c8")
    if not np.isfinite(iq).all():
        raise ValueError("IQ enthaelt NaN oder Unendlich")
    return iq


class CoralClassifier:
    def __init__(self, model, factory=None):
        if factory is None:
            if not Path(model).is_file():
                raise ValueError("Kompiliertes Modell fehlt: " + str(model))
            from pycoral.utils.edgetpu import make_interpreter
            factory = make_interpreter
        self.interpreter = factory(str(model))
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]
        for detail, shape in ((self.inp, (1, 64, 256, 1)), (self.out, (1, 3))):
            if tuple(detail["shape"]) != shape or detail["dtype"] != np.uint8:
                raise ValueError("Modell muss uint8 und die vereinbarte Tensorform verwenden")
            if detail["quantization"][0] <= 0:
                raise ValueError("Ungueltige Quantisierung")
        self.interpreter.set_tensor(self.inp["index"], np.zeros((1, 64, 256, 1), np.uint8))
        self.interpreter.invoke()  # Modelltransfer vor ready abschliessen.

    def classify(self, iq):
        start_total = time.perf_counter()
        feature = spectrogram(iq)
        scale, zero = self.inp["quantization"]
        tensor = np.clip(np.rint(feature / scale + zero), 0, 255).astype(np.uint8)[None]
        self.interpreter.set_tensor(self.inp["index"], tensor)
        start_invoke = time.perf_counter()
        self.interpreter.invoke()
        invoke_ms = (time.perf_counter() - start_invoke) * 1000
        scale, zero = self.out["quantization"]
        scores = (self.interpreter.get_tensor(self.out["index"])[0].astype(float) - zero) * scale
        best = int(np.argmax(scores))
        return dict(label=LABELS[best], scores=dict(zip(LABELS, scores.tolist())),
                    invoke_ms=invoke_ms, processing_ms=(time.perf_counter()-start_total)*1000)


def send(message):
    print(json.dumps(message, allow_nan=False), flush=True)


def serve(classifier, stream):
    send(dict(type="ready", protocol=1, sample_rate=SAMPLE_RATE,
              samples=SAMPLES, labels=LABELS))
    while True:
        line = stream.readline(MAX_LINE + 1)
        if not line:
            return 0
        if len(line) > MAX_LINE:
            send(dict(type="error", id=None, message="Nachricht zu lang; Prozess wird beendet"))
            return 1
        request_id = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("JSON-Objekt erwartet")
            request_id = request.get("id")
            if request_id is not None and not isinstance(request_id, (str, int)):
                request_id = None
                raise ValueError("id muss eine Zeichenfolge oder Ganzzahl sein")
            if request.get("type") == "shutdown":
                send(dict(type="bye", id=request_id))
                return 0
            if request.get("type") != "classify":
                raise ValueError("type muss classify oder shutdown sein")
            result = classifier.classify(decode_iq(request))
            send(dict(type="result", id=request_id, **result))
        except (ValueError, KeyError, RuntimeError) as error:
            send(dict(type="error", id=request_id, message=str(error)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    try:
        classifier = CoralClassifier(args.model)
    except Exception as error:
        send(dict(type="error", id=None, message="Initialisierung: " + str(error)))
        return 1
    return serve(classifier, sys.stdin)


if __name__ == "__main__":
    sys.exit(main())
