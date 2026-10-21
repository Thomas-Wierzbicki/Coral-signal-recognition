"""Gemeinsame, versionierte IQ-Vorverarbeitung fuer Training und Inferenz."""
import json
from pathlib import Path
import numpy as np

MODES = ("raw", "rms-v1")
TARGET_RMS = 0.5


def prepare_iq(iq, mode="raw"):
    if mode not in MODES:
        raise ValueError("Unbekannte Vorverarbeitung: " + str(mode))
    values = np.asarray(iq, dtype=np.complex64)
    if not np.isfinite(values).all() or values.size == 0:
        raise ValueError("Endlicher, nichtleerer IQ-Block erforderlich")
    if mode == "raw":
        return values
    rms = float(np.sqrt(np.mean(np.abs(values.astype(np.complex128)) ** 2)))
    if rms == 0:
        return values.copy()
    return (values.astype(np.complex128) * (TARGET_RMS/rms)).astype(np.complex64)


def model_preprocessing(model):
    """Alte Modelle ohne Manifest bleiben raw; neue Manifeste muessen daneben liegen."""
    manifest = Path(model).parent / "preprocessing.json"
    if not manifest.exists():
        return "raw"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if (data.get("schema") != 1 or data.get("mode") not in MODES
            or data.get("labels") != ["noise", "carrier", "fm"]
            or data.get("input_shape") != [64, 256, 1]):
        raise ValueError("Ungueltiges preprocessing.json neben dem Modell")
    if data["mode"] == "rms-v1" and data.get("target_rms") != TARGET_RMS:
        raise ValueError("RMS-Zielwert des Modells passt nicht zur Implementierung")
    return data["mode"]
