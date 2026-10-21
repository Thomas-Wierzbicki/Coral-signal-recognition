"""Reproduzierbare synthetische IQ-Daten und Spektrogramme; keine Erkennung."""
import argparse
import json
from pathlib import Path
import numpy as np

LABELS = ("noise", "carrier", "fm")
SAMPLE_RATE = 240000
FFT_SIZE = 256
FRAMES = 64
SAMPLES = FFT_SIZE * FRAMES


def generate_iq(label, rng, *, return_metadata=False, snr_db=None):
    t = np.arange(SAMPLES, dtype=np.float64) / SAMPLE_RATE
    offset = rng.uniform(-60000, 60000)
    amplitude = rng.uniform(0.2, 0.9)
    sampled_snr = rng.uniform(-5, 25)
    snr_db = sampled_snr if snr_db is None else float(snr_db)
    if not np.isfinite(snr_db) or not -100 <= snr_db <= 100:
        raise ValueError("SNR muss endlich und zwischen -100 und 100 dB liegen")
    phase0 = rng.uniform(-np.pi, np.pi)
    deviation = rng.uniform(1500, 6000)
    audio_hz = rng.uniform(300, 3000)
    phase = 2 * np.pi * offset * t + phase0
    if label == "fm":
        phase += deviation / audio_hz * np.sin(
            2 * np.pi * audio_hz * t + rng.uniform(-np.pi, np.pi))
    elif label not in ("carrier", "noise"):
        raise ValueError(label)
    sigma = amplitude * 10 ** (-snr_db / 20) / np.sqrt(2)
    iq = sigma * (rng.standard_normal(SAMPLES) + 1j * rng.standard_normal(SAMPLES))
    if label != "noise":
        iq += amplitude * np.exp(1j * phase)
    iq = iq.astype(np.complex64)
    if return_metadata:
        metadata = dict(label=label, snr_db=float(snr_db), offset_hz=float(offset),
                        amplitude=float(amplitude),
                        deviation_hz=float(deviation) if label == "fm" else None,
                        audio_hz=float(audio_hz) if label == "fm" else None)
        return iq, metadata
    return iq


def spectrogram(iq):
    """Zeit x Frequenz, feste dBFS-Skalierung; gleiche Funktion fuer spaetere Inferenz."""
    window = np.hanning(FFT_SIZE)
    blocks = np.asarray(iq).reshape(FRAMES, FFT_SIZE)
    fft = np.fft.fftshift(np.fft.fft(blocks * window, axis=1), axes=1)
    power = np.abs(fft / window.sum()) ** 2
    db = 10 * np.log10(np.maximum(power, 1e-12))
    return np.clip((db + 80) / 80, 0, 1).astype(np.float32)[..., None]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-class", type=int, default=200)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", type=Path, default=Path("data"))
    args = p.parse_args()
    if args.per_class < 10:
        p.error("--per-class muss mindestens 10 sein")
    args.output.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    # Separate Zufallsstroeme; keine Ausschnitte derselben Aufnahme in verschiedenen Splits.
    streams = np.random.SeedSequence(args.seed).spawn(3)
    counts = (args.per_class, max(10, args.per_class // 5), max(10, args.per_class // 5))
    for split, count, seed in zip(("train", "validation", "test"), counts, streams):
        rng = np.random.default_rng(seed)
        x, y = [], []
        for index, label in enumerate(LABELS):
            for number in range(count):
                feature = spectrogram(generate_iq(label, rng))
                x.append(feature)
                y.append(index)
                if number == 0:
                    Image.fromarray((feature[..., 0] * 255).astype(np.uint8)).save(
                        args.output / (split + "_" + label + ".png"))
        order = rng.permutation(len(y))
        np.savez_compressed(args.output / (split + ".npz"),
                            x=np.stack(x)[order], y=np.array(y, dtype=np.int64)[order])
        print(split + ": " + str(len(y)) + " Beispiele")
    metadata = dict(labels=LABELS, sample_rate=SAMPLE_RATE, fft_size=FFT_SIZE,
                    frames=FRAMES, seed=args.seed, scale_dbfs=[-80, 0],
                    source="synthetic", version=1)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Fertig. PNG-Vorschauen und Datensaetze in: " + str(args.output.resolve()))
    print("Noch kein trainiertes Modell; keine Aussage zur Erkennung echter Funksignale.")


if __name__ == "__main__":
    main()
