"""Trainiert ein kleines CNN; exportiert und prueft vollquantisiertes TFLite."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from signal_preprocessing import MODES, TARGET_RMS

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def load_split(folder, name):
    with np.load(folder / (name + ".npz"), allow_pickle=False) as data:
        x, y = data["x"], data["y"]
    if (x.ndim != 4 or x.shape[1:] != (64, 256, 1) or len(x) == 0
            or y.shape != (len(x),) or not np.issubdtype(y.dtype, np.integer)
            or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1
            or set(y.tolist()) != {0, 1, 2}):
        raise ValueError("Ungueltiger Datensatz: " + name)
    return x.astype(np.float32), y.astype(np.int64)


def metrics(y, predictions, labels):
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    np.add.at(matrix, (y, predictions), 1)
    return dict(accuracy=float(np.mean(y == predictions)),
                confusion_matrix_rows_true_columns_predicted=matrix.tolist(),
                recall={label: float(matrix[k, k] / matrix[k].sum())
                        for k, label in enumerate(labels)})


def tflite_predictions(interpreter, x):
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    scale, zero = inp["quantization"]
    if inp["dtype"] != np.uint8 or out["dtype"] != np.uint8 or scale <= 0:
        raise ValueError("Export hat keine gueltigen uint8-Ein-/Ausgaenge")
    bounds = np.iinfo(inp["dtype"])
    results = []
    for sample in x:
        quantized = np.clip(np.rint(sample / scale + zero),
                            bounds.min, bounds.max).astype(inp["dtype"])
        interpreter.set_tensor(inp["index"], quantized[None, ...])
        interpreter.invoke()
        results.append(int(np.argmax(interpreter.get_tensor(out["index"])[0])))
    return np.array(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("models/signal"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs und --batch-size muessen positiv sein")
    try:
        train_x, train_y = load_split(args.data, "train")
        val_x, val_y = load_split(args.data, "validation")
        test_x, test_y = load_split(args.data, "test")
        metadata = json.loads((args.data / "metadata.json").read_text(encoding="utf-8"))
        labels = metadata["labels"]
        mode = metadata.get("preprocessing", "raw")
        if mode not in MODES:
            raise ValueError("Unbekannte Vorverarbeitung in metadata.json")
        if mode == "rms-v1" and metadata.get("target_rms") != TARGET_RMS:
            raise ValueError("RMS-Zielwert in metadata.json passt nicht")
        if mode != "raw" and args.output == Path("models/signal"):
            raise ValueError("Normalisiertes Modell separat speichern: --output models/signal-rms")
        if labels != ["noise", "carrier", "fm"]:
            raise ValueError("Klassenreihenfolge passt nicht zum Training")
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, str(error) + "\nZuerst python signal_lab.py ausfuehren.\n")
    try:
        import tensorflow as tf
    except ImportError:
        parser.exit(1, "TensorFlow fehlt. Separate Umgebung coral-train verwenden; siehe TRAINING.md.\n")
    tf.keras.utils.set_random_seed(args.seed)
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(64, 256, 1)),
        tf.keras.layers.Conv2D(8, 3, strides=2, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(16, 3, strides=2, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(24, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    # Kein Augmentieren anhand von Testdaten. Validation entscheidet nur ueber EarlyStopping.
    model.fit(train_x, train_y, validation_data=(val_x, val_y),
              epochs=args.epochs, batch_size=args.batch_size, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  monitor="val_loss", patience=6, restore_best_weights=True)])
    args.output.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output / "signal_float.h5"), include_optimizer=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Nur Trainingsbeispiele fuer die Kalibrierung; alle Klassen sind enthalten.
    rng = np.random.default_rng(args.seed)
    calibration_indices = np.concatenate([
        rng.choice(np.flatnonzero(train_y == k),
                   min(100, int(np.sum(train_y == k))), replace=False)
        for k in range(3)])
    def representative_data():
        for index in calibration_indices:
            yield [train_x[index:index + 1]]
    converter.representative_dataset = representative_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    quantized = converter.convert()
    quantized_path = args.output / "signal_int8.tflite"
    quantized_path.write_bytes(quantized)

    float_pred = np.argmax(model.predict(test_x, batch_size=1, verbose=0), axis=1)
    quant_pred = tflite_predictions(tf.lite.Interpreter(model_content=quantized), test_x)
    report = dict(
        tensorflow_version=tf.__version__, seed=args.seed, source="synthetic",
        labels=labels, test_examples=len(test_y), preprocessing=metadata,
        float_model=metrics(test_y, float_pred, labels),
        quantized_model=metrics(test_y, quant_pred, labels),
        prediction_agreement=float(np.mean(float_pred == quant_pred)))
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "labels.txt").write_text(
        "\n".join(str(i) + " " + label for i, label in enumerate(labels)) + "\n",
        encoding="utf-8")
    manifest = dict(schema=1, mode=mode,
                    target_rms=TARGET_RMS if mode == "rms-v1" else None,
                    input_shape=[64, 256, 1], labels=labels)
    (args.output / "preprocessing.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print("Vorverarbeitung:", mode)
    print("\nTestgenauigkeit Float: {:.1%}".format(report["float_model"]["accuracy"]))
    print("Testgenauigkeit INT8:  {:.1%}".format(report["quantized_model"]["accuracy"]))
    print("INT8-Konfusionsmatrix (Zeilen: wahr, Spalten: Vorhersage):")
    print(np.array(report["quantized_model"]["confusion_matrix_rows_true_columns_predicted"]))
    print("Klassenreihenfolge:", labels)
    print("Export:", quantized_path.resolve())
    print("Nur synthetische Daten. Noch NICHT fuer Edge TPU kompiliert oder dort getestet.")


if __name__ == "__main__":
    main()
