# Training unter Windows

Die Umgebung `coral` bleibt fuer PyCoral/Inferenz.
Training und Konvertierung laufen in einer eigenen Umgebung `coral-train`.
Die Versionen sind bewusst fuer den aelteren Coral-Werkzeugstand festgelegt.

## Aktualisieren

Im vorhandenen Repository-Ordner in Anaconda Prompt:

```text
git pull --ff-only
conda create -n coral-train python=3.9 pip -y
conda activate coral-train
python -m pip install -r requirements-training.txt
python train_signal.py
```

Die vorhandenen Dateien unter `data` werden gelesen, nicht neu erzeugt.
Der Standard trainiert maximal 30 Epochen und stoppt bei ausbleibender
Verbesserung des Validierungsfehlers. CPU reicht fuer diesen kleinen Versuch.

Das Programm erzeugt unter `models/signal`:

- `signal_float.h5`: trainiertes Keras-Modell.
- `signal_int8.tflite`: vollquantisiertes Modell mit uint8-Ein-/Ausgabe.
- `labels.txt`: Klassenreihenfolge.
- `report.json`: Testgenauigkeit, Recall je Klasse, Konfusionsmatrizen und
  Uebereinstimmung der Float- und INT8-Vorhersagen.

Trainingsdaten werden fuer die Quantisierung kalibriert.
Validierungsdaten steuern EarlyStopping; Testdaten werden erst danach ausgewertet.
Das Ergebnis ist ein experimentelles Modell fuer synthetische Daten.
Keine Mindestgenauigkeit wird zugesichert; niedriger Recall einer Klasse
oder deutliche Verluste durch Quantisierung muessen vor dem Hardwareeinsatz untersucht werden.
Mit den kleinen Standarddatensaetzen sind die Kennzahlen nur eine erste Orientierung.
Wiederholtes Optimieren anhand desselben Testdatensatzes entwertet dessen Aussagekraft.

## Danach: Edge-TPU-Compiler

`signal_int8.tflite` ist noch kein Coral-Modell.
Der Edge-TPU-Compiler benoetigt ein geeignetes Linux-x86-64-System.
Dort lautet der eigentliche Kompilierschritt nach Installation des Compilers:

```text
edgetpu_compiler -s -o compiled signal_int8.tflite
```

Den Compilerbericht auf unterstuetzte und auf die CPU ausgelagerte Operationen pruefen.
Die erzeugte `signal_int8_edgetpu.tflite` muss danach auf deinem Coral
gegen die CPU-Ergebnisse getestet werden. Erfolgreiche Konvertierung allein
beweist weder Compiler- noch Laufzeitkompatibilitaet.

Zurueck zur funktionierenden Coral-Umgebung:

```text
conda activate coral
```

## Referenzen

- https://www.tensorflow.org/lite/performance/post_training_quantization
- https://coral.ai/docs/edgetpu/compiler/
