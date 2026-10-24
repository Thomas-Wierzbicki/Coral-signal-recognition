# GPU-Referenztraining: zwölf Klassen

Der Datensatz `data/modulation12-v1` wurde auf dem Trainingsrechner erfolgreich erzeugt und zurückgelesen: 5040 Trainings-, 1080 Validierungs- und 1080 Testbeispiele. Ziel ist zunächst eine überprüfbare Modulationserkennung; Drohnenprotokolle bleiben ein späteres eigenes Arbeitspaket.

## Start unter WSL

```bash
cd ~/rf-learning
source .venv/bin/activate
git -C project pull --ff-only
python project/train_modulation12.py
python -m pip freeze > models/modulation12-v1/environment-freeze.txt
```

Die bereits geprüfte Umgebung verwendet TorchSig 2.2.0 und PyTorch 2.9.1+cu126. Das Training fordert standardmäßig CUDA an. Es überschreibt keinen vorhandenen Ausgabeordner. Für einen ausdrücklich gewünschten neuen Lauf beispielsweise `--output models/modulation12-run2` angeben.

## Modell und Auswahl

Eingabe: 4096 komplexe I/Q-Abtastwerte bei 32 kHz, komplexer Block-RMS 0,5. Im Netz werden I und Q als zwei Kanäle eines Tensors `[N,2,1,4096]` angeordnet. Das Modell hat 287812 Parameter: drei Conv2D/ReLU/Pooling-Stufen, eine dichte Schicht mit 128 Einheiten, Dropout 0,2 und zwölf Ausgabelogits.

Klassenfolge: am-dsb, fm, bpsk, qpsk, 16qam, 64qam, 2fsk, 2gfsk, 2gmsk, ofdm-64, ook, noise. Noise ist eine Erkennungskategorie, keine Modulationsart.

Adam mit Lernrate 0,001, Batchgröße 32, maximal 40 Epochen, Seed 2051. Die Auswahl erfolgt ausschließlich nach kleinstem Validierungsverlust; nach acht Epochen ohne Verbesserung endet das Training. Der Testsatz wird erst danach geladen. Die Datensatzdateien werden anhand ihrer SHA256-Prüfsummen und Quell-IDs kontrolliert. Es gibt hier noch keine zusätzliche Trainingsaugmentation.

## Auswertung und Dateien

`report.json` enthält Testgenauigkeit, Macro-F1, Konfusionsmatrix, Präzision/Recall je Klasse sowie Ergebnisse nach SNR. Macro-F1 verwendet immer alle zwölf Klassen, auch in Teilgruppen ohne Beispiele einzelner Klassen. Rauschen hat keinen Nutzsignal-SNR und wird separat aufgeführt.

Zwei vorab festgelegte, gepaarte Kontrollen verschieben dieselben Testblöcke um -1600 beziehungsweise +1600 Hz. Angegeben werden Trefferquote und Übereinstimmung mit den ursprünglichen Vorhersagen. Diese Kontrollen prüfen Frequenzempfindlichkeit; sie liefern keine unabhängigen zusätzlichen Testbeispiele und prüfen keine Abtastratenänderung.

Weitere Ausgaben:

- `best_state_dict.pt`: ausgewählte Modellgewichte.
- `model_traced.pt`: Float-Inferenzmodell im TorchScript-Format; nach dem Speichern erneut geladen und numerisch geprüft.
- `preprocessing.json`: Klassenfolge, Eingabeformat, RMS-Vorgabe und Modellhash.
- `training_config.json`, `history.json`: Einstellungen und Lernverlauf.
- `test_predictions.npz`, `confusion_matrix.png`: Einzelvorhersagen und Matrix.
- `TRAINING_COMPLETE`: wird erst nach erfolgreicher Auswertung und Exportprüfung geschrieben.

Die zwölf Logits sind keine kalibrierten Wahrscheinlichkeiten. Eine unbekannte Signalart kann bislang keiner eigenen Unbekannt-Kategorie zugeordnet werden.

## Aussagegrenzen und nächste Entscheidung

Ein vollständiger Ein-Epochen-CPU-Probelauf mit allen 7200 Beispielen einschließlich Export und erneutem Laden wurde geprüft; daraus wird keine Erkennungsleistung abgeleitet. Das eigentliche GPU-Training muss auf dem Trainingsrechner ausgeführt werden.

Trainings-, Validierungs- und Testsignale stammen aus derselben synthetischen Generatorkette. Gute Ergebnisse belegen daher zunächst nur diese Aufgabe. Die Klassen können trotz gemeinsamer Parameterbereiche unterschiedliche Bandbreitenverteilungen haben. Reale Empfänger, Fading, Bursts, andere Generatoren und Drohnenprotokolle sind dadurch nicht validiert.

Nach dem ersten GPU-Lauf werden Matrix, Lernverlauf und beide Frequenzkontrollen gemeinsam bewertet. Erst danach folgen gezielte Robustheitsprüfungen und unabhängige reale Aufnahmen. Dieses PyTorch-Modell ist noch kein TFLite-/Coral-Modell; eine spätere Konvertierung und vollständige Hardware-Abbildung müssen separat nachgewiesen werden.
