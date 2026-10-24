# Lerntest: alle zwölf Klassen

## Anlass

Der Nutzervergleich mit drei Seeds ergab für das bisherige CNN im Mittel 55,4 % (Macro-F1 0,529), für die Harper-Portierung stets 8,33 % (Macro-F1 0,013). Je Referenzlauf wurde nur eine Klasse ausgegeben. Der hochgeladene Verlauf für Seed 2051 zeigt bereits während des Trainings rund 7–9 % Wiedererkennung und Verlust rund 2,485, nahe ln(12). Das ist ein Lernstillstand und kein erst beim Export entstandenes Problem. Die hohe Übereinstimmung unter Frequenzverschiebung ist bei konstanten Ausgaben kein Robustheitsnachweis.

Die bisherigen Funktionstests belegten keinen erfolgreichen Lernvorgang auf dieser Aufgabe. Ein kleiner, ausdrücklich auf Wiedererkennung ausgelegter Lerntest wird deshalb vor weiteren vollständigen Vergleichen durchgeführt.

## Start

```bash
cd ~/rf-learning
source .venv/bin/activate
git -C project pull --ff-only
python project/check_modulation_learning.py
```

Der Ausgabeordner wird automatisch mit Zeitstempel unter `models/modulation12-learning-...` angelegt. Bestehende Ausgaben, Modelle und Datensätze werden nicht überschrieben. Standard ist CUDA; CPU ist explizit mit `--device cpu` möglich.

## Vorab festgelegter Versuch

Nur `train.npz` wird geladen. Validierungs- und Testarrays werden nicht geöffnet. Je Klasse werden zwei feste Beispiele ohne Zurücklegen ausgewählt; bei Signalklassen nur solche mit mindestens 15 dB SNR. Rauschen hat keinen Nutzsignal-SNR. Zusammen sind das 24 Beispiele aller zwölf Klassen, weiterhin mit den im Datensatz vorhandenen Bandbreiten und Frequenzversätzen.

Drei Fälle laufen nacheinander:

1. Bisheriges CNN, Adam-Lernrate 0,001 als Kontrolle des Lernablaufs.
2. Harper-Portierung, Adam-Lernrate 0,001.
3. Harper-Portierung, Adam-Lernrate 0,0001.

Die beiden Harper-Fälle starten nachweislich mit exakt denselben Gewichten. Sie unterscheiden sich nur in der Lernrate. Gleiche Beispiele, ganzer ausgewogener Batch, Seed 2070, keine zusätzliche Augmentation, kein Scheduler, kein Gradient Clipping, PyTorch-Adam mit eps=1e-8. Damit ist dies weiterhin keine vollständige Nachbildung des originalen TensorFlow-Optimierers. Die Kontrolle mit dem bisherigen CNN ist ein Architekturvergleich und isoliert die Lernrate nicht.

Maximal 400 Optimierungsschritte pro Fall. Protokollierung nach Schritt 1, anschließend alle 25 Schritte und am Ende. Der Fall endet vorzeitig, wenn die Wiedererkennung derselben Beispiele mindestens 99 % und deren Verlust höchstens 0,1 erreicht. Bei 24 Beispielen bedeutet diese Schwelle tatsächlich 24/24 richtig. Die Klassifikation wird dabei im Auswertungsmodus gemessen; beim alten CNN bleibt Dropout während der Optimierung aktiv.

Ein nicht bestandener Lauf bedeutet lediglich: Unter diesem Protokoll und innerhalb dieses Schrittbudgets wurde das Ziel nicht erreicht. Er beweist keine prinzipielle Lernunfähigkeit. Eine bestandene Wiedererkennung beweist umgekehrt keine Generalisierung oder Erkennung realer Signale. Schon ein einzelner bestandener Seed genügt nicht zur Freigabe der vollständigen Trainingskette.

## Diagnoseausgaben

- `protocol.json`: Auswahlregeln, Klassen, Einstellungen, Daten-/Code-Prüfsummen und Hash der gemeinsamen Anfangsgewichte.
- `training_selection.npz`: dieselben 24 Beispiele mit Quell-IDs und SNR, damit die Auswahl nachvollziehbar bleibt.
- Je Fall `history.json`: Verlust, Wiedererkennung, Anzahl der Vorhersagen je Klasse, Streuung der Logits zwischen Beispielen.
- Zusätzlich je protokolliertem Schritt: RMS/Streuung/Nullanteil an Faltungs-, linearen und Statistik-Pooling-Ausgaben vor dem Update; Gradientennorm, Gewichtsnorm und tatsächliche Änderung jedes Parameters. Relative Änderung bei vorheriger Gewichtsnorm null wird als null statt als irreführender Quotient gespeichert.
- `report.json`: Anfangs-/Endergebnisse, Matrizen, Laufzeiten und Anfangs-/Endhash der Gewichte.
- `LEARNING_CHECK_COMPLETE`: alle drei Fälle technisch abgeschlossen; nicht gleichbedeutend mit bestandenem Lerntest.

Bitte nach dem Lauf `report.json` sowie die beiden Harper-`history.json` aus dem neuen Ausgabeordner zur Auswertung bereitstellen. Die Dateien mit IQ-Daten und Modellgewichte werden dafür zunächst nicht benötigt.

## Weitere Entscheidung

Kein weiterer vollständiger Sechsfachlauf, bevor die Lerndiagnose ausgewertet ist. Lernen beide Harper-Fälle auf der kleinen Auswahl, müssen Unterschiede zum vollständigen Training gezielt untersucht werden, insbesondere die Zusammensetzung der Batches und die Verteilung der Empfangsbedingungen. Ein Erfolg der niedrigeren Lernrate auf der kleinen Auswahl allein rechtfertigt noch keinen Nachweis einer behobenen Ursache im vollständigen Training. Clipping, Optimiererformulierung und Initialisierung sind getrennte weitere Faktoren.

Hintergrund und Quellen: [MODULATION12_REFERENCE.md](MODULATION12_REFERENCE.md), [Stanford CS231n: Sanity Checks](https://cs231n.github.io/neural-networks-3/), [Autoren-Training](https://github.com/caharper/Automatic-Modulation-Classification-with-Deep-Neural-Networks/blob/902cca45c3bf04b5da9f78030372b5a03f730f42/experiments/radioml2018.01a/run.py), [Keras Adam](https://keras.io/api/optimizers/adam/), [PyTorch 2.9 Adam](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Adam.html).

## Lokaler CPU-Befund vor Veröffentlichung

Auf einem lokal neu erzeugten vollständigen Datensatz gleicher Builder-Konfiguration, Seed 2070 und der oben beschriebenen Auswahl bestanden alle drei Fälle:

| Fall | Schritte bis protokolliertem Bestehen | Wiedererkennung | Verlust |
|---|---:|---:|---:|
| Bisheriges CNN, 0,001 | 75 | 100 % | 0,0066 |
| Harper-Portierung, 0,001 | 175 | 100 % | 0,0567 |
| Harper-Portierung, 0,0001 | 275 | 100 % | 0,0489 |

Das ist ein CPU-Lerntest auf 24 mehrfach verwendeten Beispielen, keine neue Testgenauigkeit und keine Ausführung auf dem Nutzerrechner. Die kleinere Lernrate war hier nicht schneller. Damit sind weder eine grundsätzlich lernunfähige Portierung noch die höhere Lernrate allein als Erklärung des vollständigen Lernstillstands belegt. Der Übergang von diesem ausgewogenen festen Batch zum vollständigen Training mit wechselnden Batches und Bedingungen ist der nächste gezielt zu untersuchende Unterschied. Andere Fehlerquellen sind damit nicht sämtlich ausgeschlossen.

Drei neue Prüftests kontrollieren Auswahl und Reproduzierbarkeit, Gewichtsprüfsummen sowie einen tatsächlich erfolgreichen Lernablauf mit protokollierten Parameteränderungen. Zusätzlich bestanden die sechs bestehenden Modell-/Referenztests. Ein separater Ein-Schritt-Ablauf prüft die endgültige Ausgabeform des Diagnosewerkzeugs.
