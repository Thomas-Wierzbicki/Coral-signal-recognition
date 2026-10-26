# Schichtweise Diagnose bestehender Modelle – 26.10.2026

## Bisheriger Befund

Der Full-Batch-Vergleich mit gleichen 400 Schritten ergab auf denselben 120
Trainingsbeispielen 75,0 % bei LR 0,001 und 73,3 % bei LR 0,0001. Die größte
protokollierte Verlustzunahme zwischen aufeinanderfolgenden Schritten betrug
5,898 bzw. 0,022. Die kleinere Lernrate war in diesem Versuch stabiler, nach dem
festen Budget aber nicht besser. Keine der Varianten memorierte alle Beispiele.
Insbesondere FM/FSK sowie QAM bleiben problematisch; die Fehler unterscheiden
sich je Lernrate. Zusammenfassung: [reports/training_lr_summary.json](reports/training_lr_summary.json).
Die Ursache ist nicht vollständig isoliert, kein Generalisierungsnachweis.

## Jetzt ohne neues Training

Die Diagnose öffnet nur die gespeicherte Auswahl und drei vorhandene Zustände:
Anfangsgewichte, letzter Checkpoint LR 0,001, letzter Checkpoint LR 0,0001.
Es werden keine Original-NPZ-Splits, Validierungs- oder Testdaten geöffnet.
Modellgewichte bleiben gleich. Plan vor Ausführung:
[FEATURE_DIAGNOSTIC_PLAN_V1.json](FEATURE_DIAGNOSTIC_PLAN_V1.json).

Auf Schokobaer:

```bash
cd ~/rf-learning/project
git pull --ff-only
cd ~/rf-learning
source .venv/bin/activate
python project/diagnose_training_features.py
```

Wenn genau ein abgeschlossener `models/training-lr-*`-Ordner vorliegt, wird dieser
verwendet und im Ergebnis mit absolutem Pfad dokumentiert. Bei mehreren Läufen
wird keine Auswahl geraten. Dann denselben Befehl mit dem tatsächlichen Ordner
angeben, zum Beispiel `--run models/training-lr-ZEITSTEMPEL`.
Keine Pakete installieren und keine Checkpoints verschieben.

## Messungen und Aussagegrenzen

- Auswahl-Datei, Checkpoints, Anfangszustand und Modellquellen anhand der im
  Trainingsbericht aufgezeichneten SHA256-Werte prüfen.
- Verwechslungsmatrizen der gespeicherten Modelle erneut berechnen und mit dem
  Trainingsbericht vergleichen. Bei Abweichung zuerst Reproduzierbarkeit klären;
  Batchgröße und Rechenbackend können Grenzentscheidungen verändern. Eine gleiche
  Matrix garantiert keine identischen Einzelvorhersagen.
- In 26 Schichten RMS, Anteil exakt null und Verteilung der sechs SE-Gates messen.
- Für zeitliche Aktivierungen Kanal-Mittelwerte und Populationsvarianzen bilden;
  nach Pooling und im Klassifikationskopf die vorhandenen Vektoren verwenden.
- Klassenstreuung und einen festgelegten Nearest-Centroid-Vergleich protokollieren,
  bei dem die eigene Probe aus ihrem Klassenmittel ausgeschlossen wird.

Die Centroid-Zahlen sind beschreibende Trainingsdiagnosen; Modell und Merkmale
wurden bereits an denselben Daten gelernt. Sie sind keine unabhängige Evaluation.
Kanal-Momente verwerfen Zeitstruktur. Schlechte Trennung solcher Beschreibungen
beweist keinen Informationsverlust im vollständigen Tensor. Kleine SE-Gewichte
beweisen keinen Gradientenfehler. Schichtunterschiede lokalisieren Kandidaten für
spätere kontrollierte Eingriffe, keine gesicherte Ursache. Das Verfahren wählt
keine Schicht zur Bereitstellung und trainiert keinen neuen Klassifikator.

## Ausgabe

Neuer Unterordner im gewählten Lauf: `feature-diagnosis-ZEITSTEMPEL`.
Darin `protocol.json`, `report.json` und drei NPZ-Dateien mit Beschreibungen und
SE-Gates. Kein Überschreiben und kein automatischer Upload. Ein Abschlussmarker
erscheint erst nach allen drei Zuständen. Anschließend die neue **report.json**
aus diesem Unterordner hochladen. Vorherige Trainingsergebnisse bleiben erhalten.

Technische Tests verwenden künstliche Daten; keine Aussage über echte Signale,
Coral, Drohnenprotokolle oder Erkennung auf unbekannten Beispielen. Alle zwölf
Klassen bleiben erhalten; das historische FP32-Gate bleibt unverändert.

Prüfung vor Veröffentlichung: vier neue automatisierte Tests bestanden, darunter
vollständige Dateiverarbeitung aller drei Zustände mit 120 künstlichen Blöcken
von jeweils 4096 komplexen Werten, Prüfsummen-Ablehnung, unveränderte Gewichte,
entfernte Hooks, Populationsvarianz und Ausschluss der eigenen Probe aus ihrem
Klassenmittel. Acht bestehende Trainings-/Modelltests bestehen ebenfalls.
Umgebung: Python 3.12, PyTorch 2.9.1+cpu, NumPy 2.5.2. Nutzer-Checkpoint-Auswertung
auf der RTX noch nicht ausgeführt.
