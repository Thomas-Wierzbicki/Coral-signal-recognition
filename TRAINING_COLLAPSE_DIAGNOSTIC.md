# Diagnose des Trainingsstillstands – Versuch V1

Stand 26.10.2026. Der Plan wurde vor Ausführung separat festgeschrieben:
[TRAINING_COLLAPSE_PLAN_V1.json](TRAINING_COLLAPSE_PLAN_V1.json).

Die historischen FP32-Grenzfälle bleiben Fehlschläge des ursprünglichen Gates.
Die lokalisierte ReLU-Grenze und TF-Biassummierung rechtfertigen keine pauschale
numerische Freigabe. Dieser Versuch ist eine begrenzte Trainingsdiagnose mit
expliziter SELU-Auswahl, keine Literaturreproduktion und kein Leistungsbenchmark.
Die ursprünglichen Trainingsprogramme und ihre Standardaktivierung bleiben gleich.

## Ausführen auf Schokobaer

Im WSL-Terminal:

```bash
cd ~/rf-learning/project
git pull --ff-only
cd ~/rf-learning
source .venv/bin/activate
python project/diagnose_training_collapse.py
```

Keine neuen Pakete in der vorhandenen TorchSig-/PyTorch-Umgebung installieren.
Der Versuch wählt ausschließlich aus dem hashgeprüften Trainingssplit zehn
Beispiele je Klasse (Signal-SNR >=15 dB, Rauschen ohne SNR). Alle zwölf Klassen
bleiben erhalten. Validierungs- und Testdateien werden nicht geöffnet.

Zwei frische Adam-Optimierer, identische Anfangsgewichte und dieselben 120
Beispiele: 100 Epochen Full-Batch und 100 Epochen gemischte Mini-Batches mit 32.
Die korrigierte TensorFlow-kompatible SELU wird ausdrücklich verwendet.
`--activation historical` ist ein gesonderter abweichender Versuch; Ergebnisse
beider Varianten dürfen nicht vermischt werden. Änderungen von `--epochs` oder
`--per-class` sind ebenfalls als Abweichungen im Protokoll markiert.

## Auswerten und sichern

Automatisch neuer Ordner `models/training-collapse-ZEITSTEMPEL`. Enthalten sind
Auswahl-NPZ, Quell-/Datenhashes und Umgebung im `protocol.json`, zwei detaillierte
`history.json`, die letzten Modell-/Adam-Zustände und ein `report.json`.
Dateien werden pro Epoche atomar ersetzt. `DIAGNOSTIC_COMPLETE` erscheint erst
nach beiden Fällen. Kein automatisches Resume; ein abgebrochener Ordner bleibt
unvollständig und wird nicht überschrieben. Den vollständigen Ausgabeordner für
eine Sicherung kopieren; es erfolgt kein Upload.

Zunächst `report.json` übermitteln; für Zeitverläufe zusätzlich beide
`history.json`. Angegebene Wiedererkennung betrifft dieselben Trainingsbeispiele,
keine Testgenauigkeit. Normen aller Parameter/Gradienten/Updates werden am ersten
Batch jeder Epoche erfasst, Klassenbelegung und Verlust an jedem Batch.

Full-Batch hat einen, Mini-Batch vier Optimierungsschritte pro Epoche. Daher werden
Schritte und verarbeitete Beispiele ausgewiesen. Ein Unterschied allein beweist
keine isolierte Batchgrößenursache. Bei auseinanderlaufendem Verhalten folgt ein
separat geplanter Vergleich der Optimierungsbudgets. Bei beidseitigem Stillstand
folgt die Untersuchung der frühesten Abschwächung von Aktivierungen/Gradienten.
Ein einzelner Seed ist explorativ; keine Modellauswahl nach diesen Zahlen.

## Lokale technische Prüfung

`test_training_collapse.py` prüft den Ablauf an 36 künstlichen kurzen I/Q-Blöcken:
Trainingssplit exklusiv, unterschiedliche Batchzahlen, vollständige Matrizen,
Gradientenprotokoll, Checkpoints, Abschlussmarker und Schutz vor Überschreiben.
Das ist ein Softwaretest, kein Ergebnis auf dem echten Datensatz oder Nutzer-GPU.

Prüfung vor Veröffentlichung: 7 automatisierte Tests bestanden (PyTorch 2.9.1+cpu,
NumPy 2.5.2, Python 3.12). Zusätzlich Vorwärts-/Rückwärtslauf mit echten
4096 Abtastwerten: endlicher Verlust und Gradienten für alle 32 Parametertensoren.
Kein Trainingsversuch auf dem Nutzerdatensatz/GPU ausgeführt.
