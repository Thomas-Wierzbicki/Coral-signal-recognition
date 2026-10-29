# Direkter Vergleich: vier gespeicherte CNNs und klassische Baseline

Plan: `CNN_CLASSICAL_HOLDOUT_PLAN_V1.json`, vor CNN-Holdout-Inferenz festgelegt.
Der klassische Holdout-Wert ist bereits bekannt. Deshalb kein vollstaendig
verblindeter neuer Benchmark. Alle vier Modelle werden berichtet.

In der bestehenden Umgebung aus `~/rf-learning`:

```bash
python project/compare_cnn_classical_holdout.py
```

Automatische Quellen: genau ein abgeschlossener
`models/iq-feature-comparison-...`-Lauf und genau ein abgeschlossener
`models/classical-holdout-...`-Lauf. Bei mehreren Treffern:

```bash
python project/compare_cnn_classical_holdout.py --source models/DEIN-CNN-VERGLEICH --holdout models/DEIN-KLASSISCHER-HOLDOUT
```

Optionen: `--data data/modulation12-v1`, `--device cpu` (Standard CUDA),
`--output NEUER-ORDNER`. Wenn die klassische Baseline verschoben wurde,
`--classical-source models/DEINE-KLASSISCHE-BASELINE` verwenden; sonst wird ihr
urspruenglicher Pfad aus dem vorhandenen Holdout-Protokoll gelesen.

Neue Ausgabe: `models/cnn-classical-holdout-ZEITSTEMPEL/report.json`.
Bitte diesen **report.json** hochladen. Keine neuen Signale erzeugen,
keine Optimierung/kein Training, keine Veraenderung bestehender Artefakte.

## Gepruefte Vergleichsgrundlage

Genau dieselben1680 IQ-Bloecke, Quell-IDs und Reihenfolge fuer alle Methoden.
Holdout-Artefakthashes und IQ-Einzelhashes werden geprueft, Label-, Bedingungs-
und SNR-Zuordnung abgeglichen. Die klassischen Merkmale und Vorhersagen werden
aus den gespeicherten IQ-Daten mit unveraendertem Modell reproduziert.

CNN-Quellen: Seeds2081/2082, jeweils `iq_zero` und `iq_envelope_phase`.
Auswahl der besten Checkpoints bleibt die damalige originale Validierungs-CE-
Entscheidung; keine Auswahl nach dem neuen Holdout. Geprueft werden Abschluss,
registrierter Plan, Codehashes, Trainingsauswahl, Checkpoint-Metadaten,
History/Updatebudget, beste Epoche und Modellzustands-Digest.

Vor jeder neuen Vorhersage muss das CNN seine damalige Validierungsmatrix und
seinen CE reproduzieren. Toleranz bleibt atol1e-7/rtol1e-6; kein Aufweichen bei
Fehler. Je Variante die urspruengliche Merkmalsvorverarbeitung, eval/inference
mode und Batch32. Kein Gradient, kein Optimizer. Checkpoint und Zustand muessen
auch danach unveraendert sein. Der alte Testsatz wird nicht geoeffnet.

## Ergebnis

Je CNN: Macro-F1 ueber alle zwoelf Kategorien, Accuracy, Matrix, Klassenrecalls,
FSK-Mittelrecall, Signal-SNR-Auswertungen sowie Differenz zur klassischen Baseline.
Die Paarung zeigt ausserdem: beide richtig, nur klassisch richtig, nur CNN richtig,
beide falsch und Anzahl unterschiedlicher Klassenausgaben. Die vier Modelle
werden nicht als unabhaengige Datensaetze zusammengefasst. Keine p-Werte,
Ensembles, Best-Seed-Auswahl oder automatische Modelluebernahme.

Die Auswertung kontrolliert identische Daten, nicht effektive Kapazitaet oder
Rechenaufwand der Methoden. Ein Vorteil erklaert nicht isoliert die Ursache.
Es bleibt ein synthetischer Satz mit gemeinsamen Stoerbedingungen je Klasse,
keine Aussage ueber reale Empfangs-/Drohnen- oder Coral-Erkennungsleistung.
Historische FP32-Paritaetsgrenzen und Fehlschlaege bleiben unveraendert.

Gespeichert werden Protokoll, alle vier Vorhersagedateien mit Quell-IDs und
Hashes sowie ein gemeinsamer Bericht. Erfolgsmarker erst nach allen vier
reproduzierten Modellen. Bei Fehlern bleibt der Teilbericht erhalten.

## Lokale Tests

```bash
python -m unittest test_cnn_classical_holdout test_iq_feature_comparison
```

Die Tests pruefen Paarzaehlung, echte Checkpoint-Wiederherstellung und
Varianten-Verwechslung, alle vier Inferenzpfade sowie erhaltenen Fehlerbericht
bei nicht reproduzierter Validierung. Kleine CPU-Fixtures ersetzen den echten
GPU-Vergleich nicht.
