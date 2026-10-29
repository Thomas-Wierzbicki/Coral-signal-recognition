# Neuer synthetischer Pruefsatz fuer die eingefrorene klassische Baseline

Vorab registrierter Plan: `CLASSICAL_HOLDOUT_PLAN_V1.json`.
Die35 Merkmale, Standardisierung und5NN-Trainingsvektoren bleiben unveraendert.
Keine Parametersuche und kein Nachtraining.

Mit aktivierter vorhandener TorchSig-Umgebung aus `~/rf-learning`:

```bash
python project/evaluate_classical_holdout.py
```

Der Lauf sucht genau einen abgeschlossenen
`models/classical-modulation-...`-Ordner. Bei mehreren Treffern:

```bash
python project/evaluate_classical_holdout.py --source models/DEINE-KLASSISCHE-BASELINE
```

Optional `--data data/modulation12-v1`, `--output NEUER-ORDNER`.
Seed und Anzahl sind fest im Plan, keine entsprechenden CLI-Suchparameter.
Ausgabe: `models/classical-holdout-ZEITSTEMPEL/report.json`.
Bitte anschliessend **report.json** hochladen, auch bei einem Fehler.

## Vor dem Erzeugen

Quellbericht, registrierter Baselineplan, Featurecode, Klassifikator und
gespeicherte Merkmalsdateien werden geprueft. Das gespeicherte Modell muss
genau zu den Trainingsmerkmalen und Skalierungswerten passen. Seine vorherigen
Validierungsvorhersagen werden reproduziert. Die SHA256-Werte des Quellberichts,
Klassifikators und der Modellarrays sowie die Code-/Umgebungsversionen stehen
vor der neuen Datenerzeugung im `protocol.json`.

Die urspruengliche Erzeugungsumgebung (TorchSig2.2.0, Torch, NumPy, SciPy)
muss den Datensatzmetadaten entsprechen; NumPy auch dem Klassifikatorbericht.
Keine GPU-Inferenz, aber die vorhandenen Generatorpakete werden benoetigt.

## Neuer Datensatz

- Seed2100, Splitkennung4; getrennte Signal-/Rauschseeds je Klasse/Bedingung.
- 140 Beispiele je Kategorie, insgesamt1680.
- 20 gemeinsame Bedingungsgruppen je SNR:-5,0,5,10,15,20,25dB.
- Dieselben Generatoren und Parameterverteilungen, Blocklaenge4096,32kHz,
  Block-RMS0.5. Kontrolliertes FM wie im Training; alle zwoelf Kategorien.
- Noise hat keinen Nutzsignal-SNR; die gemeinsame Gruppe bleibt erhalten.

Neue Quell-IDs und IQ-Hashes muessen gegen die bisherigen Trainings- und
Validierungs-Parameterdateien disjunkt sein; neue IQ-Bloecke duerfen untereinander
nicht bitidentisch sein. Der alte Testsatz und seine Parameterdatei bleiben zu.
Die neuen IQ-Daten werden gespeichert, zurueckgelesen und gegen ihre Einzelhashes
geprueft. Gezogene FSK-Parameter werden fuer spaetere Diagnose aufgezeichnet,
aber nicht als Klassifikator-Eingaben verwendet.

Erst dann werden unveraendert die35 Merkmale berechnet und mit den gespeicherten
Trainingsmittelwerten/-standardabweichungen skaliert. Der Bericht prueft zuletzt,
dass Klassifikatordatei und in-memory Modellarrays unveraendert sind.

## Interpretation

Primaer Macro-F1 ueber zwoelf Kategorien und Accuracy; dazu Matrix, Klassenrecalls,
FSK-Mittelrecall und getrennte SNR-Auswertungen. Die Differenz zur alten
Entwicklungsvalidierung ist deskriptiv. Keine neue Erfolgsgrenze, keine automatische
Uebernahme als Standardsystem und keine kausale Erklaerung der CNN-Probleme.

Neue Realisierungen bedeuten keine unabhaengigen Generatoren oder realen
Empfangsdaten. Klassen teilen sich Stoerbedingungen; alle1680 Zeilen sind daher
keine unabhaengigen Versuche fuer naive Konfidenzintervalle. Ein neuer Seed,
keine Auswahl des besten Seeds. Ein identischer Wiederholungslauf ist derselbe
Datensatz und darf nicht als neue Evidenz gezaehlt werden. Bereits abgeschlossene
Standardlaeufe mit diesem Plan werden beim CLI-Aufruf abgelehnt.

Vorhandene Dateien werden nicht ueberschrieben. Gesichert werden Protokoll,
`holdout.npz`, Parameter-JSONL, Merkmale/Vorhersagen/Nachbarstimmen und
Artefakthashes im Bericht. `CLASSICAL_HOLDOUT_COMPLETE` entsteht nur bei Erfolg.
Unvollstaendige Fehlerberichte bleiben erhalten; keine automatische Fortsetzung.
Etwaige weitere Modellanpassungen nach Sichtung dieses Satzes machen ihn fuer
diese Anpassungen zur Entwicklungsdatenquelle.

## Tests

```bash
python -m unittest test_classical_holdout test_classical_baseline
```

Die lokalen Tests verwenden Ersatzgeneratoren und synthetische Modell-/Merkmals-
Fixtures. Sie pruefen Einfrieren und Manipulationsabwehr, IQ-Ueberschneidung,
kompletten Daten-/Vorhersageablauf, Dateihashes, Unveraenderlichkeit des Modells
und erhaltene Fehlerberichte. Der echte TorchSig-Prueflauf auf den vorhandenen
Daten muss auf dem Projekt-Rechner ausgefuehrt werden.
