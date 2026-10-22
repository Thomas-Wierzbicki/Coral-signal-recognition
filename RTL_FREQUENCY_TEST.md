# Gepaarter Frequenzlagentest

## Vorab festgelegtes Experiment

Gleiche 1874 reale Bloecke, drei Varianten, gleiches Modell und 32-kS/s-Abtastrate:

1. original: unveraendert.
2. centered: blockweisen mittleren zirkulaeren Phasenschritt entfernen.
3. radioml_fm_offset: denselben Phasenschritt auf fest 0,031295 Zyklen/Sample setzen.

Der Zielwert stammt aus der vorherigen Diagnose der RadioML-FM-Trainingsdaten
mit SNR >=10 dB. Er wird nicht anhand neuer Coral-Ausgaben optimiert. Das sind
normierte Einheiten; RadioML wird keine physikalische Abtastrate zugeschrieben.

## Ausfuehren

```bat
git pull --ff-only
conda activate coral
python sweep_rtl_frequency.py "data\recordings\rtl-20261022-171509-339614\channel-analysis-v1"
```

Keine Neukompilierung und kein Training. Ein Interpreter, gleiche RMS-/Quantisierungs-
verarbeitung wie beim ersten Realtest, je Variante ein ungemessenes Warmup.
Verschobene Varianten werden nur im Arbeitsspeicher erzeugt. Quelle unveraendert.
Neuer Ordner frequency-test-v1: drei CSVs mit allen Entscheidungen/Bewertungen,
shifts.csv mit Frequenzverschiebung pro Block und summary.json mit Modell-/Kanalhash,
Parametern, Klassenanteilen, Clipping und gepaarten Uebergangsmatrizen.
FREQUENCY_TEST_COMPLETE markiert den vollstaendigen Lauf. Vorhandene Ordner
werden abgelehnt; fuer Wiederholung --output verwenden.

Wenn der fruehere Realtest coral-real-v1 vorliegt, werden Kanalhash, Modellhash,
Start und Blockzahl geprueft und alle original-Vorhersagen mit diesem verglichen.
Eine Abweichung wird berichtet. Vor Interpretation verschobener Varianten muss
die unveraenderte Referenz reproduzierbar sein.

## Interpretation

Die Mischung ist z[n] * exp(j*2*pi*(Ziel-Schaetzung)*n), n beginnt pro Block bei 0.
Amplituden und zeitliche Aenderungen der Phasenschritte bleiben erhalten, nur der
konstante Phasenanstieg aendert sich. Die blockweise Schaetzung enthaelt unter
Umstaenden auch langsame Modulation; sie ist keine kalibrierte Senderkorrektur.
Lokale Oszillatorphase wird je Block zurueckgesetzt, da jeder Block einzeln
klassifiziert wird. Keine kontinuierliche korrigierte Empfangsdatei entsteht.
Auch Quantisierung der gedrehten I/Q-Koordinaten kann Entscheidungen beeinflussen;
Clipping und Scores werden deshalb fuer jede Variante gespeichert.

Ein hoeherer FM-Anteil nach Verschiebung zeigt Empfindlichkeit gegenueber dieser
Transformation, noch keinen alleinigen Ursachenbeweis und keine allgemein
verbesserte Erkennung. Ein unveraendertes Ergebnis widerlegt nicht andere
Domaenenunterschiede. Keine zusaetzliche Bandbreiten-/Zeit-Anpassung in diesem
Experiment. Die Aufnahme ist schon bekannt; kein unabhaengiger Test.
Keine gesicherte Ground Truth je Block, keine Unknown-/Noise-Klasse.

## Tests und Status

```bat
python -m unittest test_rtl_frequency test_rtl_classify test_rtl_domain -v
```

Acht Tests bestanden. Tests fuer Frequenzziel, Huellkurvenerhalt, erhaltene
Phasenvariationen, Uebergangsmatrix, drei vollstaendige Varianten mit simuliertem
Interpreter und bestehende Diagnose/Klassifikation. Echter Coral-Lauf steht aus.
Anschliessend Ergebnisse mit checkpoint_radioml.py sichern.

## Gemeldetes Hardwareergebnis

Original 192/1874 FM (10,2 %), centered 0/1874 FM (344 BPSK, 1530 QPSK),
radioml_fm_offset 1860/1874 FM (99,3 %), 14 QPSK. Original stimmt mit frueherem
Lauf zu 100 % ueberein. Klassenwechsel 535/1874 bzw. 1668/1874.
Kein Quantisierungs-Clipping. Naechster Gegencheck:
[Alle RadioML-Trainingsklassen mit denselben Frequenzvarianten](RADIOML_FREQUENCY_CHECK.md).
