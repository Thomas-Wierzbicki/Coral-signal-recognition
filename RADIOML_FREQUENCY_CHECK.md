# Gegencheck: Frequenzlage aller RadioML-Klassen

## Motivation und festes Protokoll

Reale Aufnahme: unveraendert 10,2 % FM, zentriert 0 %, auf 0,031295 Zyklen/Sample
verschoben 99,3 %. Unveraenderte Referenz reproduziert frueheren Lauf zu 100 %.
Kein Quantisierungs-Clipping. Das belegt Empfindlichkeit in diesem gepaarten
Versuch, keine allgemein verbesserte Erkennung.

Nun dieselben drei festgelegten Varianten auf allen fuenf RadioML-Klassen,
ausschliesslich Trainingssplit, keine Aenderung der Targetfrequenz nach Ergebnis.
Bei vorhandenem Import: 3850 Beispiele x 3 = 11550 Inferenzaufrufe plus Warmup.
Modell unveraendert, kein Resampling, kein neues Training.

## Ausfuehren

```bat
git pull --ff-only
conda activate coral
python check_radioml_frequency.py
```

Standarddaten data/radioml2018-iq-v1/train.npz, Standardmodell unter
models/radioml2018-iq-v1/compiled. Kein Zugriff auf Validierungs-/Testsplits.
Ein Interpreter, ein ungemessenes Warmup; keine Laufzeitvergleiche in diesem Test.
Original verwendet exakt gespeicherte Float-Eingaenge. Verschobene Varianten
verwenden dieselbe Block-RMS-Verarbeitung wie der Real-Frequenztest.
Frequenzschaetzung/Mischung kommt aus sweep_rtl_frequency.py, Ziel fest 0,031295.

Ausgabe models/radioml2018-iq-v1/frequency-check-train-v1:
cases.csv mit source_index, wahrer Klasse, SNR, Schaetzung, Verschiebung,
Vorhersage, allen Bewertungen und Clipping. summary.json enthaelt Matrizen,
Recall, FM-Anteil je wahrer Klasse, SNR-Auswertungen einschliesslich SNR>=10,
gepaarte Klassenwechsel/neue/verlorene FM-Ausgaben sowie Modell-/Trainingshash.
FREQUENCY_CHECK_COMPLETE markiert den kompletten Lauf. Vorhandene Ordner werden
abgelehnt; Wiederholung mit neuem --output moeglich.

## Interpretation

Kernfrage: Werden andere Modulationsarten nach Verschiebung auf die FM-Lage
haeufiger faelschlich als FM ausgegeben? Und verliert echtes RadioML-FM nach
Zentrierung seine FM-Zuordnung? Die gepaarten Aenderungen und die hoch-SNR-
Auswertung helfen, diese Fragen von der Rohgesamtquote zu unterscheiden.

Es sind bekannte Trainingsdaten: Trainings-Uebereinstimmung ist keine neue
Testgenauigkeit. Auch diese Transformation kann mit Quantisierung und anderen
Merkmalen interagieren; aus einer veraenderten Ausgabe folgt keine ausschliessliche
Abhaengigkeit von einem Merkmal. Klassen bleiben fuenf, keine Noise/Unknown-Klasse.
Ein schwach bestimmter Phasenmittelwert ist keine kalibrierte Frequenzmessung.
Die Diagnose soll eine spaetere robuste Trainingsstrategie informieren, nicht
jetzt das Ergebnis durch beliebige Verschiebungen optimieren.

## Tests und Sicherung

```bat
python -m unittest test_radioml_frequency test_rtl_frequency -v
python checkpoint_radioml.py
```

Fuenf Tests bestanden: klassenspezifische FM-Anteile, vollstaendiger dreifacher
Ablauf mit simuliertem Interpreter und ausschliesslich Trainingsdatei, erhaltene
Signalmerkmale bei Verschiebung und gepaarte Uebergaenge. Kein echter Coral hier.
Der Nutzer muss den Hardwarelauf ausfuehren. Daten und Modell bleiben erhalten.
