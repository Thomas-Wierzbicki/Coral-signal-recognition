# Kontrollierter digitaler Robustheitstest der klassischen Baseline

Plan: `CLASSICAL_ROBUSTNESS_PLAN_V1.json`, vor Umsetzung/echtem Lauf registriert.
Die1680 vorhandenen synthetischen IQ-Bloecke und der eingefrorene Klassifikator
bleiben Quelle. Kein Nachtraining, keine neuen Signale, keine Parametersuche.
Alle zwoelf Kategorien bleiben erhalten. CPU und bestehende NumPy-Umgebung reichen.

Aus `~/rf-learning` mit aktivierter Umgebung:

```bash
python project/classical_robustness.py
```

Es wird genau ein abgeschlossener `models/classical-holdout-...` gesucht.
Bei mehreren Treffern `--holdout models/DEIN-HOLDOUT` angeben. Optional
`--data data/modulation12-v1`, `--output NEUER-ORDNER`; bei verschobener
klassischer Baseline `--classical-source models/DEINE-KLASSISCHE-BASELINE`.
Sonst wird deren Pfad aus dem bisherigen Holdout-Protokoll verwendet.

Ausgabe: `models/classical-robustness-ZEITSTEMPEL/report.json`.
Bitte anschliessend **report.json** hochladen. Laufzeit wird je Variante
protokolliert; Abtastrateninterpolation braucht mehr Rechenzeit als Frequenzrotation.
Die15 Varianten sind gepaart, keine25200 unabhaengigen Empfangsbeispiele.

## Festgelegte Varianten

| Faktor | Werte |
|---|---|
| Gemeinsame Kontrolle | kein zusaetzlicher Fehler |
| Zusaetzlicher Frequenzversatz | -3200,-1600,-800,+800,+1600,+3200 Hz |
| Abtastkoordinaten-Schritt | -100,-50,+50,+100 ppm |
| Relativer I/Q-Amplitudenfehler | -1,+1 dB |
| Quadratur-Phasenfehler | -5,+5 Grad |

Immer nur ein Faktor pro Variante. Keine Kombination, kein anschliessendes
Auswaehlen guenstiger Fehlerwerte. Die urspruenglichen SNR-Labels bleiben als
Quellbedingungen erhalten; sie sind keine neu gemessenen SNR nach Transformation.
Noise hat weiterhin keinen Nutzsignal-SNR.

## Gemeinsamer Zuschnitt und Definitionen

Alle Varianten verwenden die mittleren4032 von4096 Samples:32 je Rand entfallen.
Das schafft echte Quellrandsamples fuer die Abtastrateninterpolation. Kein
Null-Padding, periodisches Fortsetzen oder Extrapolieren. Die4032-Sample-Kontrolle
wird gesondert mit der vollstaendigen4096-Sample-Referenz verglichen. Alle
Stoerungsdifferenzen beziehen sich dann auf die identische4032-Sample-Kontrolle.
Die35 festen Merkmale sind laengenabhaengige Statistiken; diese kleine Aenderung
der Fensterlaenge wird deshalb nicht als Stoerwirkung versteckt.

Frequenz: komplexe Rotation bei32000 Samples/s, mit den urspruenglichen
Quellindizes32..4063. Digitale Spektren koennen an Nyquist umlaufen; dies ist
keine Nachbildung eines konkreten analogen Empfaengerfilters.

Abtastrate: Quellposition `center+(n-center)*(1+ppm*1e-6)`.
Positives ppm bedeutet groessere Schritte durch die Quellzeit.33-Tap-Sinc mit
Hammingfenster, pro Position auf Koeffizientensumme1 normiert. Alle benoetigten
Quellsamples muessen vorhanden sein. Die Interpolation ist eine Naeherung und
veraendert auch die Rauschkorrelation; der gemessene Effekt kann daher nicht
allein einem realen Oszillatorfehler zugeschrieben werden.

IQ: `I'=10^(g/40)*I`,
`Q'=10^(-g/40)*(cos(theta)*Q+sin(theta)*I)`.
g ist die relative I/Q-Amplitudendifferenz in dB. g und theta werden getrennt
variiert. Anschliessend fuer alle Varianten dieselbe vorhandene Block-RMS0.5-
Konvention und complex64-Darstellung. Keine Kompensation der relativen IQ-Fehler.

## Integritaet und Ergebnis

Zuerst werden Quellbericht/Code/Plan und Artefakthashes geprueft. Modell/Scaler
muessen zur urspruenglichen Trainingsmerkmalsdatei passen. Die gespeicherten
vollen Holdout-Merkmale und Vorhersagen werden exakt reproduziert. IQ-Hashes,
Quell-IDs, Labels, Bedingungen und SNR werden abgeglichen; alter Testsatz bleibt zu.
Modellarrays werden schreibgeschuetzt. Protokoll vor den Transformationen.

Je Variante Matrix, Accuracy, Macro-F1, Klassenrecalls, FSK-Mittelrecall und
Signal-SNR-Ergebnisse. Dazu gepaarte Vorhersagewechsel, beide richtig,
nur Kontrolle richtig, nur Variante richtig und beide falsch. Keine p-Werte,
Schwellwertsuche, automatische Modelluebernahme oder Hardware-Ursachenbehauptung.

Gespeichert werden gemeinsame IDs/Labels/Quell-SNR, Vorhersagen/Nachbarstimmen
je Variante, Transformationsrezept und SHA256 der transformierten IQ-Matrix.
Keine15 vollstaendigen IQ-Kopien; Varianten sind aus der geprueften Quelle
deterministisch reproduzierbar. Originaldaten und Modell muessen am Ende
unveraendert sein. Erfolgsmarker erst nach allen15 Varianten, sonst Teilbericht.

Dies ist ein digitaler Stresstest desselben synthetischen Datensatzes, kein
Nachweis realer Empfangsrobustheit, Drohnen- oder Coral-Erkennung.

## Lokale Tests

```bash
python -m unittest test_classical_robustness test_classical_baseline
```

Tests umfassen bekannte Tonfrequenzen, Vorzeichen der Zeitstreckung, gleiche
Fensterlaenge, IQ-Definitionen, echte Quellrandsamples, den gesamten15-Varianten-
Ablauf, Quelleinlesen/Reproduktion, Manipulationsabwehr und erhaltene Fehlerberichte.
CPU-Fixtures ersetzen den echten Lauf nicht.
