# Klassische Kontrollbaseline fuer alle zwoelf Kategorien

Der Plan `CLASSICAL_BASELINE_PLAN_V1.json` wurde vor Implementierung und echtem
Datenlauf registriert. Ein fester Satz von35 Signalmerkmalen wird mit einem
5-Nachbarn-Klassifikator verwendet. Kein CNN, keine GPU und keine neuen Pakete.
Die bestehende NumPy-Umgebung reicht.

In Ubuntu mit aktivierter Umgebung aus `~/rf-learning`:

```bash
python project/classical_modulation_baseline.py
```

Der Lauf sucht genau einen abgeschlossenen urspruenglichen
`models/se-gain-development-...`-Ordner. Bei mehreren Treffern:

```bash
python project/classical_modulation_baseline.py --source models/DEIN-DEVELOPMENT-ORDNER
```

Quelle der IQ-Daten: standardmaessig `data/modulation12-v1`, optional `--data`.
Neue Ausgabe: `models/classical-modulation-ZEITSTEMPEL`; optional `--output`
auf einen noch nicht existierenden Ordner. Bitte danach **report.json** hochladen.

## Vergleich und Grenzen

Die1680 Trainingsbeispiele und1080 Validierungsbeispiele sind dieselben wie
in den vorangegangenen CNN-Vergleichen. Die gespeicherte Auswahl und Daten
werden per SHA256 geprueft. Jede Trainingsbedingung enthaelt alle zwoelf Klassen;
je SNR sind20 Bedingungen enthalten. Training und Validierung muessen nach
Quell-IDs und Bedingungen getrennt sein. Der Testsatz wird nicht geoeffnet.

Die35 Merkmale umfassen zentrierte Phasendifferenzen, deren Quantile und
Autokorrelation, relative Huelle, komplexe Lag-Kohaerenz sowie Spektralbreite,
Entropie und Flatness. Keine internen Generatorparameter, Frequenzmittelwerte,
FFT-Peakpositionen oder absoluten Empfangspegel als Eingabemerkmale.
Die genaue Reihenfolge wird in `feature_names` gespeichert.

Merkmalsmittelwerte und Standardabweichungen werden ausschliesslich am
Trainingssatz berechnet. Die Validierung dient nicht zur Wahl von Merkmalen,
Nachbarzahl, Skalierung oder sonstigen Parametern. Feste euklidische Distanz,
fuenf gleich gewichtete Nachbarn. Stabile Distanzsortierung; bei Gleichstand
der Klassenstimmen gewinnt der kleinere Klassenindex. Stimmen sind keine
kalibrierten Wahrscheinlichkeiten.

Der Bericht zeigt Matrix, Accuracy, Macro-F1 ueber alle zwoelf Klassen,
Klassenrecalls, FSK-Mittelrecall und Signal-SNR-Auswertungen. Das Trainingsresultat
ist Leave-one-out: der jeweilige eigene Datenpunkt wird ausgeschlossen. Es bleibt
eine deskriptive Trainingsauswertung, keine unabhaengige Guetemessung.

Gesichert werden Merkmale/IDs, Trainingslabels, Skalierungswerte, geordnete
skalierte Trainingsmerkmale und Vorhersagen mit Nachbarstimmen. Der gespeicherte
Klassifikator wird neu geladen; seine Validierungsvorhersagen muessen exakt passen.
Alle Artefakthashes stehen im Bericht. `CLASSICAL_BASELINE_COMPLETE` entsteht
erst nach erfolgreichem Abschluss. Fehlerberichte bleiben erhalten, bestehende
Modelle und Daten unveraendert.

Die Methode ist unabhaengig vom bisherigen CNN, die Daten sind es nicht.
Es bleibt dieselbe wiederholt verwendete synthetische Entwicklungsvalidierung.
Keine angeglichene effektive Modellkapazitaet oder Rechenbudget-Gleichheit
mit CNNs. Schlechte Ergebnisse beweisen keine Untrennbarkeit der Daten;
gute Ergebnisse belegen keine reale Empfangs-/Drohnen- oder Coral-Erkennung.
Phasenhauptwerte koennen an pi springen; keine Behauptung exakter Offset-Invarianz.
Alle zwoelf Kategorien bleiben erhalten, auch Noise als eigene bekannte Kategorie.

## Lokale Tests

```bash
python -m unittest test_classical_baseline
```

Tests pruefen bekannte Signalmerkmale, Skalierung, Selbstnachbar-Ausschluss,
Stimmengleichstaende, gespeicherten Klassifikator, Trainings-only-Skalierung,
Artefakthashes und erhaltene Fehlerberichte. Der echte Datenlauf steht aus.
