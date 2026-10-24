# Diagnose der FSK-Familie

## Ausgangspunkt

Der erste GPU-Lauf modulation12-v1 erreichte 56,0 % und Macro-F1 0,531, beste Epoche 25. Bei hohem SNR bleiben besonders GFSK/GMSK schlecht. Frequenzverschiebungen um -/+1600 Hz ergeben 53,1/51,6 % und 68,3/66,7 % Vorhersageübereinstimmung. Diese Ergebnisse sind Entwicklungsbefunde, keine unabhängige Abschlussbewertung.

## Gesicherter Generatorbefund

Der geprüfte installierte TorchSig-2.2.0-Quellcode zieht für FSK h ungefähr zwischen 0,7 und 1,01 (zusätzlich häufig exakt 1), für GFSK h aus [0,1;0,5), für GMSK exakt 0,5. Beide Gaußvarianten verwenden BT zwischen 0,1 und 0,5 und Filterhalbspannen von ein bis vier Symbolen. h ist der Modulationsindex, nicht der Frequenzhub in Hz.

Die Baseband-Funktion erzeugt bei gleichem h=0,5, BT=0,3, Filterspan und Symbolzufall für GFSK und GMSK bitidentische Signale. Das ist ein Grenzfall: Der normale GFSK-Zufallszug erreicht 0,5 nicht exakt. Es beweist keine doppelten Beispiele im Trainingsdatensatz und erklärt nicht allein dessen Fehler.

## Ausführen

```bash
cd ~/rf-learning
source .venv/bin/activate
git -C project pull --ff-only
python project/diagnose_fsk_separability.py --reference-model models/modulation12-v1/model_traced.pt
```

Der neue Ausgabeordner ist `data/fsk-separability-v1`. Bestehende Ordner werden abgewiesen. Das Zwölfklassen-Modell bleibt unverändert.

Pro Klasse entstehen 120 neue Beispiele: 80 für einen einfachen Merkmalsvergleich, 40 für dessen separaten Prüfsatz. Bedingungen: 32 kHz, 4096 Samples, angeforderte Bandbreite 6 kHz, SNR 30 dB, Frequenzversatz/Phase null, Block-RMS 0,5. Die zufälligen internen Generatorparameter bleiben erhalten und werden je Beispiel protokolliert. Gleiche angeforderte Bandbreite bedeutet ausdrücklich nicht gleiche Symbolrate oder gleichen Frequenzhub.

Der Vergleich nutzt zentrierte Phasenschritte (Streuung, Quantile, Autokorrelation) und Hüllkurvenvariation. Standardisierung wird nur auf den 80 Trainingsbeispielen je Klasse angepasst; ein 5-Nächste-Nachbarn-Klassifikator wertet die 40 übrigen Beispiele aus. Das eingefrorene Referenzmodell wird auf genau diesen 120 Prüfbeispielen mit seinen unveränderten zwölf Ausgaben ausgewertet; seine Matrix hat deshalb drei Zeilen und zwölf Spalten. Dieser Vergleich isoliert weder allein die Architektur noch allein das Eingabeformat.

## Lokal geprüfter Befund

Vollständiger CPU-Lauf mit TorchSig 2.2.0: Merkmalsvergleich 107/120 = 89,2 %. Zeilen/Spalten FSK, GFSK, GMSK:

```
40  0  0
 0 30 10
 0  3 37
```

Das belegt nutzbare Unterschiede unter diesen erleichterten synthetischen Bedingungen. Es ist kein Nachweis guter Erkennung über alle Bandbreiten/SNR oder auf realen Aufnahmen. Das trainierte GPU-Referenzmodell war hier nicht vorhanden; dessen Ergebnis liefert der obige Lauf beim Nutzer. Zwei Prüftests kontrollieren Matrixorientierung und Frequenzversatz-Invarianz der Merkmalsberechnung im Bereich ohne Phasenfaltung.

Ausgaben: `report.json`, `parameters.json`, `examples.npz`, `phase_steps.png`, Abschlussmarker `DIAGNOSIS_COMPLETE`. Quellcode- und optional Modellhash sind im Bericht enthalten.

## Gemeldeter Modellvergleich und korrigierte Einordnung

Das eingefrorene Zwölfklassen-Modell erreichte auf diesen 120 Prüfbeispielen 5,0 %. Der Merkmalsklassifikator wurde dagegen speziell auf drei Klassen unter festen Bedingungen trainiert. Auch liegt der Diagnose-SNR von 30 dB außerhalb des bisherigen Trainingsrasters (-5 bis +25 dB). Der Abstand 89,2 % gegen 5,0 % isoliert deshalb keinen Architektur- oder Darstellungsfehler. Das Pooling als alleinige Ursache ist nicht nachgewiesen.

Der nächste Entwicklungsversuch behält alle zwölf Klassen, dieselben Daten und dieselbe I/Q-Darstellung. Er vergleicht das bisherige Netz mit einer dokumentierten Portierung einer veröffentlichten Architektur über drei Seeds: [MODULATION12_REFERENCE.md](MODULATION12_REFERENCE.md). Die Ergebnisse bleiben Entwicklungsbefunde; reale Modulationserkennung und spätere Drohnenprotokollerkennung sind weiterhin nicht validiert.
