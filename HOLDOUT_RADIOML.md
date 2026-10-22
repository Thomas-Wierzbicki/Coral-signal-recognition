# Zusaetzlicher Test mit unbenutzten RadioML-Quellzeilen

## Zweck

Das eingefrorene I/Q-Modell erreichte auf den bisherigen 825 Testzeilen 97,2 %.
Dieser Testsatz war beim Variantenvergleich bereits bekannt. Jetzt werden
neue Zeilen aus derselben HDF5-Datei ausgewaehlt, ohne Training oder Anpassung.
Alle 5500 bisherigen Train/Val/Test-Zeilen des I/Q-Imports werden ausgeschlossen.
Der Spektrogramm-Import nutzte dieselben Zeilen und ist damit ebenfalls abgedeckt.
Wenn weitere Datenvarianten trainiert wurden, deckt dieser Ausschluss diese
NICHT automatisch ab; hier gilt die dokumentierte bisherige Projektlinie.

## 1. Neuen Testsatz importieren (Windows, Anaconda Prompt)

```bat
cd /d C:\Users\thwie\source\repos\Coral-signal-recognition
git pull --ff-only
conda activate radioml
python radioml_holdout_import.py "C:\Users\thwie\Downloads\GOLD_XYZ_OSC.0001_1024.hdf5\GOLD_XYZ_OSC.0001_1024.hdf5"
```

Erwartung: 11000 neue Beispiele, (11000,1,1024,2).
5 Klassen x 11 SNR-Stufen (0..20 dB) x 200 je Klasse/SNR.
Seed 2032; Zufallsschluessel-Auswahl aus allen noch verfuegbaren Zeilen je
Stratum. Alle One-Hot-Labels/SNR werden blockweise geprueft. I/Q-Werte werden
nur fuer die ausgewaehlten Zeilen gelesen und auf endliche Werte geprueft.
Die Dateigroesse wird mit der Referenz verglichen, kein erneuter voller HDF5-Hash.
Es muss dasselbe zuvor gepruefte Original sein.

Ausgabe data/radioml2018-iq-holdout-v1: nur test.npz, metadata.json und
IMPORT_COMPLETE. Kein train.npz/validation.npz: bewusst nicht direkt mit dem
Trainer verwendbar. I/Q-Block-RMS 0,5, Kanaele I/Q, kein Spektrogramm.
Die bisherigen Imports bleiben erhalten. Bestehende Ausgabeordner werden abgelehnt.
Das kompilierte Modell muss bereits am Standardpfad vorhanden sein, weil sein
SHA256 vor der Testauswahl in den Metadaten festgehalten wird.

## 2. Modell unveraendert auf Coral auswerten

```bat
conda activate coral
python evaluate_radioml_iq_coral.py --holdout --data data/radioml2018-iq-holdout-v1 --output models/radioml2018-iq-v1/holdout-v1
```

Keine neue Kompilierung, kein Training. Der Auswerter prueft Modellhash,
Stratifizierung, Klassenfolge, Form, eindeutige Quellzeilen und Disjunktheit
zu den bisherigen Train/Val/Test-Zeilen. Neuer Ergebnisordner ist Pflicht;
alte Resultate werden nicht ueberschrieben. Kein CPU-Vergleich mit den alten
825 Vorhersagen: sie gehoeren zu einem anderen Testsatz.

Ausgabe: coral_report.json und coral_predictions.npz (einschliesslich source_index).
Bericht enthaelt Modellhash, Metadaten, Matrix/Recall, SNR und invoke-Median/p95.
Vorverarbeitung ist nicht Teil der gemessenen invoke-Zeit.

## Einordnung vor dem ersten Ergebnis

Alle ausgewaehlten Zeilen auswerten, auch Fehler; keine gezielte Neuauswahl.
Erstes Ergebnis samt Modellhash festhalten. Ein nachfolgend anhand dieses
Ergebnisses veraendertes Modell macht diesen Satz zum Entwicklungssatz.
Der Ausschluss belegt neue Zeilen, keine unabhaengigen Aufnahmen und keine
Generalisierung auf andere Generatoren, Kanaele oder echte SDR-Aufnahmen.
SNR-Abdeckung bleibt 0..20 dB, und die Klassen bleiben dieselben fuenf.

## Vorherige Sicherung und anschliessende Sicherung

Nutzer hat die lokale Sicherung erfolgreich ausgefuehrt:
coral-radioml-20261022-160900-867963.zip, alle Nutzdateien SHA256-geprueft,
HDF5-Original nicht enthalten. Nach dem neuen Test erneut
python checkpoint_radioml.py ausfuehren, um auch den Holdout und neue Berichte
zu sichern. Das neue Archiv ersetzt das bisherige nicht.

## Lokale Tests

```bat
python -m unittest test_radioml_holdout test_checkpoint_radioml test_radioml_iq test_train_radioml -v
```

Neun Tests bestanden. Neue Tests pruefen stratifizierte wiederholbare Auswahl,
Ausschluss, zu kleine Pools, ueberlappende Zeilen und veraendertes Modell.
Die Auswahl wurde mit synthetischen Label/SNR-Bloecken geprueft. Kein Zugriff
auf die echte HDF5-Datei und keine Coral-Hardware hier; diese beiden Schritte
fuehrt der Nutzer mit obigen Befehlen aus.
