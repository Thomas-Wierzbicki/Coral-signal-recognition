# RadioML 2018.01A: eigene Entwicklungslinie

Stand: 22. Oktober 2026. Erste Stufe: Datei prüfen und einen kleinen Datensatz
für eine neue Modellvariante vorbereiten.

Die bestehende Drei-Klassen-RMS-Variante bleibt unter `models/signal-rms`.
RadioML-Dateien und importierte Daten werden unter `data/radioml2018` abgelegt.
Es wird noch kein neues Modell trainiert und der Drei-Klassen-Worker ist
nicht mit den neuen Klassen oder Tensorformen kompatibel.

## 1. Originaldaten beschaffen

Offizielle Seite: https://www.deepsig.ai/datasets/

Dort unter **RADIOML 2018.01A** den Download öffnen. Der Downloadanbieter verlangt
derzeit ein Formular mit Kontaktangaben. Der Importer lädt keine Daten automatisch
herunter und überträgt keine Kontaktangaben.

Die HDF5-Datei kann im Downloads-Ordner bleiben; siehe Abschnitt 3.
Zusätzliche Klassen-/Lizenzdateien können beispielsweise hier abgelegt werden:

```text
C:\Users\thwie\source\repos\Coral-signal-recognition\data\radioml2018\raw
```

Typischer Name der HDF5-Datei:

```text
GOLD_XYZ_OSC.0001_1024.hdf5
```

Dateiname und tatsächliche Struktur werden vor dem Import geprüft.
Die Originaldatei ist groß; komprimiertes Archiv und entpackte HDF5-Datei
benötigen zusammen entsprechend freien Speicher.
Die Rohdaten werden nicht auf GitHub hochgeladen.

DeepSig kennzeichnet diese Daten als historisch, nennt bekannte Fehler und
CC BY-NC-SA 4.0. Sie dienen hier dem Experiment und Benchmark.
Reale SDR-Aufnahmen und unsere zusätzlichen Klassen Rauschen/Träger müssen
später gesondert geprüft beziehungsweise ergänzt werden.

## 2. Eigene Importumgebung

In Anaconda Prompt, einmalig:

```text
cd /d C:\Users\thwie\source\repos\Coral-signal-recognition
git pull --ff-only
conda create -n radioml python=3.9 pip -y
conda activate radioml
python -m pip install -r requirements-radioml.txt
```

`coral` und `coral-train` werden für diesen Import nicht verändert.
Die neue Umgebung benötigt zunächst nur NumPy und h5py, kein TensorFlow.

## 3. HDF5 prüfen

Der Import verwendet jetzt standardmäßig die Datei im Downloads-Ordner des
angemeldeten Windows-Benutzers. Auf dem aktuellen Rechner ist das:

```text
C:\Users\thwie\Downloads\GOLD_XYZ_OSC.0001_1024.hdf5
```

Die große Datei muss nicht verschoben oder kopiert werden. Ein explizit
angegebener Dateipfad hat weiterhin Vorrang vor diesem Standard.


```text
python radioml_import.py --inspect
```

Erwartete Struktur:

| Feld | Form | Bedeutung |
|---|---|---|
| X | (N, 1024, 2) | I und Q je Signalabschnitt |
| Y | (N, 24) | One-Hot-Modulationsklasse |
| Z | (N, 1), alternativ (N,) | SNR in dB |

Das Programm liest alle Y-/Z-Zeilen blockweise, prüft endliche Werte und
gültige One-Hot-Labels und zeigt Klassenbelegung und SNR-Stufen an.
Es lädt nicht sämtliche IQ-Daten in den Arbeitsspeicher.
Die IQ-Inhalte der ausgewählten Beispiele werden erst beim eigentlichen Import
auf endliche Werte geprüft.

Zunächst reicht diese Prüfung. Anschließend die Ausgabe und die mitgelieferte
Klassenliste prüfen.

## 4. Klassenreihenfolge ausdrücklich festlegen

Die Namen lassen sich nicht aus den One-Hot-Zahlen von Y ableiten.
Klassenname an Position i einer Liste bedeutet: Y-Spalte i.
Eine alphabetische Sortierung oder ein geratenes Indexschema wäre gefährlich.

Der eigentliche Import verlangt daher `--class-map` mit einer expliziten Liste
aus 24 eindeutigen Namen. Unterstützt werden:

- JSON-Liste von 24 Zeichenfolgen.
- Textdatei mit genau einem Namen je Zeile.
- Als Text gespeicherte Python-Liste von Zeichenfolgen, sicher mit literal_eval gelesen.

Die tatsächlich zur Datei gehörende Reihenfolge muss überprüft werden.
Es gibt im Umlauf Dateien mit Namen wie `classes.txt` und
`classes-fixed.json`; allein der Name beweist keine korrekte Zuordnung.
Der Importer validiert die Struktur der Liste, kann aber aus den Zahlenwerten
keine falsche semantische Beschriftung erkennen.

Mit einer bestätigten JSON-Datei am selben Ort:

```text
python radioml_import.py "C:\Users\thwie\Downloads\GOLD_XYZ_OSC.0001_1024.hdf5" --inspect --class-map data/radioml2018/raw/classes-fixed.json
```

Die AMD-Anleitung verwendet ebenfalls eine ausdrückliche Klassenliste:
https://docs.amd.com/r/en-US/Vitis-Tutorials-AI-Engine-Development/Import-the-Radio-ML-Database?contentId=T7uIW8xfS8zAsJ0XET8O_A

## 5. Kleine Auswahl vorbereiten

Erst nach bestätigter Klassenreihenfolge:

```text
python radioml_import.py "C:\Users\thwie\Downloads\GOLD_XYZ_OSC.0001_1024.hdf5" --class-map data/radioml2018/raw/classes-fixed.json
```

Standardauswahl:

| Neue Klasse | Bedeutung |
|---|---|
| AM-DSB-WC | AM-Doppelseitenband mit Träger |
| FM | FM-Klasse des RadioML-Datensatzes |
| BPSK | Zweiphasenmodulation |
| QPSK | Vierphasenmodulation |
| 16QAM | Quadraturamplitudenmodulation mit 16 Symbolen |

Es wird nicht pauschal behauptet, dass die RadioML-FM-Klasse unseren bisherigen
FM-Testgenerator oder einen bestimmten realen FM-Dienst abbildet.

Standard: SNR 0 bis +20 dB, 100 Beispiele je Klasse und vorhandener SNR-Stufe.
Bei elf SNR-Stufen sind das 5.500 ausgewählte Originalzeilen.
Pro Klasse/SNR werden 70 % Training, 15 % Validierung und der Rest Test.
Bei 100 Beispielen sind das 70/15/15. Die Zeilenindizes werden mitgespeichert.
Die Auswahl erfolgt über Zufallsschlüssel über alle passenden Zeilen;
sie nimmt nicht einfach die ersten Abschnitte aus jedem Block.

Bei zu wenigen Daten in einer Klasse/SNR-Kombination bricht der Import ab.
Es erfolgt keine stille Reduktion einzelner Klassen.

Ausgaben unter `data/radioml2018-v1`:

- `train.npz`, `validation.npz`, `test.npz`.
- Je Datei: x, y, snr_db und source_index.
- `metadata.json`: gewählte Klassen, ursprüngliche Klassenreihenfolge,
  Y-Spalten, SNR-Stufen, Seed und Vorverarbeitung.
- `IMPORT_COMPLETE`: Kennzeichnung, dass alle Ausgaben vollständig geschrieben wurden.

Der Ausgabeordner darf noch nicht existieren. Nach einem abgebrochenen Import
einen neuen Ordner verwenden oder den unvollständigen Ordner bewusst aufräumen.
Ein unvollständiger Import besitzt kein `IMPORT_COMPLETE`.

Beispiel für andere Parameter:

```text
python radioml_import.py "C:\Users\thwie\Downloads\GOLD_XYZ_OSC.0001_1024.hdf5" --class-map data/radioml2018/raw/classes-fixed.json --snr-min -10 --snr-max 20 --per-bucket 200 --output data/radioml2018-v2
```

Mehr Beispiele bedeuten mehr Arbeitsspeicher für die vorbereiteten
Spektrogramme. Das vollständige ursprüngliche IQ-Array wird weiterhin nicht
auf einmal geladen.

## 6. Angepasste Vorverarbeitung

Die 1.024 komplexen Samples werden als ein Originalabschnitt verarbeitet.
Es werden keine unterschiedlichen Abschnitte aneinandergehängt und keine
Samples erfunden, um auf 16.384 zu kommen.

| Eigenschaft | RadioML-Variante |
|---|---|
| IQ-Samples | 1.024 |
| RMS-Normalisierung | Ziel 0,5 für den vollständigen Block |
| FFT | 128 Punkte, Hann-Fenster |
| Hop | 32 Samples |
| Zeitfenster | 29 überlappende Fenster innerhalb desselben Abschnitts |
| Spektrogrammform | (29, 128, 1) |
| Skalierung | -80 bis 0 dB gemäß FFT-Normierung, dann 0 bis 1 |
| Physikalische Abtastrate | Noch nicht gesetzt; keine Übernahme der bisherigen 240 kS/s |

Die Quelle wird mit Dateigröße und Änderungszeit identifiziert.
Die Klassenlistendatei wird per SHA-256 erfasst.
Ein vollständiger Hash der großen HDF5-Datei wird bislang nicht berechnet.

Überlappende FFT-Fenster bleiben innerhalb eines Originalbeispiels.
Training/Validierung/Test enthalten keine gleichen Quellzeilen.
Das beweist keine unabhängigen Aufnahmesitzungen: Entsprechende
Herkunfts-/Gruppenmetadaten liegen diesem Import nicht vor.

Leistungsspektrogramme sind eine zu prüfende erste Modellvariante.
Gerade für PSK/QAM kann direkte IQ-Verarbeitung oder eine zusätzliche
Phasenrepräsentation vorteilhaft sein. Es gibt noch keine Ergebniszusage.

## 7. Prüfstand und nächste Schritte

```text
python -m unittest test_radioml_import -v
```

Drei Tests mit kleinen künstlichen HDF5-Dateien bestanden:
Datenstruktur/Labels, disjunkte stratifizierte Aufteilung und RMS-Spektrogramme.
Diese Testdaten sind keine Modulationsbenchmarks.

Der tatsächliche RadioML-Download, die Prüfung seiner Klassenreihenfolge
und der Import der Originaldaten stehen noch aus.
Nach der Dateiprüfung folgt ein eigener Trainer für die fünf Klassen und
die Eingangsdimension (29,128,1), anschließend Quantisierung/Coral-Prüfung.
`train_signal.py`, `evaluate_coral.py` und `coral_worker.py` sind weiterhin
für die bisherige Drei-Klassen-Variante ausgelegt und dürfen nicht einfach
mit den RadioML-Dateien verwendet werden.
