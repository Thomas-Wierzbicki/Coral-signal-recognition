# Coral-signal-recognition

## Aktuell: kontrollierter Zwölfklassen-Vergleich (24.10.2026)

Das erste TorchSig-basierte Zwölfklassen-Modell erreichte 56,0 % auf dem synthetischen Entwicklungsprüfsatz. Als nächster Schritt werden das bisherige Netz und eine dokumentierte PyTorch-Portierung des veröffentlichten Harper-2023-Modells bei gleichen Daten und Trainingsregeln über drei Seeds verglichen. **Alle zwölf Klassen bleiben erhalten.** Start und Aussagegrenzen: [MODULATION12_REFERENCE.md](MODULATION12_REFERENCE.md). Die 89,2 % des separaten Dreiklassen-Merkmalstests sind kein gleichwertiger Architekturvergleich mit dem Zwölfklassen-Netz. Reale Erkennung und Drohnenprotokolle sind weiterhin nicht nachgewiesen.


Experimentelle Klassifikation synthetischer HF-Signale mit dem Google Coral USB Accelerator.

**Stand: 22. Oktober 2026 – RadioML-I/Q: 98,1 % auf 11.000 neuen Testzeilen.**

| Variante | Klassen | Coral-Test | invoke Median / p95 |
|---|---:|---:|---:|
| Eigener Generator, RMS-Spektrogramm | noise, carrier, fm | 98,7 % (592/600) | 0,464 / 0,655 ms |
| RadioML-Spektrogramm | AM-DSB-WC, FM, BPSK, QPSK, 16QAM | CPU-INT8 71,6 %; kein Coral-Test dokumentiert | – |
| RadioML direkte I/Q, RMS – ursprünglicher Test | dieselben fünf Modulationen | 97,2 % (802/825) | 0,457 / 0,811 ms |
| **RadioML I/Q – neuer Holdout, Modell unverändert** | dieselben fünf Modulationen | **98,1 % (10.789/11.000)** | **0,424 / 0,514 ms** |

Beim RadioML-I/Q-Test stimmen **alle 825 Coral-Vorhersagen mit CPU-INT8 überein**.
Alle zwölf Modelloperationen sind auf einem Edge-TPU-Subgraphen abgebildet.
invoke-Zeiten enthalten Python/USB, aber keine Vorverarbeitung oder Quantisierung.
Die verschiedenen Drei-/Fünfklassenaufgaben sind kein direkter Genauigkeitsvergleich.

**Aktuelle Dokumente:**

- [RadioML-I/Q: Import, Training, Compiler und Hardwaretest](RADIOML_IQ.md)
- [Bestätigte Ergebnisse, Matrizen, SNR und Grenzen](RESULTS_RADIOML.md)
- [Sichern, Prüfen und Wiederherstellen](BACKUP_RADIOML.md)
- [Zusätzlicher Test mit unbenutzten Quellzeilen](HOLDOUT_RADIOML.md)
- [Echte I/Q-Aufnahme mit RTL-SDR Blog V3 vorbereiten](RTL_CAPTURE.md)
- [RadioML-Originalimport](RADIOML.md) und [Klassenherkunft](RADIOML_CLASS_ORDER.md)

**Aktuellen lokalen Stand sichern (Anaconda Prompt im Projektordner):**

```bat
git pull --ff-only
conda activate coral
python checkpoint_radioml.py
```

Code und Dokumentation liegen auf GitHub. Trainierte Modelle und NPZ-Daten bleiben
auf dem Windows-Rechner; das Skript sichert sie dort als neue geprüfte ZIP.
Die Sicherung wurde am 22.10.2026 erfolgreich lokal ausgeführt und SHA256-geprüft:
`coral-radioml-20261022-160900-867963.zip`. Es erfolgte kein Upload.
Die Original-HDF5 unter Downloads gehört nur mit `--include-source` zum Archiv.

Der frühere persistente RMS-Prozesstest wurde ebenfalls ausgeführt:
29/30 Treffer, Anfrage-Median 6,63 ms. Dieser Drei-Klassen-Prozess ist noch
keine Schnittstelle für das neue Fünf-Klassen-I/Q-Modell.

Keine SDR-Anbindung und keine Bewertung echter Empfangssignale.
Der zusätzliche Testsatz mit ausgeschlossenen bisherigen Quellzeilen ist abgeschlossen.
Nächster Schritt: echte RTL-SDR-Aufnahmen erfassen und deren Kanal/Skalierung prüfen.
Aufnahmewerkzeuge sind vorbereitet, echte Hardwareaufnahme noch nicht durchgeführt.
Die bestätigte Sicherung liegt vor dem Holdout; den erweiterten Stand erneut sichern.
Der folgende ausführliche Verlauf beschreibt die ursprüngliche Drei-Klassen-Linie.
Ältere Aussagen zu damals ausstehenden Schritten sind historisch einzuordnen.

## Inhaltsverzeichnis

1. [Aufgabenteilung und Voraussetzungen](#1-aufgabenteilung-und-voraussetzungen)
2. [Coral und Python unter Windows einrichten](#2-coral-und-python-unter-windows-einrichten)
3. [Repository auf den Rechner holen](#3-repository-auf-den-rechner-holen)
4. [Gerät und Laufzeit prüfen](#4-gerät-und-laufzeit-prüfen)
5. [Erster Bildtest: Vogelarten erkennen](#5-erster-bildtest-vogelarten-erkennen)
6. [Synthetische Signaldaten erzeugen](#6-synthetische-signaldaten-erzeugen)
7. [CNN trainieren und quantisieren](#7-cnn-trainieren-und-quantisieren)
8. [Compiler unter Ubuntu/WSL bereitstellen](#8-compiler-unter-ubuntuwsl-bereitstellen)
9. [Funkmodell für die Edge TPU kompilieren](#9-funkmodell-für-die-edge-tpu-kompilieren)
10. [Funkmodell auf dem Coral testen](#10-funkmodell-auf-dem-coral-testen)
11. [Ergebnisse und ihre Aussagekraft](#11-ergebnisse-und-ihre-aussagekraft)
12. [Dateien, Sicherung und Wiederholung](#12-dateien-sicherung-und-wiederholung)
13. [Fehler und Warnmeldungen](#13-fehler-und-warnmeldungen)
14. [Nächste Entwicklungsschritte](#14-nächste-entwicklungsschritte)
15. [Referenzen](#15-referenzen)
16. [Persistenter Coral-Prozess und Schnittstellentest](#16-persistenter-coral-prozess-und-schnittstellentest)
17. [Gezielte Fehlerdiagnose nach Signalparametern](#17-gezielte-fehlerdiagnose-nach-signalparametern)
18. [Kontrollierter Pegeltest](#18-kontrollierter-pegeltest)
19. [Zweite Modellvariante mit RMS-Normalisierung](#19-zweite-modellvariante-mit-rms-normalisierung)
20. [Validierter RMS-Stand, Sicherung und Prozesstest](#20-validierter-rms-stand-sicherung-und-prozesstest)

## 1. Aufgabenteilung und Voraussetzungen

### Zwei Konsolen und zwei Python-Umgebungen

| Aufgabe | Wo ausführen? | Umgebung |
|---|---|---|
| Coral-Gerätetest, Bilderkennung, Signaldaten erzeugen, Coral-Auswertung | **Anaconda Prompt unter Windows** | `coral` |
| Modell auf der CPU trainieren und TFLite exportieren | **Anaconda Prompt unter Windows** | `coral-train` |
| TFLite-Modell für den Coral kompilieren | **Ubuntu-Terminal unter WSL** | Linux-Compiler, kein Conda erforderlich |

**Wichtig:** Python-Befehle hier in Anaconda Prompt ausführen. In der normalen
PowerShell war die passende Python-Installation nicht erreichbar.
Die Umgebung wird mit `conda activate ...` gewählt; ihr Name erscheint links
in der Eingabezeile.

Der Coral bleibt für die Inferenz an Windows angeschlossen. Für die Kompilierung
unter WSL braucht er nicht an Ubuntu durchgereicht zu werden.

### Im Projekt erfolgreich verwendeter Stand

| Bestandteil | Version beziehungsweise Einstellung |
|---|---|
| Hardware | Google Coral USB Accelerator, Modell G950-01456-01 |
| Windows-Python in Conda | Python 3.9.25 |
| PyCoral | 2.0.0 |
| tflite-runtime | 2.5.0.post1 |
| NumPy für `coral` | 1.26.4 gemäß `requirements.txt` |
| Pillow für `coral` | mindestens 10, kleiner als 12 |
| TensorFlow für `coral-train` | 2.10.1 |
| NumPy für `coral-train` | 1.23.5 |
| protobuf für `coral-train` | 3.19.6 |
| Edge-TPU-Laufzeit unter Windows | `RuntimeVersion(14)`, Build vom 20. Juli 2021 |
| Compiler unter Ubuntu | 14.1.317412892 |
| Ubuntu | 24.04 / noble, Architektur `x86_64` |

Diese Tabelle dokumentiert den verwendeten Projektstand, keine allgemeine
Kompatibilitätszusage für beliebige andere Versionen.
Python 3.14 war zunächst vorhanden; für das verwendete PyCoral-Paket wurde
zusätzlich Python 3.9 in einer eigenen Conda-Umgebung eingerichtet.

## 2. Coral und Python unter Windows einrichten

### 2.1 Hardware anschließen

1. Coral möglichst direkt an einen USB-3-Anschluss des Rechners anschließen.
2. Ein USB-Datenkabel verwenden.
3. Im Geräte-Manager nach **Coral USB Accelerator** suchen.
4. Unter Eigenschaften → Allgemein den Gerätestatus prüfen.

Im Projekt leuchtete die LED und Windows spielte beim Anschluss einen USB-Ton.
Der Eintrag im Geräte-Manager und später der erfolgreiche Modelltest bestätigten
die Verbindung.

Die **Edge-TPU-Laufzeit für Windows** muss ebenfalls installiert sein.
Bei einer Neuinstallation die Windows-Schritte der
[offiziellen USB-Accelerator-Anleitung](https://coral.ai/docs/accelerator/get-started/)
verwenden. Die Laufzeit ist eine separate Installation; `pip install pycoral`
ersetzt sie nicht. Auf dem hier getesteten Rechner war die Laufzeit bereits
funktionsfähig; ihre ursprüngliche Installation wurde in diesem Projektablauf
nicht separat protokolliert.

### 2.2 Conda-Umgebung erstellen

In **Anaconda Prompt**, einmalig:

```text
conda create -n coral python=3.9 pip -y
conda activate coral
python --version
```

Im erfolgreichen Ablauf wurde `Python 3.9.25` ausgegeben.
Wenn die Umgebung bereits existiert, nur `conda activate coral` ausführen.

### 2.3 PyCoral installieren

```text
python -m pip install --extra-index-url https://google-coral.github.io/py-repo/ "pycoral~=2.0"
python -m pip install "numpy==1.26.4" "Pillow>=10,<12"
```

Dabei wurden PyCoral 2.0.0 und tflite-runtime 2.5.0.post1 installiert.
Der erste Installationslauf wählte automatisch NumPy 2.0.2. Für die ältere
Coral-Laufzeit wurde NumPy anschließend auf 1.x festgelegt.

**TensorFlow nicht in dieser Umgebung installieren.** Das Training verwendet
die separate Umgebung `coral-train`.

## 3. Repository auf den Rechner holen

Das eigenständige Projekt liegt unter:

https://github.com/Thomas-Wierzbicki/Coral-signal-recognition

Es ist ein separates Repository neben `Intel_RTL_Analyzer_Next`.
Der zuvor dort angelegte Branch `feature/coral-signal-recognition` wird für
die hier beschriebenen Schritte nicht benötigt.

### Erstmaliges Klonen

In **Anaconda Prompt**:

```text
cd /d C:\Users\thwie\source\repos
git clone https://github.com/Thomas-Wierzbicki/Coral-signal-recognition.git
cd Coral-signal-recognition
conda activate coral
python -m pip install -r requirements.txt
```

### Bereits vorhandene Kopie aktualisieren

```text
cd /d C:\Users\thwie\source\repos\Coral-signal-recognition
git pull --ff-only
```

Nicht erneut klonen, wenn dieser Projektordner bereits existiert.
Bei lokalen Änderungen, die ein Update blockieren, zuerst `git status` prüfen.

Der Projektordner lautet auf dem Windows-Rechner:

```text
C:\Users\thwie\source\repos\Coral-signal-recognition
```

Alle folgenden Python-Skripte werden aus diesem Ordner gestartet.
Bei einem anderen Benutzernamen oder Speicherort die Pfade entsprechend anpassen.

## 4. Gerät und Laufzeit prüfen

In **Anaconda Prompt**, Umgebung `coral`:

```text
conda activate coral
python -c "from pycoral.utils.edgetpu import list_edge_tpus; print(list_edge_tpus())"
```

Erfolgreiche Ausgabe auf dem getesteten Rechner:

```text
[{'type': 'usb', 'path': '/sys/bus/usb/devices/22-4'}]
```

Dieser von PyCoral gemeldete Gerätepfad erschien auch unter Windows.
Er bedeutet in diesem Fall nicht, dass die Berechnung unter WSL lief.
Der genaue Pfad kann auf anderen Rechnern abweichen.

Laufzeit prüfen:

```text
python -c "from pycoral.utils.edgetpu import get_runtime_version; print(get_runtime_version())"
```

Beobachtetes Ergebnis:

```text
BuildLabel(COMPILER=MSVC 192528612,DATE=Jul 20 2021,TIME=14:31:41), RuntimeVersion(14)
```

### Wiederholter Hardwaretest

```text
python coral_check.py
```

Das Skript lädt bei Bedarf ein offizielles Vogelmodell nach
`models/bird_edgetpu.tflite`, öffnet eine Verbindung, führt einen Aufwärmlauf
und anschließend standardmäßig 100 Berechnungen mit Null-Eingabedaten aus.

Es prüft die Modellausführung, keine sinnvolle Bildklassifikation.
Das Vogelmodell erkennt keine Funksignale.

Im ersten manuellen Test wurden ebenfalls 100 Berechnungen erfolgreich
ausgeführt. Dabei waren zwei USB-Tonfolgen hörbar. Die genaue Ursache wurde
nicht bestimmt; der Test zeigte keinen Berechnungsabbruch.

## 5. Erster Bildtest: Vogelarten erkennen

Dieser optionale Test wurde vor dem eigenen Signalmodell erfolgreich ausgeführt.
Das Modell unterscheidet Vogelarten, keine allgemeinen Gegenstände.

In **Anaconda Prompt**, Umgebung `coral`, Projektordner:

```text
python -c "import urllib.request; b='https://raw.githubusercontent.com/google-coral/test_data/master/'; urllib.request.urlretrieve(b+'mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite','coral_test.tflite'); urllib.request.urlretrieve(b+'parrot.jpg','parrot.jpg'); urllib.request.urlretrieve(b+'inat_bird_labels.txt','inat_bird_labels.txt'); print('Downloads fertig')"
```

Bild anzeigen:

```text
python -c "from PIL import Image; Image.open('parrot.jpg').show()"
```

Drei beste Ergebnisse berechnen:

```text
python -c "from PIL import Image; from pycoral.utils.edgetpu import make_interpreter; from pycoral.utils.dataset import read_label_file; from pycoral.adapters import common, classify; i=make_interpreter('coral_test.tflite'); i.allocate_tensors(); img=Image.open('parrot.jpg').convert('RGB').resize(common.input_size(i), Image.Resampling.LANCZOS); common.set_input(i,img); i.invoke(); labels=read_label_file('inat_bird_labels.txt'); print('\n'.join('{}: {:.1f} Prozent'.format(labels.get(c.id,c.id),100*c.score) for c in classify.get_classes(i,top_k=3)))"
```

Beobachtetes Ergebnis:

| Modellklasse | Modellbewertung |
|---|---:|
| Ara macao / Scarlet Macaw / Scharlachara | 75,8 % |
| Platycercus elegans / Crimson Rosella / Pennantsittich | 7,4 % |
| Coracias caudatus / Lilac-breasted Roller / Gabelracke | 1,6 % |

Die Werte sind Modellbewertungen, keine garantierten Wahrscheinlichkeiten.
Für ein eigenes Vogelfoto den Bildpfad `parrot.jpg` im Erkennungsbefehl ersetzen.

Diese optionalen Downloads liegen bei den obigen Befehlen im Projektstamm.
Sie werden für das Funkmodell nicht gebraucht und sollen nicht versehentlich
als zusätzliche Projektdateien committed werden.

## 6. Synthetische Signaldaten erzeugen

### 6.1 Klassen und Signalparameter

| Index | Kennung | Bedeutung |
|---|---|---|
| 0 | `noise` | Komplexes gaußverteiltes Rauschen ohne Nutzsignal |
| 1 | `carrier` | Unmodulierter komplexer Träger mit Rauschen |
| 2 | `fm` | Mit einem Sinuston frequenzmodulierter Träger mit Rauschen |

Der Generator variiert:

| Parameter | Bereich |
|---|---|
| Frequenzversatz | −60 bis +60 kHz |
| Signalamplitude | 0,2 bis 0,9 |
| nominelles SNR für Träger/FM | −5 bis +25 dB |
| FM-Frequenzhub | 1,5 bis 6 kHz |
| FM-Modulationston | 300 bis 3.000 Hz |
| Phase | zufällig |

Bei der Klasse `noise` wird nur der nach derselben Regel skalierte
Rauschanteil erzeugt. Ihr gezogener SNR-Parameter beschreibt kein tatsächlich
vorhandenes Nutzsignal.

### 6.2 Vorverarbeitung

| Eigenschaft | Wert |
|---|---|
| IQ-Abtastrate | 240.000 Samples/s |
| IQ-Samples je Beispiel | 16.384 |
| Zeitdauer je Beispiel | etwa 68,27 ms |
| FFT-Länge | 256 |
| FFT-Fenster | Hann |
| Aufeinanderfolgende Zeitblöcke | 64, ohne Überlappung |
| Frequenzanordnung | `fftshift`, negative bis positive Basisbandfrequenzen |
| Frequenzabstand je FFT-Bin | 937,5 Hz |
| numerischer Pegelbereich | −80 bis 0 dBFS gemäß Generator-Normierung |
| Eingangswerte | auf 0 bis 1 skaliert und begrenzt |
| Form eines Beispiels | `(64, 256, 1)`, Zeit × Frequenz × Kanal |
| Datentyp vor Quantisierung | `float32` |

Die dBFS-Skalierung ist eine rechnerische Referenz im Generator.
Sie ist keine kalibrierte Leistungsmessung in dBm an einem realen Empfänger.
Es findet keine individuelle Helligkeitsnormalisierung jedes Beispiels statt.

### 6.3 Erster kleiner Datensatz

```text
conda activate coral
python signal_lab.py
```

Dieser erste Lauf erzeugte:

| Aufteilung | Je Klasse | Gesamt |
|---|---:|---:|
| Training | 200 | 600 |
| Validierung | 40 | 120 |
| Test | 40 | 120 |

### 6.4 Größerer Datensatz für den erfolgreichen Coral-Test

```text
python signal_lab.py --per-class 1000
```

Ergebnis:

| Aufteilung | Je Klasse | Gesamt |
|---|---:|---:|
| Training | 1.000 | 3.000 |
| Validierung | 200 | 600 |
| Test | 200 | 600 |

Ausgaben unter `data`:

- `train.npz`, `validation.npz`, `test.npz`: Arrays `x` und `y`.
- `metadata.json`: Klassen, Abtastrate, FFT-Größe, Skalierung und Seed.
- Neun PNG-Vorschauen: je Klasse und Aufteilung ein Beispiel.

PNG-Dateien dienen der Ansicht. Das Modell arbeitet mit numerischen
Spektrogrammen aus den NPZ-Dateien.

**Erneutes Generieren überschreibt diese Dateien.** Der Standard-Seed ist 2026;
Training, Validierung und Test verwenden getrennte Zufallsströme.
Ein geänderter Datenumfang verändert auch Teile der erzeugten Aufteilungen.
Die kleinen und großen Trainingsläufe wurden daher nicht auf einem identischen
festgehaltenen Testdatensatz verglichen.

Für einen neuen Ausgabeordner ist beispielsweise möglich:

```text
python signal_lab.py --per-class 1000 --seed 2027 --output data-neu
```

Dann beim Training auch `--data data-neu` angeben.
Der zusätzliche Ordner `data-neu` wird nicht von der vorhandenen
`data/`-Regel in `.gitignore` abgedeckt.

## 7. CNN trainieren und quantisieren

### 7.1 Separate Trainingsumgebung erstellen

Einmalig in **Anaconda Prompt**:

```text
conda create -n coral-train python=3.9 pip -y
conda activate coral-train
python -m pip install -r requirements-training.txt
```

Die Datei legt TensorFlow 2.10.1, NumPy 1.23.5 und protobuf 3.19.6 fest.
Wenn die Umgebung schon vorhanden und eingerichtet ist, nur aktivieren.

### 7.2 Training starten

```text
cd /d C:\Users\thwie\source\repos\Coral-signal-recognition
conda activate coral-train
python train_signal.py
```

Das Training findet auf dem Rechner statt, nicht auf dem Coral.
Der Coral beschleunigt später die Ausführung des trainierten Modells.

Standardwerte:

- Maximal 30 Epochen, Batchgröße 32, Seed 2026.
- Optimierer Adam; Sparse Categorical Crossentropy.
- EarlyStopping anhand von `val_loss`, Geduld sechs Epochen.
- Die Gewichte mit dem besten Validierungsverlust werden wiederhergestellt.

Netzstruktur:

1. Eingang `64 × 256 × 1`.
2. Conv2D, acht Filter, Kernel 3, Schrittweite 2, ReLU.
3. MaxPooling2D, Größe 2.
4. Conv2D, 16 Filter, Kernel 3, Schrittweite 2, ReLU.
5. MaxPooling2D, Größe 2.
6. Flatten.
7. Dense mit 24 Einheiten und ReLU.
8. Dense mit drei Ausgaben und Softmax.

### 7.3 Quantisierung und Auswertung

Das Skript exportiert ein vollständig ganzzahlig quantisiertes TFLite-Modell
mit `uint8`-Ein- und -Ausgängen. Zur Kalibrierung verwendet es bis zu
100 Trainingsbeispiele je Klasse. Testdaten werden nicht zur Kalibrierung
oder zur Wahl des EarlyStopping-Zeitpunkts verwendet.

Anschließend prüft es Float- und INT8-Modell auf dem Testdatensatz.

Erzeugte Dateien:

| Datei unter `models/signal` | Inhalt |
|---|---|
| `signal_float.h5` | Trainiertes Keras-Modell ohne gespeicherten Optimierer |
| `signal_int8.tflite` | Quantisiertes CPU-TFLite-Modell, noch nicht für die TPU kompiliert |
| `labels.txt` | Klassenindex und Klassenname |
| `report.json` | Genauigkeit, Recall, Konfusionsmatrizen, Vorhersageübereinstimmung und Metadaten |

Das Skript überschreibt diese Ausgabedateien beim nächsten Lauf.

### 7.4 Beobachtete Warnungen

Während des erfolgreichen Ablaufs erschienen:

```text
Found untraced functions such as _jit_compiled_convolution_op ...
Statistics for quantized inputs were expected, but not specified; continuing anyway.
fully_quantize: 0, ... input_inference_type: UINT8, output_inference_type: UINT8
```

Diese Meldungen haben in diesem konkreten Lauf den Export nicht verhindert.
Das exportierte Modell wurde anschließend auf der CPU ausgeführt, erfolgreich
für die TPU kompiliert und auf dem Coral ausgewertet.

Die einzelne Textzeile `fully_quantize: 0` ist daher hier kein Beleg für einen
fehlgeschlagenen INT8-Export. Maßgeblich sind die konfigurierten
Ganzzahloperationen, die geprüften Tensoren sowie Compiler- und Laufzeitergebnis.
Bei einem Abbruch oder einer anderen Fehlermeldung den vollständigen Fehler prüfen.

## 8. Compiler unter Ubuntu/WSL bereitstellen

### 8.1 Ubuntu und Architektur prüfen

Unter Windows kann PowerShell vorhandene WSL-Systeme auflisten:

```powershell
wsl --list --verbose
```

Ubuntu über das Startmenü öffnen. Im **Ubuntu-Terminal**:

```bash
uname -m
```

Im erfolgreichen Ablauf:

```text
x86_64
```

Der zunächst getestete Befehl

```bash
edgetpu_compiler --version
```

lieferte `command not found`.

### 8.2 Paketquelle lieferte HTTP 403

Die offizielle APT-Quelle unter

```text
https://packages.cloud.google.com/apt coral-edgetpu-stable main
```

lieferte bei der Einrichtung am 21. Oktober 2026:

```text
403 Forbidden
The repository ... is not signed.
```

Das Abrufen der Metadaten war blockiert. Die folgende Signaturmeldung ließ
sich durch das erneute Hinzufügen des Schlüssels nicht sinnvoll beheben.
Die Signaturprüfung wurde nicht abgeschaltet.

Wenn diese Quelle bereits wie im Projekt eingerichtet wurde, sie deaktivieren:

```bash
sudo mv /etc/apt/sources.list.d/coral-edgetpu.list /etc/apt/sources.list.d/coral-edgetpu.list.disabled
sudo apt-get update
```

Wenn die Datei schon deaktiviert ist oder nie angelegt wurde, den
`mv`-Befehl überspringen. Andere Paketquellen nicht verändern.

### 8.3 Erfolgreicher Ersatz: Compiler aus dem offiziellen GitHub-Repository

Im **Ubuntu-Terminal**:

```bash
cd ~
git clone --depth 1 https://github.com/google-coral/edgetpu.git coral-tools
~/coral-tools/compiler/x86_64/edgetpu_compiler --version
```

Falls Git fehlt, zuerst:

```bash
sudo apt-get install -y git
```

Wenn `~/coral-tools` bereits existiert und der Compiler funktioniert, nicht
erneut klonen.

Erfolgreich getestete Ausgabe:

```text
Edge TPU Compiler version 14.1.317412892
```

Der verwendete Startwrapper liegt unter `compiler/x86_64/edgetpu_compiler`.
Er startet den Compiler mit den im Repository mitgelieferten Bibliotheken.
Deshalb den Ordner zusammenlassen und den vollständigen Wrapperpfad verwenden.

Dies ist eine ältere Compiler-Version. Sie hat das hier verwendete Modell
erfolgreich verarbeitet. Daraus folgt keine Kompatibilität mit beliebigen
neueren TensorFlow-Modellen oder Operatorversionen.
Der Klonbefehl verwendet den jeweils verfügbaren Repository-Stand.
Für eine spätere Reproduktion den lokalen Commit zusätzlich festhalten:

```bash
git -C ~/coral-tools rev-parse HEAD
```

## 9. Funkmodell für die Edge TPU kompilieren

Weiter im **Ubuntu-Terminal**:

```bash
cd /mnt/c/Users/thwie/source/repos/Coral-signal-recognition/models/signal
mkdir -p compiled
~/coral-tools/compiler/x86_64/edgetpu_compiler -s -o compiled signal_int8.tflite
```

WSL greift über `/mnt/c` auf Laufwerk C: zu. Es wird dieselbe Datei gelesen,
die zuvor unter Windows exportiert wurde; eine Kopie nach Linux ist nicht nötig.

Die Ausgabe liegt danach auch unter Windows im Projektordner:

```text
C:\Users\thwie\source\repos\Coral-signal-recognition\models\signal\compiled\signal_int8_edgetpu.tflite
```

Zusätzlich entsteht `signal_int8_edgetpu.log`.

### Beobachtetes Compiler-Ergebnis

| Kennzahl | Ergebnis |
|---|---:|
| Kompilierdauer | 41 ms |
| Eingabemodell | 30,22 KiB |
| Ausgabemodell | 84,56 KiB |
| On-Chip-Cache für Modellparameter | 74,25 KiB |
| Verbleibender Parametercache | 7,71 MiB |
| Off-Chip-Streaming ungecachter Parameter | 0 Byte |
| Edge-TPU-Teilgraphen | 1 |
| Operationen insgesamt | 10 |

| Operator | Anzahl | Zuordnung |
|---|---:|---|
| FULLY_CONNECTED | 2 | Edge TPU |
| RESHAPE | 1 | Edge TPU |
| MAX_POOL_2D | 2 | Edge TPU |
| QUANTIZE | 2 | Edge TPU |
| SOFTMAX | 1 | Edge TPU |
| CONV_2D | 2 | Edge TPU |

Alle zehn Operationen wurden auf die Edge TPU abgebildet.
Daten laden, Spektrogramme erzeugen, Eingaben vorbereiten und Ergebnisse
auswerten bleiben Aufgaben des Host-Rechners.

**Nach jedem erneuten Training muss erneut kompiliert werden.**
Ein vorhandenes kompiliertes Modell aktualisiert sich nicht automatisch.

## 10. Funkmodell auf dem Coral testen

Zurück in **Anaconda Prompt unter Windows**, nicht im Ubuntu-Terminal:

```text
conda activate coral
cd /d C:\Users\thwie\source\repos\Coral-signal-recognition
git pull --ff-only
python evaluate_coral.py
```

Standardmäßig liest das Programm:

- `data/test.npz`.
- `data/metadata.json`.
- `models/signal/compiled/signal_int8_edgetpu.tflite`.

Es öffnet den Interpreter einmal, führt einen Aufwärmlauf aus und wertet
anschließend alle Testbeispiele über dieselbe Verbindung aus.
Die Eingaben werden mit Skala und Nullpunkt des Modells nach `uint8` quantisiert.

Andere Pfade können explizit angegeben werden:

```text
python evaluate_coral.py --model models/signal/compiled/signal_int8_edgetpu.tflite --data data
```

Es ist keine erneute Installation von TensorFlow in `coral` erforderlich.

### Erfolgreiche Ausgabe

```text
Coral-Geraete: [{'type': 'usb', 'path': '/sys/bus/usb/devices/22-4'}]
100/600 berechnet
200/600 berechnet
300/600 berechnet
400/600 berechnet
500/600 berechnet
600/600 berechnet

Coral-Testgenauigkeit: 91.5%
Konfusionsmatrix: Zeilen wahr, Spalten vorhergesagt
[[200   0   0]
 [  0 169  31]
 [  0  20 180]]
Klassenreihenfolge: ['noise', 'carrier', 'fm']
noise: 200/200 richtig
carrier: 169/200 richtig
fm: 180/200 richtig
Inferenz inklusive Python/USB, ohne Vorverarbeitung:
Median 0.427 ms; p95 0.543 ms
Nur synthetische Testdaten; keine Bewertung echter Empfangssignale.
```

## 11. Ergebnisse und ihre Aussagekraft

### Entwicklung der Testgenauigkeit

| Lauf | Trainingsbeispiele | Testbeispiele | Float / CPU | INT8 / CPU | INT8 / Coral |
|---|---:|---:|---:|---:|---:|
| Kleiner Startdatensatz | 600 | 120 | 77,5 % | 80,0 % | nicht getestet |
| Größerer Datensatz | 3.000 | 600 | 91,3 % | 91,5 % | 91,5 % |

Die Genauigkeit stieg im beobachteten größeren Lauf deutlich.
Da sich auch der Testdatensatz änderte, ist dies kein kontrollierter Vergleich
ausschließlich des Einflusses der Trainingsmenge.

### Konfusionsmatrix des größeren Coral-Tests

Zeilen = tatsächliche Klasse, Spalten = vorhergesagte Klasse:

| Tatsächlich / vorhergesagt | Rauschen | Träger | FM |
|---|---:|---:|---:|
| Rauschen | 200 | 0 | 0 |
| Träger | 0 | 169 | 31 |
| FM | 0 | 20 | 180 |

| Klasse | Richtig | Recall |
|---|---:|---:|
| Rauschen | 200 von 200 | 100 % |
| Träger | 169 von 200 | 84,5 % |
| FM | 180 von 200 | 90 % |
| Gesamt | 549 von 600 | 91,5 % Genauigkeit |

CPU-INT8 und Coral lieferten dieselbe aggregierte Konfusionsmatrix.
Die aktuellen Programme vergleichen nicht jede einzelne CPU-Vorhersage direkt
mit der zugehörigen Coral-Vorhersage. Eine identische Matrix beweist deshalb
keine vollständige Identität aller Einzelentscheidungen.

### Laufzeit richtig lesen

- Median: **0,427 ms**.
- 95. Perzentil: **0,543 ms**.
- Gemessen wird der Aufruf von `interpreter.invoke()` nach dem Aufwärmen.
- Python-Aufruf und USB-Kommunikation während dieses Aufrufs sind enthalten.
- Nicht enthalten: Daten laden, IQ-Erfassung, FFT/Spektrogramm, Quantisierung,
  `set_tensor()`, Ergebnisabholung und grafische Anzeige.
- Es wurde keine vergleichbare CPU-Laufzeit gemessen; ein Beschleunigungsfaktor
  gegenüber der CPU ist daher noch nicht bestimmt.

Ein Beispiel repräsentiert etwa 68,27 ms IQ-Daten.
Die kurze Modelllaufzeit allein ist keine Zusage für die Gesamtlatenz eines
späteren Echtzeitsystems.

### Grenzen

Die Daten sind synthetisch. Es fehlen bislang reale Empfangsbedingungen wie
abweichende Empfänger, Filter, AGC, Frequenzfehler, Übersteuerung, mehrere
gleichzeitige Signale und andere Modulationsarten.
Das Modell hat keine eigene Klasse für unbekannte Signale.
Eine hohe Softmax-Bewertung ist kein verlässlicher Nachweis, dass ein Eingang
zu einer der drei bekannten Klassen gehört.

Wiederholtes Optimieren anhand desselben Testsatzes verringert dessen Wert als
unabhängige Prüfung. Für belastbare Aussagen neue, getrennte Aufnahmesitzungen
und einen zurückgehaltenen abschließenden Testsatz verwenden.

## 12. Dateien, Sicherung und Wiederholung

### Projektdateien auf GitHub

| Datei | Zweck |
|---|---|
| `README.md` | Vollständiger Ablauf und dokumentierte Ergebnisse |
| `TRAINING.md` | Ergänzende Anleitung für Training und Export |
| `requirements.txt` | NumPy/Pillow für die Coral-Umgebung; PyCoral separat |
| `requirements-training.txt` | Festgelegte Trainingsabhängigkeiten |
| `signal_lab.py` | Synthetische IQ-Daten und Spektrogramme erzeugen |
| `train_signal.py` | CNN trainieren, quantisieren und auf CPU auswerten |
| `coral_check.py` | Hardwaretest mit offiziellem Vogelmodell |
| `evaluate_coral.py` | Eigenes kompiliertes Signalmodell auf Coral auswerten |
| `.gitignore` | Generierte Daten, Modelle und Python-Cache ausschließen |

### Lokale Ergebnisse

**`data/` und `models/` werden durch `.gitignore` ausgeschlossen.**
Ein normaler Commit oder Push sichert diese Ordner nicht auf GitHub.
Nach einem neuen Klonen fehlen deshalb die lokal trainierten Modelle und Daten.

Für den exakt erreichten Stand gemeinsam sichern:

1. Den vollständigen Ordner `data`.
2. Den vollständigen Ordner `models/signal`, einschließlich `compiled`,
   `report.json` und Compilerlog.
3. Die verwendete Git-Revision sowie die Conda-Paketlisten.
4. Bei Bedarf den Compilerordner `~/coral-tools` oder dessen Git-Revision.

Die Paketstände lassen sich in Anaconda Prompt anzeigen:

```text
conda list -n coral
conda list -n coral-train
git rev-parse HEAD
```

Der aktuelle Coral-Auswerter schreibt seine Ergebnisse nur auf die Konsole.
Für eine zusätzliche lokale Ergebnisdatei:

```text
python evaluate_coral.py > models\signal\compiled\coral_evaluation.txt 2>&1
```

Während dieses Befehls ist die laufende Ausgabe in die Datei umgeleitet.
Nach Beendigung kann sie geöffnet werden.

Zufallsseeds erleichtern die Wiederholung, garantieren aber keine bitgenau
identischen Trainingsgewichte auf unterschiedlichen Rechnern oder
Bibliotheksständen. Für exakt dasselbe Modell die erzeugten Dateien aufbewahren.

### Kurzablauf: nur das fertige Modell erneut testen

In Anaconda Prompt, sofern Daten und kompiliertes Modell noch vorhanden sind:

```text
conda activate coral
cd /d C:\Users\thwie\source\repos\Coral-signal-recognition
python evaluate_coral.py
```

### Kurzablauf: vollständig neu erzeugen

**Dieser Ablauf überschreibt die generierten Daten und Modellausgaben.**
Zuerst gegebenenfalls sichern.

Anaconda Prompt:

```text
cd /d C:\Users\thwie\source\repos\Coral-signal-recognition
conda activate coral
python signal_lab.py --per-class 1000
conda activate coral-train
python train_signal.py
```

Ubuntu:

```bash
cd /mnt/c/Users/thwie/source/repos/Coral-signal-recognition/models/signal
mkdir -p compiled
~/coral-tools/compiler/x86_64/edgetpu_compiler -s -o compiled signal_int8.tflite
```

Nach erfolgreicher Kompilierung zurück in Anaconda Prompt:

```text
conda activate coral
python evaluate_coral.py
```

Nur das zu diesem Trainingslauf gehörende, frisch kompilierte Modell auswerten.

## 13. Fehler und Warnmeldungen

| Beobachtung | Vorgehen |
|---|---|
| `python` in PowerShell nicht gefunden | Anaconda Prompt öffnen und passende Umgebung aktivieren |
| `No module named pycoral` | `conda activate coral`, PyCoral-Installation und `python --version` prüfen |
| NumPy-2-/ABI-Importfehler | In `coral`: `python -m pip install -r requirements.txt` |
| `TensorFlow fehlt` | `coral-train` aktivieren und Trainingsabhängigkeiten installieren |
| `list_edge_tpus()` liefert `[]` | Gerät, Datenkabel, USB-Anschluss, Geräte-Manager und Windows-Laufzeit prüfen |
| Delegate/Laufzeit lässt sich nicht laden | Windows-Edge-TPU-Laufzeit prüfen; kompletten Fehlertext auswerten |
| Compiler in Ubuntu nicht gefunden | Den vollständigen Wrapperpfad unter `~/coral-tools` verwenden |
| Google-APT-Quelle liefert 403 | Quelle wie oben deaktivieren und den erfolgreich getesteten GitHub-Compiler verwenden |
| `signal_int8.tflite` fehlt | Training vollständig ausführen und Arbeitsordner prüfen |
| Kompiliertes Modell fehlt | Ubuntu-Kompilierung ausführen; Standardpfad des Auswerters prüfen |
| Nach Training unerwartete alte Ergebnisse | Prüfen, ob das neue TFLite-Modell tatsächlich erneut kompiliert wurde |
| USB-An-/Abmeldetöne | Einzelne Tonfolgen wurden bei erfolgreichen Tests beobachtet; bei Wiederholungen während einer Dauerauswertung oder Fehlern USB-Verbindung genauer prüfen |
| Compiler meldet nicht unterstützte Operatorversion | Konkreten Compilerbericht prüfen; eine ältere Version unterstützt nicht jede neuere Exportvariante |

Das Auftreten eines USB-Tons allein wurde nicht als Hardwaredefekt nachgewiesen.
Umgekehrt beweist eine leuchtende LED allein noch keine funktionierende Inferenz.

## 14. Nächste Entwicklungsschritte

1. Trainings- und Auswertungsstand einschließlich Modellartefakten sichern.
2. Verwechslungen zwischen Träger und FM systematisch nach SNR und Parametern
   untersuchen; unabhängige Daten für weitere Vergleiche festhalten.
3. Echte SDR-Aufnahmen sammeln und getrennt nach Aufnahmesitzungen auswerten.
4. Abtastrate, Frequenzausschnitt und Vorverarbeitung des Qt-Analyzers an den
   Modelleingang angleichen.
5. Den bereits implementierten und auf Hardware getesteten persistenten
   Coral-Prozess mit dem Qt-Analyzer verbinden.
6. Ergebnisse und eine passende Behandlung unbekannter Signale im Analyzer anzeigen.

Die Qt-Anbindung ist bisher **geplant, nicht implementiert**.
Der Coral übernimmt die neuronale Klassifikation; FFT, Wasserfall und klassische
Signalverarbeitung des Analyzers werden dadurch nicht automatisch beschleunigt.

## 15. Referenzen

- [Projekt](https://github.com/Thomas-Wierzbicki/Coral-signal-recognition)
- [Bestehender SDR-Analyzer](https://github.com/Thomas-Wierzbicki/Intel_RTL_Analyzer_Next)
- [Coral USB Accelerator: Einrichtung](https://coral.ai/docs/accelerator/get-started/)
- [PyCoral](https://github.com/google-coral/pycoral)
- [Offizielle Coral-Testmodelle und Bilder](https://github.com/google-coral/test_data)
- [Verwendeter Compiler-Ordner bei Google Coral](https://github.com/google-coral/edgetpu/tree/master/compiler/x86_64)
- [Edge-TPU-Compiler](https://coral.ai/docs/edgetpu/compiler/)
- [Modellanforderungen der Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)
- [TensorFlow-Lite-Quantisierung](https://www.tensorflow.org/lite/performance/post_training_quantization)

Die numerischen Ergebnisse in dieser README stammen aus den am 21. Oktober 2026
vom Projektinhaber ausgeführten und im Projektverlauf protokollierten Läufen.

## 16. Persistenter Coral-Prozess und Schnittstellentest

Nach dem erfolgreichen 600-Beispiele-Hardwaretest wurde eine Prozessschnittstelle
als Vorbereitung der Qt-Anbindung ergänzt. Der eigentliche Analyzer ist noch
nicht angeschlossen.

### Test unter Windows

In Anaconda Prompt:

```text
conda activate coral
cd /d C:\Users\thwie\source\repos\Coral-signal-recognition
git pull --ff-only
python live_demo.py
```

Das bereits kompilierte Modell muss unter
`models/signal/compiled/signal_int8_edgetpu.tflite` vorhanden sein.
Erneutes Training oder Kompilieren ist für diesen Schnittstellentest nicht nötig.

`live_demo.py` startet `coral_worker.py` mit dem Python-Interpreter der aktiven
Umgebung. Das Modell wird einmal geladen und aufgewärmt. Anschließend werden
je zehn neue synthetische IQ-Blöcke der Klassen noise, carrier und fm geschickt.
Jeder Block ist unabhängig erzeugt; dies ist kein kontinuierlicher Empfangsstrom.
Der Standard-Seed 2027 unterscheidet sich vom bisherigen Generator-Standard 2026.

Ausgegeben werden Sollklasse, erkannte Klasse, Modellbewertung, Dauer des
TPU-Aufrufs und Dauer der vollständigen Anfrage. Eine abschließende Matrix
zeigt die Verwechslungen. Keine bestimmte Trefferzahl wird vorausgesetzt.
Dieser kleine Test prüft die Schnittstelle und ist kein unabhängiger
Genauigkeitsnachweis.

Längerer Test:

```text
python live_demo.py --per-class 100 --interval 0.1
```

Abbruch mit Strg+C. Der Testclient beendet seinen Unterprozess beim normalen
Ende sowie bei Abbruch und Fehlern.

### Protokoll für einen späteren Qt-QProcess-Client

Kein Netzwerkport wird geöffnet. Kommunikation erfolgt über die Standardeingabe
und Standardausgabe des Unterprozesses. Jede Nachricht ist ein JSON-Objekt,
abgeschlossen durch einen Zeilenumbruch. Diagnoseausgaben gehören auf stderr.

Startbefehl des Dienstes:

```text
python -u coral_worker.py
```

Der Dienst meldet nach erfolgreicher Initialisierung `type: ready`,
`protocol: 1`, `sample_rate: 240000`, `samples: 16384` und die drei Klassen.
Beim manuellen Start wartet er danach auf Aufträge. Für den normalen Test
stattdessen `live_demo.py` verwenden.

Eine Klassifikationsanfrage enthält:

| Feld | Inhalt |
|---|---|
| `type` | `classify` |
| `id` | Ganzzahl oder Zeichenfolge zur Zuordnung der Antwort |
| `sample_rate` | Genau 240000 |
| `iq_base64` | Base64-kodierte 16.384 komplexe float32-Samples, Little Endian |

Jedes Sample besteht aus Realteil I und Imaginärteil Q, jeweils float32.
Die Nutzdaten entsprechen NumPy-Datentyp `<c8`, insgesamt 131.072 Byte vor
Base64-Kodierung. Es werden keine Sollklassen übertragen.

Der Worker führt dieselbe Spektrogrammvorverarbeitung wie der Generator aus.
Andere Abtastraten, falsche Blocklängen und nichtendliche Werte werden abgelehnt.
Er führt noch kein Resampling aus.

Eine erfolgreiche Antwort enthält:

| Feld | Inhalt |
|---|---|
| `type` | `result` |
| `id` | Auftragskennung |
| `label` | `noise`, `carrier` oder `fm` |
| `scores` | Dequantisierte Modellbewertungen für alle drei Klassen |
| `invoke_ms` | Nur Interpreter-Aufruf, einschließlich darin anfallender Python-/USB-Kosten |
| `processing_ms` | Vorverarbeitung, Quantisierung und Auswertung im Worker; ohne Prozessübertragung |

Die Modellbewertungen sind keine garantierten Wahrscheinlichkeiten.
Der Client misst zusätzlich die Anfragedauer einschließlich Prozesskommunikation,
Vorverarbeitung und Antwort; die vorherige Signalerzeugung/Base64-Kodierung ist
darin nicht enthalten.

Fehlerantwort: `type: error`, `id` soweit zuordenbar und `message`.
Normale fehlerhafte Aufträge beenden den Dienst nicht.
Überlange Nachrichten über 256 KiB führen zum Beenden.
`type: shutdown` wird mit `type: bye` beantwortet; anschließend endet der Prozess.
Auch das Schließen seiner Eingabe beendet ihn.

Der Testclient hält jeweils nur eine Anfrage offen. Das verhindert einen
wachsenden Rückstau im Test. Ein späterer Qt-Client muss ebenfalls begrenzen,
wie viele noch nicht beantwortete Signalblöcke er vorhält.

### Automatische Tests ohne Hardware

```text
python -m unittest test_coral_worker -v
```

Die Tests verwenden einen künstlichen Interpreter, keine Ersatz-Erkennung.
Geprüft werden IQ-Kodierung, Ablehnung ungültiger Daten, Quantisierung,
Antwortzuordnung, Fehlerbehandlung und Wiederverwendung eines Interpreters.

Diese Tests wurden in der Entwicklungsumgebung erfolgreich ausgeführt.
Auch der Prozessablauf wurde inzwischen auf dem echten Coral unter Windows
erfolgreich ausgeführt: 30/30 Aufträge verarbeitet, 23 richtige Klassifikationen.
Die 600-Beispiele-Ergebnisse stammen weiterhin aus `evaluate_coral.py`;
sie sind ein anderer Lauf als der Schnittstellentest.

Zusätzliche Projektdateien:

- `coral_worker.py`: persistenter Inferenzprozess.
- `live_demo.py`: synthetischer Prozess-/Schnittstellentest.
- `test_coral_worker.py`: Tests ohne Hardware.

### Beobachteter Prozesstest auf dem echten Coral

Am 21. Oktober 2026 wurde `python live_demo.py` in der Windows-Umgebung
`coral` mit den Standardparametern ausgeführt. Der Dienst meldete sich bereit;
alle 30 Anfragen wurden beantwortet und der Test lief bis zur Abschlussmatrix.

| Eigenschaft | Ergebnis |
|---|---|
| Testart | Synthetische IQ-Blöcke über einen persistenten Unterprozess |
| Modelleinsatz | Einmal initialisiert, für sämtliche Aufträge wiederverwendet |
| Beispiele | Je zehn Rauschen, Träger und FM; insgesamt 30 |
| Verarbeitete Aufträge | 30 von 30 |
| Richtige Klassifikationen | 23 von 30, entsprechend 76,7 % |
| Median der vollständigen Anfrage | 4,61 ms |
| Beobachtete TPU-Aufrufzeiten | 0,421 bis 0,615 ms |

Konfusionsmatrix, Zeilen = Sollklasse, Spalten = Vorhersage:

| Tatsächlich / vorhergesagt | Rauschen | Träger | FM |
|---|---:|---:|---:|
| Rauschen | 10 | 0 | 0 |
| Träger | 0 | 5 | 5 |
| FM | 0 | 2 | 8 |

Damit wurden Rauschen in 10/10 Fällen, Träger in 5/10 Fällen und FM in 8/10
Fällen richtig erkannt. Eine fehlerfreie Prozesskommunikation bedeutet nicht,
dass jede Klassifikation richtig sein muss.

Auffällige Einzelentscheidungen:

- Ein tatsächlicher Träger wurde als FM mit **92,6 % Modellbewertung** ausgegeben.
- Ein tatsächliches FM-Signal wurde als Träger mit **84,4 % Modellbewertung**
  ausgegeben.

Die Werte zeigen, dass auch hohe Modellbewertungen bei falschen Entscheidungen
vorkommen. Eine einfache Schwelle auf der höchsten Bewertung ist deshalb allein
kein nachgewiesener Schutz vor solchen Verwechslungen.

#### Vergleich mit dem vorherigen Test

| Merkmal | `evaluate_coral.py` | `live_demo.py` |
|---|---|---|
| Daten | Gespeicherte 600 Testbeispiele | 30 neu erzeugte synthetische Blöcke |
| Generator-Seed | 2026 für den ursprünglichen Datensatz | 2027 |
| Kommunikation | Auswertung im selben Python-Prozess | Aufträge und Antworten über Prozess-Pipes |
| Trefferquote | 91,5 % | 76,7 % |
| Gemeldeter Median | 0,427 ms für `invoke()` | 4,61 ms für die vollständige Anfrage |

Die Laufzeitwerte messen unterschiedliche Teile des Ablaufs und dürfen nicht
als direkter Leistungsvergleich der TPU interpretiert werden.
Im Prozesstest benötigte der TPU-Aufruf meist etwa 0,5 ms; die vollständige
Anfrage einschließlich Übertragung und Vorverarbeitung dauerte meist etwa
4,4 bis 5,5 ms. Die erste Anfrage dauerte 10,71 ms.
Die Differenz lässt sich mit den vorhandenen Messungen nicht weiter in einzelne
Kosten für FFT, JSON, Base64, Pipes und Scheduling zerlegen.

Die kleinere Trefferquote stammt aus anderen Beispielen mit einem anderen Seed
und einer sehr kleinen Stichprobe. Sie belegt keinen gesicherten Rückgang der
Modellqualität gegenüber dem 600-Beispiele-Lauf. Sie bestätigt jedoch erneut
die bereits bekannte Schwäche bei der Unterscheidung von Träger und FM.

#### Nächster fachlicher Schritt

Vor der Qt-Anbindung sollen Fehlentscheidungen nach SNR, Frequenzversatz,
Signalamplitude und FM-Parametern untersucht werden. Dafür steht inzwischen `diagnose_signal.py` bereit: Es protokolliert diese
Parameter je Beispiel, ohne das bestehende Modell neu zu trainieren. Siehe Abschnitt 17.

Bis dahin ist die Prozessschnittstelle funktionsfähig getestet. Die
Zuverlässigkeit unter echten Empfangsbedingungen bleibt ungeprüft.

### Vor der tatsächlichen Analyzer-Anbindung

Die IQ-Daten des Analyzers müssen zuerst auf 240 kS/s gebracht und zu Blöcken
mit 16.384 Samples zusammengesetzt werden. Die bisherige FFT-/Wasserfallanzeige
ist kein direkter Ersatz für diesen Eingang. Ein unpassender Ausschnitt oder
eine andere Pegelskalierung kann die Erkennung trotz funktionierender
Prozessschnittstelle verschlechtern.

## 17. Gezielte Fehlerdiagnose nach Signalparametern

Nach dem Prozesstest mit 23/30 Treffern wurde eine Diagnose des vorhandenen
Modells ergänzt. Sie soll die Bedingungen der Verwechslungen zwischen Träger
und FM sichtbar machen. Eine Verbesserung des Modells ist damit noch nicht
implementiert.

### Ausführen

In Anaconda Prompt unter Windows:

```text
conda activate coral
cd /d C:\Users\thwie\source\repos\Coral-signal-recognition
git pull --ff-only
python diagnose_signal.py
```

Das vorhandene kompilierte Modell wird einmal geladen und unverändert verwendet.
Training, Datensatzerzeugung mit `signal_lab.py` und Kompilierung müssen nicht
wiederholt werden. Die Diagnose überschreibt keine Trainings- oder Testdaten.

Geprüft werden jeweils 50 neue Beispiele je Klasse und je SNR-Stufe:
**−5, 0, 5, 10, 15, 20 und 25 dB**. Insgesamt sind das **1.050 Beispiele**.
Frequenzversatz, Signalamplitude und FM-Parameter variieren weiterhin zufällig.
Standard-Seed: 2028.

Die Konsole zeigt für jede SNR-Stufe die Trefferquote je Klasse, anschließend
die Gesamtmatrix und die Anzahl falscher Entscheidungen mit mindestens
80 % Modellbewertung. Diese Grenze ist eine Diagnoseauswahl, keine validierte
Annahmeschwelle für ein späteres Produkt.

Für Rauschen ist die SNR-Angabe nur der Parameter der Rauschskalierung.
Ohne Nutzsignal gibt es kein Nutzsignal-Rausch-Verhältnis im üblichen Sinn.

### Ergebnisdateien

Unter `models/signal/diagnostics` entstehen:

| Datei | Inhalt |
|---|---|
| `cases.csv` | Eine Zeile je Beispiel mit Soll-/Vorhersageklasse, Modellbewertungen und Signalparametern |
| `summary.json` | Gesamtergebnis, Gruppen nach SNR/Pegel/Frequenzversatz und FM-Modulationsindex, hoch bewertete Fehler, Seed und SHA-256 des verwendeten Modells |

Die CSV enthält Frequenzversatz in Hz, Amplitude, SNR, bei FM Frequenzhub,
Modulationston und Modulationsindex sowie die gemessene Inferenzdauer.
Die FM-Parameter sind bei Rauschen und Träger leer.

Der Modellhash dokumentiert, welches kompilierte Modell untersucht wurde.
Die Dateien liegen unter dem bereits von Git ausgeschlossenen Ordner `models`;
bei Bedarf separat sichern.

Existieren `cases.csv` oder `summary.json` bereits, bricht das Skript ab,
damit ein voriger Lauf erhalten bleibt. Für einen weiteren Lauf einen neuen
Ausgabeordner wählen:

```text
python diagnose_signal.py --output models/signal/diagnostics-run2
```

Mehr Beispiele:

```text
python diagnose_signal.py --per-class 200 --output models/signal/diagnostics-large
```

Das ergibt 4.200 Beispiele, nicht 600. Ein anderer Seed ist mit `--seed`
möglich. Die Gesamtgenauigkeit dieses festen SNR-Rasters ist nicht direkt mit
den bisherigen 91,5 % oder 76,7 % vergleichbar, weil die Stichproben und die
SNR-Verteilung unterschiedlich sind.

Die Gruppenauswertungen zeigen Zusammenhänge, belegen allein aber keine
isolierte Ursache. Ein kontrollierter Folgetest müsste einzelne Parameter
bei sonst gleichen Bedingungen verändern.

### Implementierung und Prüfung

`generate_iq()` unterstützt jetzt optional `return_metadata=True` und
`snr_db=...`. Ohne diese Optionen bleibt die bisherige Aufrufweise erhalten.
Ein Vergleich mit dem vorigen Generator bestätigte für 30 geprüfte Signale
bitidentische Standardausgaben.

```text
python -m unittest test_coral_worker test_diagnosis -v
```

Fünf Tests ohne Hardware prüften unter anderem Metadaten, Rauschskalierung,
Gruppenauswertung und die bestehende Prozessschnittstelle.
Der Diagnoselauf wurde inzwischen auf dem Windows-Rechner ausgeführt:
947/1.050 richtige Entscheidungen (90,2 %), 16 Fehler mit mindestens 80 %
Modellbewertung. Die Matrix lautet:

| Soll / erkannt | noise | carrier | fm |
|---|---:|---:|---:|
| noise | 349 | 1 | 0 |
| carrier | 1 | 284 | 65 |
| fm | 0 | 36 | 314 |

Die hochgeladene Zusammenfassung zeigte eine auffällige Pegelabhängigkeit:

| Amplitudenbereich | Träger richtig | FM richtig |
|---|---:|---:|
| 0,2–0,4 | 96/104 (92,3 %) | 65/91 (71,4 %) |
| 0,4–0,6 | 87/93 (93,5 %) | 85/93 (91,4 %) |
| 0,6–0,9 | 101/153 (66,0 %) | 164/166 (98,8 %) |

52 der 65 Träger→FM-Verwechslungen liegen im höchsten Amplitudenbereich,
26 der 36 FM→Träger-Verwechslungen im niedrigsten.
Die groben Frequenzbereiche erreichten insgesamt jeweils etwa 90 %.
Diese Gruppen belegen Zusammenhänge, aber noch keine isolierte Ursache.
Daher wurde anschließend der gepaarte Pegeltest aus Abschnitt 18 ergänzt.

## 18. Kontrollierter Pegeltest

`level_sweep.py` untersucht die beobachtete Pegelabhängigkeit ohne neues Training.

### Durchführung

In Anaconda Prompt:

```text
conda activate coral
cd /d C:\Users\thwie\source\repos\Coral-signal-recognition
git pull --ff-only
python level_sweep.py
```

Das bestehende kompilierte Modell bleibt unverändert. Es muss nicht erneut
trainiert oder kompiliert werden.

### Was konstant bleibt

Für jedes Grundsignal wird genau ein IQ-Block einschließlich Rauschen erzeugt.
Dieser Block wird achtmal gemeinsam skaliert:

- Zielamplituden: 0,2; 0,3; 0,4; 0,5; 0,6; 0,7; 0,8; 0,9.
- Festes SNR je Grundsignal: 0, +10 oder +25 dB.
- Frequenzversatz, Phase, FM-Hub und Modulationston bleiben innerhalb jeder
  Achtergruppe gleich.
- Auch die konkrete Rauschrealisierung bleibt gleich und wird mitskaliert.
- Es werden nur Träger und FM getestet; Rauschen war in der Diagnose bereits
  nahezu durchgehend richtig erkannt worden.
- Es wird kein IQ-Clipping ergänzt; das entspricht dem bisherigen Generator.
  Die bestehende Spektrogrammbegrenzung bleibt aktiv.

Damit verändert die Skalierung das Verhältnis von Nutzsignal zu Rauschen nicht.
Verändert sich die Vorhersage, zeigt das eine Abhängigkeit dieser gesamten
Vorverarbeitungs-/Modellkette vom absoluten Pegel. Es beweist noch nicht,
welche einzelne Komponente dafür verantwortlich ist.

### Anzahl und Ausgaben

Standard: 20 Grundsignale je Klasse und SNR-Stufe, Seed 2029.

- 3 SNR-Stufen × 2 Klassen × 20 Grundsignale = **120 Grundsignale**.
- Jedes Grundsignal wird bei acht Amplituden ausgewertet:
  **960 Klassifikationen**.

Die skalierten Varianten sind gepaarte Messungen. Sie sind keine 960
unabhängigen Testbeispiele.

Die Konsole zeigt je SNR-Stufe die Erkennungsraten für alle acht Amplituden.
Danach folgt die Anzahl der Grundsignale, deren Vorhersage sich innerhalb der
Achtergruppe verändert hat, insgesamt sowie getrennt nach Klasse und SNR.

Dateien unter `models/signal/level-sweep`:

| Datei | Inhalt |
|---|---|
| `cases.csv` | Alle skalierten Varianten mit gemeinsamer pair_id, Parametern und Modellbewertungen |
| `summary.json` | Matrizen je SNR/Amplitude, Vorhersagereihen je Grundsignal, Zahl der Klassenwechsel und Modellhash |

Ein stabiler Klassenname bedeutet nicht automatisch eine richtige Entscheidung.
Die Trefferquoten müssen zusammen mit der Zahl der Klassenwechsel gelesen werden.
Die Zusammenfassung enthält zusätzlich die Spannweite der Bewertung der wahren
Klasse je Achtergruppe.

Vorhandene Ergebnisse werden nicht überschrieben. Für eine Wiederholung:

```text
python level_sweep.py --output models/signal/level-sweep-run2
```

Mehr Grundsignale:

```text
python level_sweep.py --per-class 100 --output models/signal/level-sweep-large
```

Das ergibt 600 Grundsignale und 4.800 Klassifikationen.
Die Ausgaben liegen unter dem von Git ausgeschlossenen Ordner `models`
und sollten bei Bedarf separat gesichert werden.

### Prüfung und aktueller Status

```text
python -m unittest test_coral_worker test_diagnosis test_level_sweep -q
```

Sieben Tests ohne Hardware liefen erfolgreich. Geprüft wurden unter anderem
gemeinsame Skalierung des gesamten IQ-Blocks, unveränderte Parameter je
Achtergruppe und die Anzahl der Messungen.
Der tatsächliche Pegeltest wurde inzwischen auf dem USB-Coral durchgeführt:
Bei **64/120 Grundsignalen (53,3 %)** änderte sich die Klasse allein durch
gemeinsame Skalierung von Signal und Rauschen.

| SNR | Träger mit Klassenwechsel | FM mit Klassenwechsel |
|---|---:|---:|
| 0 dB | 17/20 | 16/20 |
| +10 dB | 7/20 | 9/20 |
| +25 dB | 14/20 | 1/20 |

Bei 0 dB SNR fiel die Träger-Trefferquote von 100 % bei Amplitude 0,3/0,4
auf 30 % bei Amplitude 0,9. Gleichzeitig stieg die FM-Trefferquote von 20 %
bei Amplitude 0,2 auf 100 % ab Amplitude 0,6.
Damit ist die Pegelabhängigkeit für diese Verarbeitungskette und Stichprobe
belegt. Die genaue Verteilung nach SNR und Parametern bleibt zusätzlich relevant.

Als kontrollierte Folgemaßnahme wurde eine zweite Modellvariante mit
RMS-Normalisierung vorbereitet; siehe Abschnitt 19.

## 19. Zweite Modellvariante mit RMS-Normalisierung

### Ziel und Stand

Die Rohdatenvariante unter `models/signal` bleibt Referenz.
Die neue Variante verwendet **denselben Generator und dieselbe Netzarchitektur**,
normalisiert aber jeden vollständigen IQ-Block vor der FFT auf komplexen
RMS-Wert **0,5**. Nutzsignal und Rauschen werden gemeinsam skaliert;
das SNR wird dadurch nicht gezielt verändert.

Die Vorverarbeitung liegt zentral in `signal_preprocessing.py`.
Ein vollständig leerer Nullblock bleibt null; NaN/Unendlich werden abgelehnt.
Normalisiert wird vor Fensterung, FFT und dB-Skalierung.

Die neue Variante muss neu trainiert werden. Eine nachträgliche Normalisierung
vor dem alten Modell ist kein gleichwertiger Vergleich.
Zehn Tests ohne Hardware, einschließlich Skalierungsinvarianz und Manifestprüfung,
bestanden. Ein kleiner normalisierter Datensatz wurde erfolgreich erzeugt.
Training, Kompilierung und Hardwareauswertung der RMS-Variante wurden inzwischen
erfolgreich ausgeführt: 98,7 % Testgenauigkeit und 0/120 Klassenwechsel im
kontrollierten Pegeltest. Die genauen Ergebnisse stehen in Abschnitt 20.

### Schritt A: Neue Daten separat erzeugen

Anaconda Prompt:

```text
conda activate coral
cd /d C:\Users\thwie\source\repos\Coral-signal-recognition
git pull --ff-only
python signal_lab.py --per-class 1000 --preprocessing rms-v1 --output data/normalized
```

Es entstehen 3.000 Trainings-, 600 Validierungs- und 600 Testbeispiele
unter `data/normalized`. Die bisherigen Dateien direkt unter `data`
bleiben erhalten. Gleicher Standard-Seed und gleiche Parameterziehung
ermöglichen einen Vergleich auf entsprechenden synthetischen Grundsignalen.

Das Programm verhindert die versehentliche Speicherung normalisierter Daten
im bisherigen Standardordner `data`.
Erneutes Ausführen in `data/normalized` überschreibt dortige Daten.

### Schritt B: Neues Modell separat trainieren

In derselben Anaconda Prompt:

```text
conda activate coral-train
python train_signal.py --data data/normalized --output models/signal-rms
```

Die Trainingsumgebung und deren installierte Pakete können weiterverwendet werden.
Ausgaben unter `models/signal-rms`:

- `signal_float.h5`, `signal_int8.tflite`.
- `report.json` und `labels.txt`.
- Neu: **`preprocessing.json`**, das die Vorverarbeitung `rms-v1`,
  Ziel-RMS, Klassen und Eingangsdimensionen festhält.

Der Trainer liest die Vorverarbeitung aus den Datenmetadaten.
Er normalisiert die fertigen Spektrogramme nicht erneut.
Der Standardmodellordner `models/signal` ist für diese neue Variante gesperrt;
einen separaten Ausgabeordner explizit wählen.

### Schritt C: Kompilieren und Manifest danebenlegen

Nach erfolgreichem Training im **Ubuntu-Terminal**:

```bash
cd /mnt/c/Users/thwie/source/repos/Coral-signal-recognition/models/signal-rms
mkdir -p compiled
~/coral-tools/compiler/x86_64/edgetpu_compiler -s -o compiled signal_int8.tflite
```

Nur nach erfolgreicher Kompilierung:

```bash
cp preprocessing.json compiled/preprocessing.json
```

**Wichtig:** Das Manifest muss neben dem kompilierten Modell liegen.
Wird das Modell später verschoben oder gesichert, `preprocessing.json`
immer mitnehmen. Beide Dateien bilden für die IQ-Inferenz eine Einheit.

Alte Modelle ohne Manifest werden weiterhin als `raw` behandelt.
Damit bleiben die bisherigen Befehle für das Referenzmodell verwendbar.
Die explizite Angabe `--preprocessing rms-v1` beim folgenden Pegeltest
verhindert, dass ein fehlendes/falsches Manifest unbemerkt bleibt.

### Schritt D: Passenden gespeicherten Testsatz auswerten

Zurück in **Anaconda Prompt unter Windows**:

```text
conda activate coral
python evaluate_coral.py --model models/signal-rms/compiled/signal_int8_edgetpu.tflite --data data/normalized
```

Dieser Auswerter bekommt bereits vorbereitete normalisierte Spektrogramme
aus `data/normalized`; er führt keine IQ-Normalisierung aus.
Modell und Datensatz müssen daher wie im Befehl zusammenpassen.

Rauscherkennung, Träger- und FM-Recall getrennt betrachten.
Eine hohe Gesamtgenauigkeit allein genügt nicht.

### Schritt E: Gepaarter Pegeltest der neuen Variante

```text
python level_sweep.py --model models/signal-rms/compiled/signal_int8_edgetpu.tflite --preprocessing rms-v1 --output models/signal-rms/level-sweep
```

Der Worker liest das Manifest und normalisiert jeden unvorverarbeiteten
IQ-Eingang genau einmal vor der Spektrogrammberechnung.
Das Modell bleibt für alle Aufträge geladen.

Ohne zusätzliche Argumentänderungen verwendet der Test wieder 120 Grundsignale,
acht Amplituden und Seed 2029. Es sind die gleichen Grundsignale wie beim
bisherigen Referenz-Pegeltest.

Verglichen werden:

1. Klassenwechsel bei Skalierung: Referenz **64/120**.
2. Trefferquoten je Klasse, SNR und Amplitude.
3. Genauigkeit auf den entsprechenden gespeicherten Testdaten, einschließlich Rauschen.

Die Normalisierung soll die Pegelabhängigkeit reduzieren. Das belegt allein
noch keine höhere Klassifikationsgenauigkeit: Ein Modell kann über alle
Amplituden hinweg stabil falsch liegen. Auch reale Empfangsdaten fehlen weiterhin.

### Tests ohne Hardware

```text
python -m unittest test_coral_worker test_diagnosis test_level_sweep test_preprocessing -q
```

Zusätzliche beziehungsweise erweiterte Funktionen:

| Datei | Änderung |
|---|---|
| `signal_preprocessing.py` | Gemeinsame RMS-Normalisierung und Manifestprüfung |
| `signal_lab.py` | Option `--preprocessing rms-v1` für neue Datensätze |
| `train_signal.py` | Metadatenprüfung und Export des Modellmanifests |
| `coral_worker.py` | Auswahl der Vorverarbeitung über das Modellmanifest |
| `level_sweep.py` | Option `--preprocessing` zur Prüfung der erwarteten Variante |
| `test_preprocessing.py` | Invarianz-, Nullblock- und Manifesttests |

Neue Daten und Modelle liegen weiterhin unter den von Git ausgeschlossenen
Ordnern `data` und `models`. Referenz und Kandidat gemeinsam separat sichern.

## 20. Validierter RMS-Stand, Sicherung und Prozesstest

### Dokumentierte Ergebnisse auf dem Windows-Rechner

Das normalisierte Modell wurde mit derselben Netzarchitektur auf den
entsprechenden normalisierten synthetischen Daten trainiert.

| Kennzahl | Rohdatenreferenz | RMS-Variante |
|---|---:|---:|
| Float-Testgenauigkeit auf CPU | 91,3 % | 98,0 % |
| INT8-Testgenauigkeit auf CPU | 91,5 % | 98,7 % |
| INT8-Testgenauigkeit auf Coral | 91,5 % | 98,7 % |
| Rauschen richtig | 200/200 | 200/200 |
| Träger richtig | 169/200 | 194/200 |
| FM richtig | 180/200 | 198/200 |
| Träger/FM-Verwechslungen | 51 | 6 |
| Klassenwechsel durch Skalierung | 64/120 | 0/120 |
| Median Inferenz ohne Vorverarbeitung | 0,427 ms | 0,464 ms |
| p95 Inferenz ohne Vorverarbeitung | 0,543 ms | 0,655 ms |

Die Laufzeiten stammen aus verschiedenen Läufen und erlauben keinen gesicherten
Vergleich kleiner Geschwindigkeitsunterschiede. Ein Lauf misst jeweils
`invoke()` einschließlich darin anfallender Python-/USB-Kosten, ohne FFT,
Normalisierung, Quantisierung und Datentransport zum Unterprozess.

Die RMS-Konfusionsmatrix war auf CPU und Coral aggregiert gleich:

| Soll / erkannt | noise | carrier | fm |
|---|---:|---:|---:|
| noise | 200 | 0 | 0 |
| carrier | 2 | 194 | 4 |
| fm | 0 | 2 | 198 |

Es wurden 592/600 Beispiele richtig klassifiziert. Identische aggregierte
Matrizen beweisen nicht die Gleichheit jeder einzelnen CPU-/Coral-Entscheidung.

Der Compiler 14.1.317412892 ordnete auch beim RMS-Modell alle zehn Operationen
einem Edge-TPU-Teilgraphen zu. Das kompilierte Modell ist 84,56 KiB groß;
74,25 KiB On-Chip-Speicher werden für Modellparameter verwendet.

### RMS-Pegeltest: vollständige Ergebniseinordnung

Für alle acht Amplituden von 0,2 bis 0,9 blieben die Trefferquoten je SNR gleich:

| SNR | Träger richtig bei jedem Pegel | FM richtig bei jedem Pegel |
|---|---:|---:|
| 0 dB | 20/20 (100 %) | 20/20 (100 %) |
| +10 dB | 20/20 (100 %) | 18/20 (90 %) |
| +25 dB | 20/20 (100 %) | 19/20 (95 %) |

Damit änderte sich bei keinem der 120 Grundsignale die Klasse durch Skalierung.
Drei FM-Grundsignale wurden dennoch bei allen acht Pegeln falsch klassifiziert.
Die Normalisierung reduzierte also die beobachtete Pegelabhängigkeit und
verbesserte hier die Erkennung; sie beseitigte nicht sämtliche Fehler.

Die 960 Klassifikationen sind gepaarte Varianten von 120 Grundsignalen.
Es wurden keine echten Empfangssignale geprüft.

### Lokalen Stand sichern und danach RMS-Prozess testen

Die trainierten Modelle, Daten und Diagnose-JSONs befinden sich auf dem
Windows-Rechner, nicht im GitHub-Repository. Die USB-Hardware ist ebenfalls
nur dort verfügbar. Deshalb müssen Sicherung und Hardwaretest dort laufen.

In **Anaconda Prompt unter Windows**:

```text
conda activate coral
cd /d C:\Users\thwie\source\repos\Coral-signal-recognition
git pull --ff-only
python checkpoint_rms.py
```

Das Skript arbeitet in dieser Reihenfolge:

1. Vorhandensein von Float-, INT8- und kompiliertem RMS-Modell,
   `preprocessing.json`, Trainingsbericht und normalisierten Daten prüfen.
2. Die ausgewählte Vorverarbeitung und die Datenmetadaten auf `rms-v1` prüfen.
3. Eine neue ZIP-Sicherung mit Zeitstempel außerhalb des Projektordners erstellen.
4. Alle aufgenommenen Dateien erneut aus der ZIP lesen und ihre SHA-256 prüfen.
5. Erst nach erfolgreicher Sicherung den persistenten RMS-Prozesstest starten.
6. Seine Ausgabe einschließlich Exit-Code in einer separaten Protokolldatei
   neben der ZIP speichern und gleichzeitig im Terminal anzeigen.

Standardziel:

```text
C:\Users\thwie\source\repos\Coral-signal-recognition-backups
```

Die Sicherung ist lokal. Sie wird nicht auf GitHub oder einen anderen Dienst
hochgeladen. Für eine zusätzliche Sicherung auf einem anderen Datenträger
kann ein eigener externer Zielordner angegeben werden:

```text
python checkpoint_rms.py --output D:\Coral-Backups
```

Der Zielordner muss außerhalb des Projektordners liegen. Vorhandene
Sicherungen werden nicht überschrieben. Der Platzbedarf hängt vor allem von
den erzeugten Datensätzen ab. Vor dem Sichern Training und andere Programme
beenden, die die zu sichernden Dateien gerade überschreiben könnten.

### Inhalt der Sicherung

| Archivbestandteil | Inhalt |
|---|---|
| `project/` | Vorhandene versionierte Dateien und zusätzliche Python-/Markdown-/Requirements-Dateien im Projektstamm |
| `project/models/` | Beide Modellvarianten, kompilierte Modelle, Manifeste, Berichte und vorhandene Diagnosedateien |
| `project/data/` | Vorhandene Roh- und normalisierte Datensätze einschließlich Metadaten und PNG-Vorschauen |
| `repository.bundle` | Git-Commits und lokale Git-Referenzen |
| `working-tree.patch` | Nicht committete Änderungen versionierter Dateien gegenüber HEAD |
| `environment.json` | Git-Revision/-Status, aktives Python und dessen Pakete; vorhandene Conda-Paketmetadaten von coral-train |
| `SHA256SUMS.json` | Hashes der archivierten Nutzdateien |
| `RESTORE.txt` | Hinweise zum Wiederherstellen |

Installiertes Anaconda, Windows-Treiber, Edge-TPU-Laufzeit und der Compilerordner
in Ubuntu werden nicht als installierte Programme archiviert. Die README
beschreibt deren Einrichtung. Beliebige zusätzliche unversionierte Ordner
außerhalb von `data` und `models` gehören nicht automatisch zum Snapshot.

Das neue Testprotokoll wird erst **nach** der ZIP-Erstellung geschrieben und
liegt daher neben der ZIP. Es ist nicht rückwirkend im Archiv enthalten.

Wenn die Sicherung abbricht, startet der Prozesstest nicht.
Eine unvollständige Datei hat die Endung `.zip.partial` und gilt nicht als
abgeschlossene Sicherung.
Scheitert anschließend der Hardwaretest, bleibt die bereits geprüfte ZIP erhalten.

### Was der RMS-Prozesstest prüft

Der Sicherungsbefehl startet danach intern:

```text
python live_demo.py --model models/signal-rms/compiled/signal_int8_edgetpu.tflite --preprocessing rms-v1
```

Das erwartete Manifest muss direkt neben dem kompilierten Modell liegen.
Der Worker prüft die Erwartung vor der Inferenz und meldet die verwendete
Vorverarbeitung. Die Ausgabe muss **`Vorverarbeitung: rms-v1`** enthalten.
Ein fehlendes oder abweichendes Manifest führt bei dieser expliziten Erwartung
zum Abbruch statt zu einer versehentlichen Rohdaten-Inferenz.

Das Modell bleibt für 30 neue synthetische IQ-Blöcke geladen: je zehn Rauschen,
Träger und FM, Generator-Seed 2027. Die Sollklasse wird nicht an den Worker
übertragen. Die IQ-Blöcke werden im Worker einmal normalisiert und in
Spektrogramme umgewandelt.

Das Programm zeigt alle Entscheidungen, Modellbewertungen, Inferenz- und
Anfragezeiten und abschließend die Konfusionsmatrix. Es gibt keine vorgegebene
Mindesttrefferzahl. Erfolgreiche Kommunikation ist von fehlerfreier
Klassifikation zu unterscheiden.

Nur sichern, ohne Hardwaretest:

```text
python checkpoint_rms.py --backup-only
```

Nur den RMS-Prozesstest erneut ausführen:

```text
python live_demo.py --model models/signal-rms/compiled/signal_int8_edgetpu.tflite --preprocessing rms-v1
```

### Prüfung und Status

Die Archivierung wurde in einem temporären echten Git-Repository getestet:
Modellbytes und JSON-Dateien werden aufgenommen, Hashes stimmen, und das
archivierte Git-Bundle lässt sich verifizieren.
Die vorhandenen zehn Tests sowie drei Sicherungstests bestanden.

```text
python -m unittest test_coral_worker test_diagnosis test_level_sweep test_preprocessing test_checkpoint -q
```

Die aktualisierte Dokumentation und Sicherungs-/Testprogramme sind auf GitHub.
Der RMS-Prozesstest wurde inzwischen vom Nutzer ausgeführt: 29/30 Treffer,
Matrix [[10,0,0],[0,10,0],[1,0,9]], Anfrage-Median 6,63 ms.
Die frühere RMS-Sicherung blieb dabei erhalten. Für den aktuellen RadioML-Stand
ist die neue Sicherung nach [BACKUP_RADIOML.md](BACKUP_RADIOML.md) auszuführen.
