# Coral-signal-recognition

Experiment zur Klassifikation von SDR-Signalen mit dem Google Coral USB Accelerator.
Startklassen: Rauschen, unmodulierter Traeger und schmalbandiges FM-Testsignal.

## Aktueller Stand

- Coral-Hardwaretest mit offiziellem Vogelmodell und Zeitmessung.
- Generator fuer synthetische komplexe IQ-Signale und numerische Spektrogramme.
- Getrennte Trainings-, Validierungs- und Testdaten sowie PNG-Vorschauen.
- Noch kein trainiertes Funkmodell und noch keine Verbindung zum Qt-Analyzer.

Das Vogelmodell ist ausschliesslich ein Hardwaretest. Seine Ergebnisse sind
fuer Funksignale nicht verwendbar.

## Windows: erste Schritte

Anaconda Prompt verwenden. Die bereits funktionierende Umgebung beibehalten:

```text
conda activate coral
git clone https://github.com/Thomas-Wierzbicki/Coral-signal-recognition.git
cd Coral-signal-recognition
python -m pip install -r requirements.txt
python coral_check.py
python signal_lab.py
```

Falls das Repository bereits lokal vorhanden ist: in seinen Ordner wechseln und
statt erneut zu klonen zuerst `git pull --ff-only` ausfuehren.

Nur falls PyCoral in dieser Umgebung noch fehlt:

```text
python -m pip install --extra-index-url https://google-coral.github.io/py-repo/ "pycoral~=2.0"
python -m pip install -r requirements.txt
```

Die Edge-TPU-Laufzeit muss separat installiert sein; ein sichtbares USB-Geraet
allein bestaetigt noch keine erfolgreiche Inferenz.

## Ergebnis des Signalgenerators

`data/train.npz`, `validation.npz`, `test.npz` enthalten jeweils:

- `x`: float32-Spektrogramme, Form [Anzahl, 64, 256, 1], Werte 0 bis 1.
- `y`: Klassenindex 0 = noise, 1 = carrier, 2 = fm.

`data/metadata.json` beschreibt Abtastrate, Klassen und Vorverarbeitung.
Die PNG-Dateien dienen nur der Ansicht; trainiert werden soll mit numerischen Daten.

Standard: 600 Trainingsbeispiele, je 120 Validierungs- und Testbeispiele.
Mehr Daten:

```text
python signal_lab.py --per-class 1000
```

Das Programm ueberschreibt seine generierten Dateien im gewaehlten Ausgabeordner.
IQ-Abtastrate: 240 kS/s; FFT: 256; 64 aufeinanderfolgende Bloecke.
Fester Pegelbereich: -80 bis 0 dBFS, keine individuelle Bildnormalisierung.
SNR, Pegel, Frequenzversatz und FM-Parameter variieren.
Dies ist eine synthetische Machbarkeitsstudie. Reale Empfaengerdaten mit
unabhaengigen Aufnahmesitzungen werden fuer eine belastbare Bewertung benoetigt.
Das Rauschen stammt aus demselben Generator wie das Hintergrundrauschen der Signale;
eine Erkennung kann trotzdem auf den kuenstlichen Daten besser wirken als im Funkbetrieb.

## Weitere Entwicklung

1. Kompaktes CNN trainieren und auf zurueckgehaltenen Daten auswerten.
2. Vollstaendig ganzzahlig quantisiertes TFLite-Modell exportieren.
3. Unter geeignetem Linux-x86-64-System mit dem Edge-TPU-Compiler kompilieren
   und dessen Operatorzuordnung pruefen.
4. CPU- und Coral-Ergebnisse vergleichen.
5. Qt-Analyzer ueber einen dauerhaft laufenden Python-Prozess anbinden.

Die bestehende Anaconda-Umgebung dient der Inferenz. Training bekommt eine
separate Umgebung, damit die funktionierende Coral-Installation erhalten bleibt.
Der Analyzer verwendet bisher eine andere Signalverarbeitung; vor einer Kopplung
muessen Abtastrate, Vorverarbeitung und Frequenzausschnitt angeglichen werden.

## Referenzen

- https://coral.ai/docs/accelerator/get-started/
- https://github.com/google-coral/test_data
- https://coral.ai/docs/edgetpu/models-intro/
