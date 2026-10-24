# TorchSig-Einstieg: Generatorpruefung vor dem Training

Beide Projektziele bleiben gleichrangig: breite Modulationserkennung und
Drohnen-Funksystem-/Protokollerkennung. Dieser Schritt prueft die gemeinsame
synthetische Signalbasis. Er liefert noch keinen Drohnenklassifikator oder
Protokolldecoder. Echte HackRF-/Drohnenaufnahmen bleiben spaeter getrennte Tests.

## Bestaetigte Umgebung beim Nutzer

Ubuntu/WSL, Python 3.12.3, RTX 3080 Laptop mit 16 GiB, PyTorch 2.9.1+cu126,
TorchSig 2.2.0. CUDA verfuegbar, echte GPU-Matrixberechnung erfolgreich,
pip check ohne defekte Abhaengigkeiten. venv unter ~/rf-learning/.venv.
Die Datenerzeugung hier verwendet NumPy/SciPy-basierte TorchSig-Modulatoren
auf CPU. Eine geringe GPU-Auslastung waehrend dieses Schritts ist erwartbar.

## Code auf Trainingsrechner holen

Ubuntu-Terminal:

```bash
cd ~/rf-learning
source .venv/bin/activate
git clone https://github.com/Thomas-Wierzbicki/Coral-signal-recognition.git project
python project/torchsig_probe.py --output data/torchsig-probe-v1
```

Falls project bereits als Checkout existiert, stattdessen git -C project pull
--ff-only verwenden. Nicht in ein existierendes Verzeichnis erneut klonen.
Befehle werden im Linux-Dateisystem ausgefuehrt, nicht in /mnt/c/WINDOWS/system32.
Die vorhandenen Windows-Coral-Umgebungen bleiben erhalten.

## Umfang und Protokoll

144 Beispiele: zwoelf pro Klasse, 4096 komplexe Samples je Beispiel, nominell
32 kS/s (128 ms). Klassen: am-dsb, fm, bpsk, qpsk, 16qam, 64qam, 2fsk,
2gfsk, 2gmsk, ofdm-64, ook und noise. Keine direkten Drohnenprotokolllabels.
OFDM ist eine Mehrtraeger-Signalform, die PSK/QAM auf Untertraegern verwenden
kann; diese Klassenliste ist eine erste Generatorauswahl, keine endgueltige
hierarchische Taxonomie.

Generatoraufrufe wurden gegen das veroeffentlichte TorchSig-2.2.0-Wheel gelesen.
Verwendet werden dessen am/fm/constellation/fsk/ofdm_modulator-Funktionen.
Laengeres Signal mit 16384 Samples erzeugen, mittlere 4096 verwenden, damit
Filter-/Paddingraender weniger dominant werden. Pro Beispiel separate Generator-
und Rausch-Seeds, keine Ausschnitte einer gemeinsamen Aufnahme als Samples.

Ein gemeinsamer Parameterplan (Seed 2040) wird fuer alle elf Signalklassen
wiederverwendet: angeforderte Bandbreite 1920..7040 Hz, Frequenzversatz
-3840..+3840 Hz, Phase -pi..pi und nominelles AWGN-SNR zyklisch 0/10/20 dB.
RRC-Rolloff 0,2..0,5 bei Konstellationssignalen. OFDM 64 Untertraeger, CP 16.
Es ist bewusst noch keine umfassende Symbolraten-/Kanalvariation.
Rauschen hat weder Nutzsignal noch SNR, Bandbreiten- oder Offsetlabel.
TorchSig-Interne Zufallsparameter koennen trotz gleicher Zielbandbreite
unterschiedliche reale Spektralbreiten ergeben. Kein physikalisches Gleichsetzen
von Ziel-3dB-Bandbreite und gemessener 90-%-Leistungsbreite.

Nutzsignal vor AWGN auf Leistung 1 normieren, komplexes Rauschen mit erwarteter
Leistung 10^(-SNR/10) addieren, anschliessend gesamtes Beispiel auf komplexes
RMS 0,5. Realisiertes SNR schwankt mit der Rauschrealisierung. Frequenzversatz
wird vor AWGN hinzugefuegt; kein Alias-Freiheitsbeweis durch Zielbandbreite.
Die Spektren sind vor Training zu pruefen.

## Ausgaben und Beurteilung

- probe.npz: iq (144,4096) complex64, y, example_id (Parameterpaarung je Klasse).
- metadata.json: Klassen, gemeinsamer Plan, Seeds, Versionen, angeforderte
  Parameter sowie saubere und verrauschte Spektrums-/Amplitudenkennwerte.
- Zwoelf PNGs: Beispiel bei angeforderten 20 dB fuer Signalklassen, Rauschen
  ohne SNR; I/Q-Zeitverlauf, Huellkurve und relatives Spektrum.
- PROBE_COMPLETE: erst nach allen Dateien/Grafiken.

Vorhandener Ausgabeordner wird abgelehnt. Probe ist KEIN Trainings-/Testsplit.
Bei 0 dB erfasst die 90-%-Breite viel Rauschen; fuer Bandbreitenpruefung die
clean_metrics in JSON und 20-dB-Bilder heranziehen. FM/AM verwenden TorchSig-
Zufallsnachrichten, keine echte Sprache. Aktuell kein Fading, Hardware-IQ-Fehler,
mehrere Sender oder Burstmischungen. Das muss vor einem Trainingsbenchmark
schrittweise kontrolliert erweitert werden. Keine Erkennungsleistung behaupten.

Bitte Konsolenausgabe und erste Spektren pruefen. Danach erst split-/sitzungs-
sicheren Datensatz und Modellpfade fuer beide Erkennungsziele festlegen.

## Reproduzierbarkeit und Sicherung

```bash
python -m pip freeze > data/torchsig-probe-v1/environment-freeze.txt
python -m unittest discover -s project -p test_torchsig_probe.py -v
```

Projektcode auf GitHub, erzeugte Daten lokal. Die bisherige Windows-ZIP enthaelt
keine spaeter erzeugten WSL-Dateien. Fuer die Probe Quelle, metadata.json,
probe.npz, PNGs und environment-freeze.txt gemeinsam separat sichern; nicht
behaupten, die fruehere Sicherung enthalte diese neue Umgebung.

Primaerquellen:
https://pypi.org/project/torchsig/2.2.0/
https://github.com/TorchDSP/torchsig

## Hier ausgefuehrte Validierung

TorchSig 2.2.0 aus dem veroeffentlichten Wheel mit PyTorch/Torchaudio 2.9.1 CPU
in eigener Linux-Python-3.12-Umgebung installiert. Vollstaendiger Standardlauf
mit allen 144 Beispielen und zwoelf PNGs erfolgreich. Arrayform, endliche Werte,
RMS 0,5 und Klassenbalance geprueft; FM-/OFDM-PNGs visuell kontrolliert.
Zwei Hilfs-/Pipeline-Tests mit simuliertem Generator bestanden. Dies bestaetigt
die Generierung, nicht Modulations- oder Drohnenerkennungsleistung. Nutzers GPU-
Umgebung wurde separat bereits durch die Matrixberechnung bestaetigt.
Ein zweiter vollstaendiger Generierungslauf lieferte bei gleichem Seed alle
144 I/Q-Beispiele bitidentisch (zweiter Lauf ohne erneutes Plotten).
