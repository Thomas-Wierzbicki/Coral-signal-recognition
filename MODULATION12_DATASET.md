# Erste zwoelfklassige Modulationsbasis

## Ziel und Umfang

Modulationserkennung ist die erste Umsetzungsetappe. Drohnenerkennung bleibt
Projektziel; Datenformat und Sitzungs-/Quellverwaltung sollen spaeter ergaenzt
werden. Die hier erzeugten synthetischen Klassen sind keine Drohnenprotokolle.

600 Beispiele je Klasse, 7200 insgesamt:

| Split | Je Klasse | Gesamt |
|---|---:|---:|
| train | 420 | 5040 |
| validation | 90 | 1080 |
| test | 90 | 1080 |

Klassenfolge identisch zur Probe: am-dsb, fm, bpsk, qpsk, 16qam, 64qam,
2fsk, 2gfsk, 2gmsk, ofdm-64, ook, noise. Diese erste Klassenliste mischt
Modulationsordnungen und OFDM-Signalform. NFM/WFM sind noch keine getrennten
Labels. noise bedeutet reine Rauschrealisierung, nicht allgemeines unknown.

Jedes Beispiel enthaelt 4096 komplexe float32-Paare bei nominell 32 kS/s,
entsprechend 128 ms. Ein eigenes synthetisches Grundsignal wird je Beispiel
neu erzeugt. Keine Augmentierung desselben Signals ueber mehrere Splits.

## Erzeugung auf dem Trainingsrechner (Ubuntu)

```bash
cd ~/rf-learning
source .venv/bin/activate
git -C project pull --ff-only
python project/build_modulation_dataset.py --output data/modulation12-v1
python -m pip freeze > data/modulation12-v1/environment-freeze.txt
```

TorchSig 2.2.0 erforderlich. Erzeugung auf CPU, GPU wird erst beim Training
benoetigt. Bestehende Ausgabeordner werden abgelehnt. Fuer einen kleinen
Funktionslauf --per-class 20 und einen anderen --output nutzen; nicht als
Benchmark verwenden. Voller Rohumfang der I/Q-Arrays rund 236 MB vor
Kompression, zusaetzlich JSON und voruebergehende Arrays im RAM.

## Kontrolliertes FM

Eigener Generator controlled-fm-v1, getrennt vom unveraenderten TorchSig-FM
in Probe/Audit. FIR-gefilterte Gaußrauschnachricht, keine Sprache.
513-Tap-FIR, untere Nachrichtengrenze max(30 Hz, 0,1*obere Grenze), obere
Grenze zufaellig 0,08..0,22 der angeforderten Bandbreite. Nachricht ueber den
ganzen 16384-Sample-Block auf Spitze 1 normieren. Spitzenhubparameter ist
0,65..1,0 * (angeforderte Bandbreite/2 - obere Nachrichtengrenze).
FM durch Phasenintegration, mittlere 4096 Samples auswerten. Dokumentiert
werden Nachrichtenband, FIR-Laenge, Hubparameter und tatsaechlicher Hub/RMS
im mittleren Ausschnitt. Wegen endlicher Filter-/Spektralanteile ist das
keine Garantie einer scharfen belegten Bandgrenze. Carson-artige Parametrisierung
ist keine Messung der 90-%-Leistungsbreite.

Alle anderen Signalklassen verwenden die geprueften TorchSig-Modulatoren wie
in der Probe, einschliesslich Mittelausschnitt. OFDM zunaechst 64 Untertraeger,
CP16. RRC-Rolloff fuer Konstellationen variiert 0,2..0,5.

## Gemeinsamer Parameterplan und Grenzen

Je Split identischer Plan fuer alle elf Signalklassen: Bandbreitenvorgabe
1500..12000 Hz, Frequenzversatz -4800..4800 Hz, Phase -pi..pi, AWGN-SNR
-5/0/5/10/15/20/25 dB moeglichst gleichmaessig. Unabhaengige Nachrichten-/Symbol-
und Rausch-Seeds je Klasse/Beispiel. Splits verwenden andere Zufallsstroeme.
Rauschen hat keinen Nutzsignalparameter; snr_db ist fuer noise in NPZ NaN,
in JSON sind die Signalparameter null. Das ist beabsichtigt.

Sauberes Signal auf Leistung 1 normieren, Frequenz/Phase anwenden, AWGN mit
entsprechender erwarteter Leistung addieren, Gesamtblock auf RMS 0,5 skalieren.
Damit geht absolute Empfangsleistung verloren. Rausch-SNR ist nominell.
Bandbreitenvorgaben sind kein Nachweis gleicher realisierter Bandbreitenverteilungen;
Messwerte werden festgehalten. Erste Baseline, noch kein Fading, Hardware-IQ-
Fehler, weitere Sender oder Burstmodelle. Keine Zusage fuer reale NFM-/Drohnen-
Generalisation. Keine gezielte Auswahl nach einem Klassifikatorergebnis.

## Ausgaben und Datenintegritaet

train.npz, validation.npz, test.npz enthalten iq (N,4096) complex64, y,
source_id, condition_id, snr_db. *_parameters.jsonl enthaelt pro Beispiel
Generator-/Rauschseeds, saubere/verrauschte Kennwerte, Generatorversion,
I/Q-SHA256 und FM-Parameter. Gemeinsame condition_id verbindet die paarweise
gleichen aeusseren Bedingungen verschiedener Klassen innerhalb eines Splits.
Bei spaeterer Neuaufteilung diese Gruppen zusammenlassen.

metadata.json enthaelt Versionen, Skripthash, Klassen, Verteilungen,
Splitgroessen und Hashes von NPZ-/Parameterdateien. Jede NPZ wird erneut
gelesen und auf Form, RMS und Klassenbalance geprueft; eindeutige source_ids
und bitidentische Beispiel-Duplikate werden waehrend Generierung kontrolliert.
DATASET_COMPLETE entsteht zuletzt. Unvollstaendige Ordner nicht zum Training
verwenden, fuer Wiederholung neuen Ordner waehlen. Keine alten Daten/Modelle
werden ueberschrieben.

## Validierung und naechster Schritt

```bash
python -m unittest discover -s project -p test_modulation_dataset.py -v
```

Drei Tests bestanden: FM-Amplitude/Hub/Reproduzierbarkeit, verschiedene Split-
plaene, Quell-ID-Disjunktheit, RMS, Noise-NaN, Dateihashes und Ueberschreibschutz.
Kleiner echter TorchSig-Lauf mit allen Klassen (240 Beispiele) abgeschlossen.
Nach erfolgreicher Standarderzeugung folgt ein kompaktes GPU-Referenzmodell.
Testsplit nicht fuer Parameterwahl nutzen. Frequenzrobustheit und spaeter
unabhaengige Realaufnahmen separat bewerten; bisherige Relaisminute ist bekannte
Entwicklungsreferenz. WSL-Ausgaben gehoeren nicht automatisch zu alten Windows-ZIPs.
Auch die vollstaendige Standardgenerierung mit 7200 Beispielen wurde hier
mit TorchSig 2.2.0 auf CPU abgeschlossen und alle drei NPZ-Dateien erfolgreich
zurueckgelesen/geprueft. Das belegt die Datenpipeline, keine Erkennungsleistung.
