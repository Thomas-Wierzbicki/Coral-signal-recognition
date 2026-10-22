# Erster echter Empfang: RTL-SDR Blog V3

## Ziel

Zuerst eine nachvollziehbare Roh-I/Q-Aufnahme erstellen und pruefen. Noch keine
Modulationsklassifikation: RadioML legt hier keine reale Hz-Abtastrate fest.
Breitbandige RTL-Daten koennen mehrere Sender enthalten; Kanalwahl, Filterung,
Frequenzversatz und zeitliche Skalierung muessen vor einem Modellvergleich
festgelegt werden. 1024 rohe RTL-Samples direkt einzuspeisen waere noch keine
validierte Empfangskette.

## 1. Recorder finden

Windows, Anaconda Prompt, Projektordner, Umgebung coral:

```bat
git pull --ff-only
conda activate coral
python capture_rtl.py --check
```

Es wird nur der Dateipfad zu rtl_sdr.exe gesucht. Kein Zugriff auf Hardware,
kein Test der DLL-Abhaengigkeiten oder Treiber. Liegt das Programm nicht in PATH:

```bat
python capture_rtl.py --rtl-sdr "C:\Pfad\zu\rtl_sdr.exe" --check
```

Den wirklichen Pfad verwenden. Das Programm samt passenden DLLs muss bereits
installiert sein. Keine Treiberwechsel auf Verdacht. Falls das Werkzeug fehlt,
separat dessen Einrichtung klaeren; vorhandener funktionierender SDR-Empfang
soll erhalten bleiben. SDR-Analyzer/SDR-Programme vor Aufnahme schliessen,
damit nur ein Programm auf den Stick zugreift.

## 2. Bekannte Empfangsfrequenz waehlen

Fuer einen ersten Kontrollversuch eine am Standort bereits bekannte UKW-FM-
Rundfunkfrequenz verwenden. Der folgende Wert 100 MHz ist NUR ein Zahlenbeispiel,
keine recherchierte Senderfrequenz und keine Ground-Truth-Zusage:

```bat
python capture_rtl.py --frequency 100000000 --seconds 5 --gain 20.7 --label unknown
```

Bei abweichendem Recorderpfad auch hier --rtl-sdr angeben. Frequenz in Hz.
Standardrate 1024000 Samples/s, 5 Sekunden, 5120000 komplexe Samples,
10240000 Byte Rohdaten. Gewinn angefordert 20,7 dB; Recorder kann auf eine
verfuegbare Stufe runden. --gain 0 bedeutet Auto. Weitere Parameter --ppm,
--device, --sample-rate und --output; --output muss neu sein.

Ausgabe data/recordings/rtl-UTC-Zeitstempel/:

| Datei | Inhalt |
|---|---|
| capture.cu8 | Unveraenderte unsigned 8-bit I/Q-Paare, I dann Q |
| recorder.log | Vollstaendiges stdout/stderr des Recorders, inklusive realen Einstellungen/Fehlern |
| metadata.json | Angeforderte Einstellungen, Recorderhash, Kommando, Labelbeschreibung, Dateihash und Status |
| CAPTURE_COMPLETE | Nur bei erfolgreichem Exit und exakt erwarteter Dateigroesse |

Angeforderte Werte sind keine Messung der tatsaechlichen Frequenz/Rate. Log
mitlesen. Erfolg und Dateilaenge schliessen Sampleverluste/USB-Probleme nicht
allein aus. Bei Abbruch/Timeout bleibt die Aufnahme als unvollstaendig erhalten.
Keine automatische Loeschung oder Ueberschreibung alter Aufnahmen.
Es wird keine Bias-Tee-/Direct-Sampling-Funktion eingeschaltet.

## 3. Qualitaet pruefen

Den vom Recorder ausgegebenen Ordnerpfad einsetzen:

```bat
python inspect_rtl_iq.py "data\recordings\rtl-ZEITSTEMPEL"
```

SHA256 und Laenge pruefen, cu8 dekodieren als (Wert-127,5)/127,5, RMS/Gleichanteil
und ADC-Randwertanteil ueber alle Samples bestimmen. Randwerte sind ein Hinweis,
kein alleiniger Beweis fuer Uebersteuerung. FFT-Vorschau aus bis zu 64 Fenstern
mit 1024 Samples nach nominell 0,25 Sekunden (bei kurzen Aufnahmen spaeter
verfuegbarem Abschnitt). Staerkster FFT-Bin ausserhalb der mittleren fuenf Bins
ist keine automatische Kanal-/Bandbreitenerkennung. Frequenzen sind nominell.

quality.json und spectrum_preview.npz werden daneben geschrieben. Keine
Normalisierung/Resampling oder Modellausfuehrung. Nach Sichtung entscheiden wir
ueber den gezielt isolierten Kanal und den spaeteren Coral-Vergleich.

## Sicherung und Tests

Aufnahmen liegen unter data/ und werden von checkpoint_radioml.py beim naechsten
Lauf mitgesichert. Das vergroessert das Archiv. Kein automatischer Upload.

```bat
python -m unittest test_rtl_capture test_radioml_holdout test_checkpoint_radioml test_radioml_iq test_train_radioml -v
```

Elf Tests bestanden: u.a. cu8-Kanalfolge, synthetischer Ton/FFT, Hashfehler,
Recorderkommando und unvollstaendige Aufnahme mit simuliertem Recorder.
Kein echter Stick/DLL-Test hier ausgefuehrt.

Referenzen fuer CLI-Parameter und Samplezaehlung:
https://github.com/osmocom/rtl-sdr/blob/master/src/rtl_sdr.c
https://github.com/osmocom/rtl-sdr/blob/master/debian/rtl_sdr.1

## Schmalbandkanal ueber die gesamte Minute auswerten

Der Nutzer meldet fuer die Aufnahme rtl-20261022-171509-339614 einen hoerbaren
Vortrag. Datenpruefung: 61440000 Samples, 60 s nominell, RMS 0,0345851,
ADC-Randwertanteil 0. Das ist ein menschlicher Referenzhinweis, noch kein
Coral-Ergebnis. Vom Nutzer angegebener Relaisausgang 438,800 MHz, Kanal 12,5 kHz.
Standort/ERP und Relaiskennung wurden im Projekt nicht unabhaengig geprueft.

```bat
git pull --ff-only
conda activate coral
python analyze_rtl_channel.py "data\recordings\rtl-20261022-171509-339614" --frequency 438800000
```

Die Rohaufnahme bleibt erhalten. Neuer Unterordner channel-analysis-v1:
channel.cf32 (Little-Endian complex64, I/Q), channel_report.json,
power_timeline.csv, channel_spectrum.npz und CHANNEL_COMPLETE.
Fuer eine Wiederholung neuen --output-Ordner angeben.

Pipeline: cu8 dekodieren, optionaler Frequenzversatz zur Kanalmitte,
1025-Tap-Hamming-Sinc-FIR mit 6250 Hz Grenzfrequenz, Dezimierung 32,
Ausgang 32000 Samples/s. Ganze Aufnahme blockweise mit fortgefuehrtem Filter-
zustand und Dezimierungsphase. Endlicher Filter mit Uebergangsbereich, keine
ideale rechteckige 12,5-kHz-Kanalmaske. Gruppenlaufzeit 0,5 ms; kein Ausgleich,
kein Auffuellen am Ende. Anfanglicher Einschwingabschnitt wird fuer Statistiken
verworfen. 20-ms-Pegelwerte ueber alle vollstaendigen Folgefenster und gemitteltes
Spektrum ueber alle vollstaendigen 1024-Sample-Fenster.

Pegel sind relative digitale Leistungen, keine dBm. Schwelle P20+6 dB ist eine
explorative Heuristik, weder SNR noch Sprecherkennung oder sicherer Aktivitaets-
nachweis. Bei durchgehendem Sender kann der Anteil oberhalb der Schwelle null
sein. Ein Gleichanteil bei der Kanalmitte wird absichtlich nicht entfernt,
um ein dort liegendes Nutzsignal nicht pauschal zu unterdruecken; DC kann die
Messung beeinflussen. Keine RMS-Normalisierung und noch keine Klassifikation.
Die RadioML-Zeitskalierung ist noch nicht an reale NFM-Aufnahmen angeglichen.

Drei neue Tests pruefen FFT-Blockfilter gegen direkte Faltung einschliesslich
Mischung und nicht ausgerichteten Blockgrenzen, Filterdurchlass/Sperrbereich
sowie eine vollstaendige synthetische cu8-Aufnahme bis zum Kanalbericht.
Die echte 60-Sekunden-Datei steht hier nicht zur Verfuegung.

```bat
python -m unittest test_rtl_channel -v
```

## FM-Hoerprobe des gefilterten Kanals

Nach erfolgreicher Kanalanalyse, Anaconda Prompt:

```bat
git pull --ff-only
conda activate coral
python demodulate_rtl_fm.py "data\recordings\rtl-20261022-171509-339614\channel-analysis-v1"
start "" "data\recordings\rtl-20261022-171509-339614\channel-analysis-v1\fm-listening-v1\fm_audio.wav"
```

Neuer Ordner fm-listening-v1 mit fm_audio.wav und audio_report.json.
Keine neuen Pakete notwendig (NumPy und Python-Standardbibliothek).
Vorhandener Ausgabeordner wird abgelehnt; Wiederholung mit neuem --output.

Phasendifferenz benachbarter Samples liefert den FM-Diskriminatorwert.
Sein Median wird entfernt, danach ein 257-Tap-FIR-Sprachband 300..3000 Hz.
Mono-PCM16 bei nominell 32 kHz, gesamte Datei gemeinsam auf 90 % Spitzenpegel
skaliert. Keine Deemphasis und keine Rauschsperre. Kanalfilter-Einschwingabschnitt
und Raender des Audiofilters fehlen in der WAV; sie ist geringfuegig kuerzer.
Der JSON-Bericht dokumentiert Quellhash, Verarbeitung und Zeitversatz.
Zeitangabe bezieht sich auf channel.cf32; zusaetzlich dessen RF-Filterlaufzeit
von 0,5 ms beachten, wenn mit der Rohaufnahme verglichen wird.

Ziel ist ausschliesslich, die Verstaendlichkeit des vom Nutzer gehoerten
Vortrags nach unserer Kanalfilterung zu pruefen. Rauschen kann ebenfalls
lautes Audio liefern. Weder erfolgreich erzeugte WAV noch Phasendifferenzen
allein sind eine Modulationsklassifikation. Entfernten Median nicht als
kalibrierten Senderfrequenzfehler interpretieren. Hoerprobe mit normaler
Wiedergabelautstaerke beginnen. Keine automatische Wiedergabe im Skript.

```bat
python -m unittest test_rtl_fm test_rtl_channel -v
```

Sechs Tests bestanden, darunter bekannter FM-Testton samt Frequenzoffset,
Audiobandfilter und WAV-Format/Tonfrequenz. Echte Aufnahme hier nicht verfuegbar;
Hoerpruefung muss am Windows-Rechner erfolgen. Danach erst experimenteller
Vergleich realer I/Q-Bloecke mit dem RadioML-Modell.

## Experimenteller Coral-Test der echten Aufnahme

Die FM-Hoerprobe der Aufnahme rtl-20261022-171509-339614 wurde vom Nutzer
als gut verstaendlich bestaetigt. Damit ist die Empfangs-/Filter-/Demodulations-
kette praktisch geprueft. Das ist noch keine Bestaetigung des RadioML-Modells.

```bat
git pull --ff-only
conda activate coral
python classify_rtl_channel.py "data\recordings\rtl-20261022-171509-339614\channel-analysis-v1"
```

Verwendet das vorhandene Modell unter models/radioml2018-iq-v1/compiled.
Keine Neukompilierung, kein Training, keine Aenderung der Aufnahme.
Start nach 33 Samples Einschwingabschnitt; nicht ueberlappende 1024-Sample-
Bloecke entsprechen nominell 32 ms bei 32 kS/s. Bei 1920000 Samples entstehen
1874 Bloecke. Rest am Ende wird dokumentiert und verworfen. Alle Bloecke
werden ausgewertet, keine Aktivitaetsselektion oder Auswahl nach Ergebnis.
Jeder Block wird auf komplexes RMS 0,5 normalisiert und mit I/Q-Kanalfolge
sowie Modellskala/Nullpunkt quantisiert. Exakte Nullbloecke fuehren zum Abbruch.
Kein weiterer Frequenz-/DC-Abzug und kein zusaetzliches Resampling.

Neuer Unterordner coral-real-v1:

- predictions.csv: Zeit bezogen auf channel.cf32, RMS vor Normalisierung,
  Vorhersage, alle fuenf dequantisierten Bewertungen, invoke-Zeit und
  Anteil beim Eingang geclippter Quantisierungswerte.
- report.json: Anzahl/Anteile pro Klasse, FM-Referenzuebereinstimmung,
  Modell-/Kanal-SHA256, Metadaten, Laufzeit und Quantisierungs-Clipping.
- CLASSIFICATION_COMPLETE: erst nach vollstaendigem Lauf.

Vorhandene Ergebnisordner werden abgelehnt, fuer Wiederholung --output nutzen.
--expected FM ist die Standard-Hoerreferenz. Es gibt keine einzeln annotierten
Sollklassen je Block. Deshalb keine Konfusionsmatrix oder Accuracy-Aussage.
Die Klasse noise/unknown fehlt: Auch Rauschen und unbekannte Signale werden
einer der fuenf Klassen zugeordnet. Bewertungen sind nicht kalibriert.
Benachbarte Bloecke sind abhaengig. RadioML und echter NFM-Empfang unterscheiden
sich in Signalstatistik, Kanal und zeitlicher Skalierung. Das erste Ergebnis
unveraendert dokumentieren, nicht durch ergebnisgetriebene Vorverarbeitung
optimieren und dann als unabhaengigen Test ausgeben.

```bat
python -m unittest test_rtl_classify test_rtl_fm test_rtl_channel -v
```

Acht Tests bestanden. Klassifikationstests verwenden einen simulierten
Interpreter und pruefen RMS/Kanalfolge, Warmup, Quantisierung, Blockzeiten,
CSV/JSON und Ueberschreibschutz. Kein echter Coral-Lauf hier. Nach Auswertung
mit checkpoint_radioml.py erneut sichern; Aufnahmen und Ergebnisse unter data/
werden mit archiviert.
