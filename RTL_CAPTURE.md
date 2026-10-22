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
