# Externe Windows-RTL-Aufnahme in die vorhandene Auswertung uebernehmen

`rtl_sdr.exe` erzeugt ohne Projektskript nur eine rohe cu8-Datei.
Der Import kopiert diese unveraendert in einen neuen Aufnahmeordner, schreibt
die fuer vorhandene Inspektions-/Kanalprogramme benoetigten Metadaten und
prueft Original/Kopie mit SHA256. Das Original bleibt erhalten.

Beispiel in Ubuntu/WSL, aus `~/rf-learning`, fuer die angegebene Windows-Aufnahme:

```bash
python project/import_rtl_capture.py /mnt/c/Users/DEIN-BENUTZER/Downloads/rtl_433675_60s_01.cu8 --frequency 433675000 --output data/recordings/rtl-433675-20261029-01 --inspect
```

Dateipfad anpassen, falls ein anderer Windows-Benutzer oder Speicherort verwendet
wurde. Vorgaben:1024000 Samples/s,60 Sekunden,20.7dB angeforderter Gain;
optional `--sample-rate`, `--seconds`, `--gain` gemaess tatsaechlichem Aufnahmebefehl.
122880000 Bytes entsprechen bei diesen angegebenen Einstellungen60 Sekunden.
Dateilaenge allein beweist weder durchgehenden Empfang noch korrekte Tunerparameter.
Das Importprotokoll kennzeichnet die Einstellungen deshalb als vom Nutzer geliefert;
ein fehlendes Recorderlog oder eine unbekannte Modulation wird nicht erfunden.

`--inspect` fuehrt direkt die vorhandene Roh-IQ-Qualitaetspruefung aus.
Die Ausgabe zeigt insbesondere ADC-Randwertanteil, RMS und eine begrenzte
Spektralvorschau. Das ist kein Modulations- oder Aussendungsnachweis.
Die Referenz bleibt `unknown`, bis eine entsprechende Aufnahmepruefung vorliegt.
Bei starker Uebersteuerung ist neu aufzunehmen; keine Normalisierung als Reparatur.

Bestehende Ausgabeordner werden nicht ueberschrieben. Erfolgreiche Importdaten
sind danach kompatibel mit `inspect_rtl_iq.py`, `analyze_rtl_channel.py` und
der folgenden Hoer-/Klassifikationspruefung. Kanalwahl und Gain muessen anhand
der tatsaechlichen Aufnahme beurteilt werden.

Tests: `python -m unittest test_import_rtl_capture`.
