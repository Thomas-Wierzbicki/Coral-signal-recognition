# Klassenreihenfolge und Dateiabgleich

Die Datei `radioml_classes.json` bildet Y-Spaltenindizes auf Namen ab.
Sie übernimmt die explizit als korrigiert dokumentierte Reihenfolge
`MODULATION_TYPES` aus RadioPile. Das ist eine dokumentierte Drittquelle,
keine nachträglich von DeepSig bezogene Begleitdatei.

Quellen, festgehaltener Commit fd6cb8a8fe824509121e3f85d8a7b153b0e9644a:

- https://github.com/xandervanparys/RadioPile/blob/fd6cb8a8fe824509121e3f85d8a7b153b0e9644a/datasets/rml2018_01a/config.py
- https://github.com/xandervanparys/RadioPile/blob/fd6cb8a8fe824509121e3f85d8a7b153b0e9644a/docs/RAW_INPUT_CHECKSUMS.txt
- Archivierte Veröffentlichung: https://zenodo.org/records/22877108

RadioPile nennt denselben Kaggle-Spiegel und dokumentiert für die verwendete
unverarbeitete Datei `GOLD_XYZ_OSC.0001_1024.hdf5` diese SHA-256:

```text
e3dd0bef66a3426959ee66a1709a8c0a95d4f8395d18aaf6f1214bdbc763bd38
```

## Aktuelle lokale Datei prüfen

In Anaconda Prompt oder PowerShell:

```text
certutil -hashfile "C:\Users\thwie\Downloads\GOLD_XYZ_OSC.0001_1024.hdf5\GOLD_XYZ_OSC.0001_1024.hdf5" SHA256
```

Dabei wird die etwa 21-GB-Datei vollständig gelesen; das kann dauern.
Eine Übereinstimmung bestätigt Bytegleichheit mit der von RadioPile erfassten
Referenzdatei. Sie ist kein unabhängiger Beweis für die physikalische Richtigkeit
aller im ursprünglichen Datensatz vergebenen Labels.
Bei abweichendem Hash vor dem Training Herkunft und Version untersuchen.

## Nach passendem Dateiabgleich

Im Projektordner mit aktivierter Umgebung radioml:

```text
python radioml_import.py "C:\Users\thwie\Downloads\GOLD_XYZ_OSC.0001_1024.hdf5\GOLD_XYZ_OSC.0001_1024.hdf5" --inspect --class-map radioml_classes.json
```

Anschließend Import der fünf Startklassen:

```text
python radioml_import.py "C:\Users\thwie\Downloads\GOLD_XYZ_OSC.0001_1024.hdf5\GOLD_XYZ_OSC.0001_1024.hdf5" --class-map radioml_classes.json
```

Die Reihenfolge in der Importauswahl lautet AM-DSB-WC, FM, BPSK, QPSK, 16QAM.
Dazu gehören die ursprünglichen Y-Spalten 19, 21, 3, 4 und 12.
Die neue Klasse 0 im importierten Datensatz ist deshalb AM-DSB-WC,
nicht die ursprüngliche Klasse 0 (OOK).
