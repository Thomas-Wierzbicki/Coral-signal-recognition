# RadioML-Stand sichern und wiederherstellen

## Sicherung ausfuehren

Anaconda Prompt, Projektordner:

```bat
cd /d C:\Users\thwie\source\repos\Coral-signal-recognition
git pull --ff-only
conda activate coral
python checkpoint_radioml.py
```

Git wird nur zum Aktualisieren verwendet; die Sicherung selbst benoetigt kein Git.
Vorher Training/Tests beenden, die Dateien veraendern. Es wird kein Hardwaretest
und kein Upload gestartet. Erfolgsnachricht abwarten. Nur dann ist die ZIP
abgeschlossen. Diese Dokumentation ist kein Nachweis, dass der Nutzer die
Sicherung bereits ausgefuehrt hat.

Standardziel ausserhalb des Projekts:
`C:\Users\thwie\source\repos\Coral-signal-recognition-backups`.
Neue ZIP mit UTC-Zeitstempel; vorhandene Sicherungen bleiben erhalten.
Jede Nutzdatei wird nach Archivierung erneut gelesen und SHA256-geprueft.
Unvollstaendige Archive enden auf .zip.partial und gelten nicht als Sicherung.

| Enthalten | Umfang |
|---|---|
| project/ | Code, Markdown, Text, JSON und Konfiguration im Projektstamm |
| project/data/ | Alle vorhandenen Datenvarianten und Metadaten rekursiv |
| project/models/ | Alle vorhandenen Modellvarianten, Manifeste, Berichte, Compilerlogs und Vorhersagen rekursiv |
| environment.json | Aktives Python samt Paketversionen und Sicherungsumfang |
| SHA256SUMS.json | Hashliste aller archivierten Nutzdateien |
| RESTORE.txt | Wiederherstellungshinweise |

Das Skript fordert I/Q-Float-/INT8-/TPU-Modell, JSON-Berichte, Vorhersagen und
beide RadioML-Datensatzvarianten an. Die alte Drei-Klassen-Linie und das
Spektrogramm-Modell werden mitgesichert, soweit lokal vorhanden.
Keine Git-Historie, keine installierten Programme/Treiber und keine beliebigen
weiteren Unterordner. Root-Dateien mit anderen Endungen sind nicht automatisch
erfasst. Python-Paketliste betrifft die aktive Umgebung; fuer die anderen
Umgebungen die Requirements und README verwenden.

## Original-HDF5 ebenfalls sichern

Die grosse Originaldatei unter Downloads ist standardmaessig NICHT enthalten.
Die ausgewaehlten Trainings-/Test-NPZs sind enthalten. Fuer spaetere neue Tests
Original und Klassen-/Lizenzdateien separat behalten. Optional einschliessen:

```bat
python checkpoint_radioml.py --include-source "C:\Users\thwie\Downloads\GOLD_XYZ_OSC.0001_1024.hdf5\GOLD_XYZ_OSC.0001_1024.hdf5"
```

Dann liegt das Original unter external-source/ im Archiv. Es braucht entsprechend
viel Platz und Zeit. Zusaetzliche Lizenzdateien vorher in den Projektstamm kopieren
(z.B. LICENSE.TXT), damit sie erfasst werden. Das Archiv aendert keine Nutzungsrechte.

## Zweiter Datentraeger und Pruefung

```bat
python checkpoint_radioml.py --output D:\Coral-Backups
python checkpoint_radioml.py --verify "D:\Coral-Backups\coral-radioml-ZEITSTEMPEL.zip"
```

Echten ZIP-Dateinamen einsetzen. Eine Sicherung auf derselben Platte schuetzt
nicht vor deren Ausfall. Fuer eine zweite Kopie die fertig gepruefte ZIP auf
einen anderen Datentraeger kopieren und dort erneut --verify ausfuehren.
SHA256 dient der Integritaetspruefung, nicht dem Nachweis einer vertrauenswuerdigen Herkunft.

## Wiederherstellung

ZIP pruefen, in einen neuen Ordner entpacken. project/ ist der wiederhergestellte
Projektstand. Modelle mit zugehoerigen preprocessing.json, labels.txt und
Berichten im jeweiligen Ordner belassen. Conda, Edge-TPU-Laufzeit und Compiler
nach README separat einrichten. In project/ mit conda activate coral und
python evaluate_radioml_iq_coral.py die Hardwareauswertung wiederholen.
Alte Daten nicht ungeprueft ueberschreiben. Git-Historie bei Bedarf gesondert
vom Repository holen; sie ist nicht im neuen Archiv enthalten.

## Geprueft

```bat
python -m unittest test_checkpoint_radioml test_radioml_iq test_train_radioml -v
```

Sieben Tests bestanden, darunter Archiv-Roundtrip, JSON-/Modellbytes,
optionale Quelldatei, Erkennung veraenderter Nutzdateien und Ablehnung eines
Sicherungsziels innerhalb des Projekts. Die echte Windows-Sicherung muss
auf dem Rechner mit den Modellen und Daten ausgefuehrt werden.
