# WSL-Projekt inklusive Modelle und Daten sichern

`checkpoint_rf_learning.py` sichert reguläre Dateien unter `~/rf-learning`,
einschließlich `project/`, `data/` und `models/`. Installierte Python-Umgebungen,
Git-Verwaltungsdaten und Caches sind ausgeschlossen. Die aktive Python-Version,
installierte Paketversionen und der lokale Git-Commit werden dokumentiert.
Externe Dateien außerhalb dieses Baums, etwa ein Windows-HDF5-Original, sind nicht
enthalten. Das Skript überträgt keine Daten an GitHub oder einen anderen Dienst.

Training beenden. Während der Sicherung keine Dateien ändern. In WSL:

```bash
cd ~/rf-learning/project
git pull --ff-only
cd ~/rf-learning
source .venv/bin/activate
python project/checkpoint_rf_learning.py --output /mnt/c/Users/thwie/Downloads/Coral-signal-recognition-backups
```

Der Zielordner wird angelegt. Ein neues ZIP mit UTC-Zeitstempel liegt danach im
Windows-Downloads-Ordner unter `Coral-signal-recognition-backups`, außerhalb der
WSL-Projektdateien. Quellcode, Rohdaten, Trainingsauswahl, Modelle, Optimiererzustände,
Berichte und Dokumentation bleiben im Archiv zusammen. Vorhandene Sicherungen
werden nicht überschrieben. Eine `.zip.sha256`-Datei enthält den Hash des ganzen ZIP.

Jede Nutzdatei wird während des Kopierens gehasht und anschließend aus dem Archiv
zurückgelesen/geprüft. Das Skript prüft außerdem Änderungen am Quellbaum während
der Sicherung. Nur nach erfolgreicher Prüfung wird `.zip.partial` zu `.zip`.
Bei Fehler bleibt ein Teilarchiv zurück; es ist keine bestätigte Sicherung.

Spätere Prüfung:

```bash
python project/checkpoint_rf_learning.py --verify "/mnt/c/Users/thwie/Downloads/Coral-signal-recognition-backups/DATEINAME.zip"
```

Zur Wiederherstellung in einen neuen leeren Ordner entpacken, zunächst nichts
überschreiben. `RESTORE.txt`, `environment.json` und `SHA256SUMS.json` beschreiben
Inhalt und Prüfung. Die Python-Umgebung und Treiber müssen erneut installiert
werden; das Archiv ist keine vollständige Systemabbildung. Modellzustände nur
mit ihren passenden JSON-/Vorverarbeitungs-/Gain-Einstellungen laden. Es erfolgt
kein automatisches Training-Resume.

Die Softwaretests prüfen einen vollständigen Sicherungs-/Prüflauf, die ZIP-Prüfsumme,
ausgeschlossene Umgebungen, unerwartete Archivdateien und eine während der Prüfung
veränderte Quelle. Eine Sicherung des echten Rechners ist erst nach dortiger
Ausführung und Erfolgsmeldung bestätigt.
