# Praktischer Test auf der vorhandenen realen FM-Aufnahme

Ziel dieses Schritts: pruefen, ob der eingefrorene klassische Klassifikator
auf dem bereits aufgenommenen realen Kanal brauchbare FM-Ausgaben liefert.
Keine weitere synthetische Optimierung vor diesem Test. Der Plan steht in
`CLASSICAL_REAL_CHANNEL_PLAN_V1.json`.

## Starten

Auf dem Trainingsrechner in Ubuntu, aus `~/rf-learning`:

```bash
python project/classify_classical_real_channel.py
```

Die konkrete bekannte Aufnahme wird an folgenden Stellen gesucht:

- Unter `data/recordings/rtl-20261022-171509-339614/channel-analysis-v1`.
- Im vorhandenen Windows-Projekt unter `source/repos/Coral-signal-recognition`
  im eigenen Benutzerprofil bzw. dessen WSL-Mount unter `/mnt/c/Users`.

Falls sie auf einem anderen Rechner liegt: aus ihrem `channel-analysis-v1`-Ordner
**channel.cf32**, **channel_report.json** und **CHANNEL_COMPLETE** gemeinsam in
den oben genannten relativen Ordner unter `~/rf-learning` kopieren.
Nicht nur eine WAV-Datei kopieren; erforderlich ist der bereits gefilterte
komplexe32kS/s-IQ-Kanal. Das grosse rohe RTL-Capture ist fuer diesen Schritt nicht noetig.
Die Dateien liegen hier im Chat nicht vor; ein realer Erfolg ist vor dem Lauf offen.

Alternativ den konkreten Ordner angeben:

```bash
python project/classify_classical_real_channel.py --channel /DEIN/PFAD/channel-analysis-v1
```

Es wird genau eine abgeschlossene `models/classical-modulation-...`-Baseline
erwartet; sonst `--source models/DEINE-KLASSISCHE-BASELINE` angeben.
Optionale Datenquelle: `--data data/modulation12-v1`.
Ausgabe in einem neuen `models/classical-real-channel-ZEITSTEMPEL`-Ordner.
Bitte anschliessend **report.json** hochladen. `predictions.csv` enthaelt
Zeitpunkte, Eingangs-RMS und Klassenstimmen fuer jeden Block.

## Festes Vorgehen

Modell und urspruenglicher Trainingsscaler werden per Hash und Quelldateien
geprueft. Kein Fit auf der Aufnahme. Der Eingang muss32000S/s und little-endian
complex64 sein. Der vorhandene Einschwingbereich wird uebersprungen, danach
4096-Sample-Fenster ohne Ueberlappung, entsprechend128ms je Block.
Keine Frequenzkorrektur, DC-Entfernung, Resampling oder nachtraegliche Filterung.
Nur die im Modell vorgesehene Block-RMS0.5-Normierung und35 feste Merkmale.
Alle zwoelf Klassen bleiben moegliche Ausgaben.

Das Standard-Referenzlabel `fm` bezieht sich auf die vom Nutzer gehoerte
438,8MHz-Aufnahme. Bei anderen Aufnahmen bewusst `--expected LABEL` oder
`--expected unlabeled` verwenden. Der Vergleich ist eine Hoerreferenz fuer die
Aufnahme und keine gesicherte Wahrheit jedes einzelnen128ms-Blocks.

Null-/nichtendliche Bloecke werden mit eigenem Status protokolliert, ohne
Vorhersage. Anzahl bleibt sichtbar; bei keinem gueltigen Block bricht der Lauf ab.
Fuenf Nachbarstimmen sind keine95-/100-Prozent-Sicherheit. Die Methode erzwingt
eine bekannte Klasse und besitzt noch keine validierte Unbekannt-Erkennung.
Der Bericht nennt den Anteil passend zur Hoerreferenz daher **nicht Genauigkeit**.

Am Ende muessen Modellarrays/-datei, Kanaldatei und Metadaten unveraendert sein.
Erfolgsmarker erst danach. Fehlerbericht bleibt erhalten; alte Ergebnisse werden
nicht ueberschrieben. CPU/NumPy, kein Coral-Zugriff oder Sendebetrieb.

## Entscheidung

Ein Misserfolg auf dieser klaren Aufnahme bedeutet: den Live-Erkenner nicht auf
diesem Modell weiterbauen; zuerst reale Referenzdaten und die Empfangsaufbereitung
bearbeiten. Ein Erfolg rechtfertigt weitere reale Gegenproben, validiert aber
weder alle zwoelf Klassen noch Fehlalarmrate oder Drohnenprotokollerkennung.
Keine korrekt wirkende Ausgabe durch nachtraegliche Frequenzverschiebung erzwingen.

Fuer eine nutzbare Erkennung werden anschliessend reale Referenzaufnahmen aus
getrennten Sitzungen/Geraeten und Negativbeispiele benoetigt. Die Klassen bleiben
erhalten; verfuegbare Daten bestimmen, welche realen Leistungsangaben belegt sind.

## Tests

```bash
python -m unittest test_classical_real_channel test_classical_baseline
```

Lokale Fixtures pruefen Kanalformat, ungueltige Bloecke, Zeit-/Tail-Zaehler,
Ganzzahlstimmen, Normierung, Modell-/Dateiunveraenderlichkeit, Pfadsuche und
Fehlerberichte. Sie ersetzen den echten Empfangstest nicht.
