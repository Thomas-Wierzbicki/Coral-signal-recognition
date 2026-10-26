# SE-Skalierung mit zwei weiteren Anfangszuständen prüfen

Stand: 26.10.2026. Plan vor Ausführung:
[SE_GAIN_REPLICATION_PLAN_V1.json](SE_GAIN_REPLICATION_PLAN_V1.json).

## Ausgangsbefund

Explorativer Seed 2070: Faktor 1 erreichte 88/120 (73,3 %), Faktor 2 erreichte
104/120 (86,7 %) nach jeweils 400 Full-Batch-Schritten bei LR 0,0001. Der Verlust
fiel von 0,646 auf 0,360. Erstmals >=50 % Wiedererkennung bei Schritt 147 bzw. 99.
Faktor 1 reproduziert die vorherige Kontrolle exakt. Prüfsummen und neu berechnete
Matrizen waren konsistent. Dokumentiert in
[reports/se_gain_exploratory_summary.json](reports/se_gain_exploratory_summary.json).

Dies betrifft dieselben wiederholt trainierten Beispiele, keine Testgenauigkeit.
Ein Vorteil mit einem Anfangszustand reicht für eine Übernahme der Variante nicht.

## Festgelegte Wiederholung

Zwei neue Modell-Seeds: **2071 und 2072**, je Faktor 1 gegen Faktor 2. Das ergibt
vier neue Trainingsläufe. Alle verwenden dieselben 120 ausgewählten Beispiele
(zehn je Klasse), Auswahl- und Batchreihenfolge-Seed bleiben 2070. Nur die
Anfangsgewichte unterscheiden sich zwischen Seeds; innerhalb eines Paares sind
sie identisch. Die erwarteten Gewichtshashes wurden vorab im Plan festgelegt.

Weitere Bedingungen unverändert: alle zwölf Klassen, LR 0,0001, frischer Adam mit
eps1e-8, Full-Batch 120, 400 Schritte, TensorFlow-kompatible SELU, kein Clipping,
Scheduler oder Early Stopping. Pro Lauf 48.000 verarbeitete Beispiele.
Validierung und Test bleiben geschlossen. Historisches FP32-Gate unverändert.

Primär werden die gepaarten Differenzen von Endverlust und End-Wiedererkennung
für jeden neuen Seed ausgewertet. Nur wenn Faktor 2 in **beiden** neuen Seeds
streng kleineren Verlust **und** höhere Wiedererkennung erreicht, wird das als
übereinstimmende Richtung der Replikation bezeichnet. Andernfalls bleibt der
Befund gemischt/nicht repliziert. Keine Seeds ersetzen oder schlechte Läufe
verwerfen. Seed 2070 wird als explorativer Befund getrennt behalten.

Zwei Anfangszustände auf demselben Trainingsausschnitt liefern weder statistische
Signifikanz noch Robustheit gegenüber neuen Daten oder reale Erkennungsleistung.
Gain 2 wirkt im gesamten Training; reine Initialisierungseffekte sind nicht isoliert.

## Auf Schokobaer starten

```bash
cd ~/rf-learning/project
git pull --ff-only
cd ~/rf-learning
source .venv/bin/activate
python project/replicate_se_gain.py
```

Keine Paketinstallation. Für diesen Versuch keine Parameter zur Änderung von
Epochen oder Beispielzahl verwenden. Das Skript durchläuft automatisch beide Seeds
und beide Gain-Werte. Die Trainingsroutine wurde für ausdrücklich übergebene
Modell-Seeds erweitert; ihr bisheriger Standardlauf bleibt Seed 2070 mit dem
ursprünglichen Plan. Das Modell selbst und seine historischen Standards bleiben
unverändert; die experimentelle Gain-Variante wird explizit ausgewählt.

## Ergebnisse und Abbruch

Neuer Ordner `models/se-gain-replication-ZEITSTEMPEL` mit Haupt-`protocol.json`
und nach jedem abgeschlossenen Seed einer Haupt-`report.json`. Unterordner
`seed2071` und `seed2072` enthalten die Auswahl, Anfangsgewichte, Verläufe und
Checkpoints beider Varianten. Die Auswahlhashes beider Seeds müssen gleich sein.
Das Gain-Metadatum bleibt wie zuvor in jedem Checkpoint erhalten.

`SE_GAIN_REPLICATION_COMPLETE` erscheint erst nach allen vier Läufen. Bei Fehler
bleiben Teilstände erhalten; kein automatisches Resume, kein Überschreiben und
kein automatischer Ersatz-Seed. Ein Neustart erzeugt einen neuen Ordner.
Der Hauptbericht ist bei nur einem abgeschlossenen Seed ausdrücklich unvollständig;
eine abgebrochene erste Seed-Ausführung kann nur einen Bericht im Unterordner haben.

Anschließend bitte die **report.json direkt aus dem Hauptordner** hochladen.
Sie enthält beide neuen Seed-Berichte und die gepaarten Unterschiede. Die bisherigen
Modelle und Ausgaben bleiben erhalten. Kein Upload durch das Skript.

Prüfung vor Veröffentlichung: 13 automatisierte Tests bestanden unter Python 3.12,
PyTorch 2.9.1+cpu und NumPy 2.5.2. Darunter vier tatsächlich ausgeführte kurze
Optimierungsfälle für beide neuen Seeds, unterschiedliche Gewichtshashes zwischen
Seeds, gleiche Gewichte innerhalb jedes Paares, Training-only-Zugriff, korrektes
Ergebnis bei gemischter Richtung, Checkpoint-/Ausgabeschutz sowie die bisherigen
Modell-/Gradiententests. Kein 400-Schritte-Lauf auf Nutzerdaten/GPU ausgeführt.
