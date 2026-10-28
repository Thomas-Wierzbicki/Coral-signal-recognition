# Fehleraudit nach Klasse, SNR und gespeicherten Signalparametern

Stand28.10.2026. Vorabplan:
[MODULATION_ERROR_AUDIT_PLAN_V1.json](MODULATION_ERROR_AUDIT_PLAN_V1.json).

## Jetzt gezielt auswerten, nicht erneut trainieren

Alle vier bereits nach Validierungsverlust ausgewählten Checkpoints des
Fortsetzungsvergleichs werden untersucht: Seeds2081/2082, jeweils `keep_lr` und
`lower_lr`. Keine Auswahl eines bevorzugten Seeds oder Checkpoints anhand der
neuen Audittabellen. Es gibt keine Gewichtsänderungen und keinen neuen Lerndurchlauf.

Das Skript reproduziert die Trainings- und Validierungsmatrizen sowie die Verluste,
verknüpft jede Vorhersage mit ihrer vorhandenen JSONL-Parameterzeile und prüft
Quelldaten, IDs, Klassen, SNR, Störparameter-Gruppen und den Hash des tatsächlichen
komplexen I/Q-Blocks. Die ursprüngliche Entwicklungs-/Fortsetzungskette und deren
Modell-/Dateihashes müssen passen. Fehlende oder widersprüchliche Daten führen zu
einem Fehler; es werden keine Signale oder Parameter stillschweigend neu erzeugt.

## Start auf Schokobaer

```bash
cd ~/rf-learning/project
git pull --ff-only
cd ~/rf-learning
source .venv/bin/activate
python project/audit_modulation_errors.py
```

Keine Installation zusätzlicher Pakete. Bei genau einem abgeschlossenen
Fortsetzungsordner wird dieser genommen. Bei mehreren muss der tatsächliche Ordner
angegeben werden: `--run models/se-gain-continuation-ZEITSTEMPEL`.
Wenn die ursprünglichen Entwicklungsdateien verschoben wurden, kann ihr neuer
Ordner mit `--development-source` angegeben werden; Prüfsummen bleiben verbindlich.

## Was ausgewertet wird

Getrennt für jeden der vier Checkpoints und für train/validation:

- Klassenverwechslungen je SNR;
- angeforderte Bandbreite: 1500–4000, 4000–8000, 8000–12000Hz;
- Betrag des Frequenzversatzes: 0–1600, 1600–3200, 3200–4800Hz;
- tatsächlich protokollierte rauschfreie 90%-Leistungsbandbreite;
- Klassenwerte bei SNR>=15dB sowie FM-/2FSK-/GFSK-/GMSK-Verwechslungen;
- zusätzlich feineres Klasse×SNR×Bandbreite-Raster dieser vier Kategorien.

Bin-Intervalle schließen ihre Untergrenze ein, die Obergrenze erst im letzten Bin.
Fehlende oder nicht anwendbare Werte werden ausdrücklich bezeichnet. Noise hat
keinen Signal-SNR, keine Trägerbandbreite und keinen Frequenzversatz. Alle zwölf
Kategorien bleiben in den allgemeinen Tabellen erhalten.

Jede Zelle zeigt Anzahl, richtige Zuordnungen, vorhergesagte Klassen und Anzahl
verschiedener Störparameter-Gruppen. Unter20 Beispielen wird `sparse=true` gesetzt.
Das feinere Raster hat erwartbar häufig kleine Fallzahlen. Vier Modelle auf denselben
Beispielen sind keine vier unabhängigen Datensätze. Die Tabellen erlauben daher
keine isolierten Kausalaussagen oder Signifikanzbehauptungen.

Der maximale Softmax-Wert wird als **unkalibrierte Modellbewertung** protokolliert,
nicht als garantierte Wahrscheinlichkeit. Fehler mit Bewertung>=0,8 und höchstens
drei hoch bewertete Fehlerbeispiele je Klasse werden rein beschreibend ausgewiesen.
Alle Einzelvorhersagen bleiben zusätzlich in CSV-Dateien erhalten.

## Wichtige Grenze der vorhandenen Generatorprotokolle

Ausgewertet werden nur tatsächlich gespeicherte Größen. Bei den FSK-Generatoren
wurden interne Zufallsziehungen für Modulationsindex, Puls-/BT-Parameter durch den
bisherigen Wrapper nicht vollständig mitgeschrieben. Diese Werte dürfen nicht aus
dem Klassennamen geraten werden. Der Bericht zählt, welche Werte vorhanden sind,
und zeigt die Lücke ausdrücklich. Signal-/Rausch-Seeds in den Einzelzeilen können
bei einem später separat geplanten und hashgeprüften Generator-Replay helfen;
dieser Audit führt kein solches Replay aus.

Angeforderte Bandbreite ist nicht gleich gemessener Bandbreite. Gemessene Breite mit
Rauschen ist ebenfalls keine reine Modulationsbandbreite. Der zufällige Roll-off-
Parameter wurde nur an die Konstellationsgeneratoren übergeben; für FM/FSK/AM/OFDM
wird er hier nicht als wirksame Stellgröße interpretiert.

## Ergebnis und nächste Entscheidung

Neuer Unterordner im Fortsetzungsordner: `error-audit-UTCZEITSTEMPEL`.
Enthalten sind `protocol.json`, ein gemeinsamer **report.json** und acht CSV-Dateien
für sämtliche Einzelvorhersagen (vier Modelle×zwei Datensplits). Die Dateien enthalten
auch Seed-/Gruppenbezüge, protokollierte Parameter und gemessene Signalmerkmale.

**Danach nur report.json aus diesem Audit-Unterordner hochladen.**
`ERROR_AUDIT_COMPLETE` erscheint erst nach allen vier Checkpoints. Bei einem Fehler
bleiben unvollständige Ausgaben ohne Abschlussmarker erhalten. Vorhandene Ausgaben,
Modelle und Originaldaten werden nicht überschrieben.

Erst anhand wiederkehrender Fehlermuster wird ein weiterer kontrollierter Eingriff
oder Generator-Replay geplant. Kein automatisches Training, Entfernen von Klassen
oder Auswählen besonders günstiger Modelle. Entwicklungsvalidierung bleibt
wiederverwendete Entwicklungsdaten; `test.npz` wird nicht geöffnet. Keine unabhängige
Test-, Realempfangs-, Coral- oder Drohnenerkennungsleistung aus diesem Audit.
Historische FP32-Gates unverändert.

## Technische Prüfung

Neun gezielte Softwaretests bestanden unter Python3.12 / PyTorch2.9.1+cpu /
NumPy2.5.2. Drei neue Tests prüfen Binkanten, vollständige Metadatenverknüpfung,
Manipulation am I/Q-Block, falsche Labels, doppelte Parameter-IDs, Rauschbehandlung,
Fallzahlen und fehlende interne FSK-Parameter. Ein synthetischer Dateitest lädt vier
Checkpoints, reproduziert beide Datensplits, schreibt die acht CSV-Dateien und
bestätigt unveränderte Quellmodelle und Ausgabeschutz. Die bereits geprüfte Quell-
Ladekette wird in diesem I/O-Test durch eine synthetische Fixture ersetzt; im echten
Lauf bleibt ihre vollständige Prüfung aktiv. Weitere Modell-/Lerntests ergänzen dies.
Die eigentliche Auswertung auf Nutzer-Checkpoints steht noch aus.
