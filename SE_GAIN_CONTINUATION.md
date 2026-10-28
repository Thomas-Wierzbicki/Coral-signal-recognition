# Optimiertes Vorgehen: kontrolliert fortsetzen statt neu starten

Stand 28.10.2026. Vor der Ausführung festgeschrieben:
[SE_GAIN_CONTINUATION_PLAN_V1.json](SE_GAIN_CONTINUATION_PLAN_V1.json).

## Eine begrenzte Entscheidung pro Versuch

Dieser Schritt untersucht ausschließlich die Lernrate bei der Fortsetzung. Keine
neue Architektur, neuen Daten oder zusätzlichen SE-Varianten. Vier Fortsetzungen:
Seeds2081/2082, jeweils bisherige LR0,0001 gegen reduzierte LR0,00001. Beide Zweige
eines Seeds beginnen am **gleichen letzten Gain2-Checkpoint bei Epoche100**,
einschließlich sämtlicher Adam-Momente, Schrittzähler und Batchgeneratorzustand.
Die bisherigen Dateien bleiben unverändert.

Die erste Etappe ist gegenüber dem zunächst diskutierten 100-Epochen-Vorschlag
auf **50 weitere Epochen (101–150)** begrenzt. Pro Fall 2650 zusätzliche Updates.
Keine automatische Verlängerung, kein Early Stopping, Scheduler, Clipping,
Augmentation oder nachträglicher Ersatz eines Seeds.

Der Vorteil der niedrigeren LR wird nur dann richtungsgleich bestätigt, wenn der
nach Validierungsverlust gewählte Checkpoint in **beiden** Seeds kleineren Verlust
und höheren Macro-F1 aufweist. Genauigkeit, Klassen-/SNR-Werte, Verlustsprünge und
Schwankungen der Klassentrefferraten ergänzen die Beurteilung. Geringere Schwankung
allein gilt nicht als bessere Erkennung. Ein gemischter Befund bleibt gemischt.

## Vor dem ersten Update prüfen

Das Skript nimmt nur einen abgeschlossenen Standardlauf aus
`models/se-gain-development-*`. Bei mehreren passenden Ordnern muss `--source`
explizit gesetzt werden; keine Auswahl des vermeintlich besten Laufs.

Es prüft Quelldateien, registrierten Entwicklungsplan, Daten- und Auswahlhashes,
Gruppentrennung, Anfangsgewichte, letzte Checkpoints und deren Verläufe. Die
ursprünglichen 1680 Trainingsbeispiele werden in exakt gleicher Reihenfolge
rekonstruiert. Die 1080 Validierungsbeispiele bleiben gleich. Beide Ausgangsmodelle
müssen vor jedem Training die vorherige Epoche100-Validierungsmatrix und ihren
Verlust reproduzieren. Unstimmigkeiten stoppen vor den Updates.

Der volle Adam-Zustand wird geladen; nur das LR-Feld wird im reduzierten Zweig
geändert. Der wiederhergestellte Modell-, Adam- und Generatorzustand wird gehasht
und innerhalb jedes Paars verglichen. Das ist eine Fortsetzung, kein neues Training
mit zufällig neu erzeugten Anfangsgewichten. Nach Konstruktion werden ausschließlich
die gespeicherten Tensoren und Zustände verwendet. Historische Hash-/FP32-Gates
werden nicht angepasst.

## Start auf Schokobaer

```bash
cd ~/rf-learning/project
git pull --ff-only
cd ~/rf-learning
source .venv/bin/activate
python project/continue_se_gain.py
```

Keine Paketinstallation auf dem Nutzerrechner nötig. Es werden vier Fortsetzungen
automatisch durchlaufen. Pro Epoche erscheinen Verlust und Zuordnungsrate. Bei
mehreren abgeschlossenen Entwicklungsordnern denselben Befehl mit dem tatsächlichen
Ordner ergänzen: `--source models/se-gain-development-ZEITSTEMPEL`.
Für den vorab festgelegten Vergleich keine Änderung von `--epochs`.

## Ein Bericht statt wiederholter manueller Dateisuche

Neuer Ordner `models/se-gain-continuation-UTCZEITSTEMPEL`, außerhalb des
Quellversuchs. Pro Seed/Zweig letzte und ausgewählte Checkpoints, Verlauf und
Fallbericht. Ein Haupt-`report.json` enthält automatisch:

- Integritäts-/Startprüfungen und Zustandspaarung;
- Endstände und nach Validierungsverlust gewählte Checkpoints;
- Trainings-, Validierungs-, Klassen- und Signal-SNR-Ergebnisse;
- Verlustsprünge, mittlere absolute Änderung der Klassentrefferraten in den letzten
  20 Übergängen, Klassen-Minima/Maxima der letzten20 Epochen und letzte Zehnerblöcke;
- gepaarte Unterschiede und die vorher festgelegte Entscheidung.

**Danach nur die report.json aus dem Hauptordner hochladen.** Weitere Verläufe sind
für die normale Auswertung nicht erforderlich. Originalverläufe bleiben trotzdem
für gezielte Nachprüfungen erhalten.

Die Checkpoint-Auswahl beginnt bei Epoche100 und umfasst die Fortsetzung101–150;
bei gleichem Verlust bleibt der früheste Checkpoint. Die ursprünglichen besten
Entwicklungscheckpoints vor Epoche100 bleiben erhalten, sind aber keine Kandidaten
innerhalb dieses kontrollierten Fortsetzungsvergleichs. Nach jedem Fall wird der
Hauptbericht aktualisiert. Erst `CONTINUATION_COMPLETE` bestätigt alle vier Fälle.
Keine Quellüberschreibung oder automatische Wiederholung; bei Abbruch bleiben
Teilstände ohne Gesamt-Abschlussmarker erhalten.

## Grenzen und weiterer Ablauf

Die vorhandene Validierung ist Entwicklungsdaten und wurde bereits zur Auswahl
verwendet. Der frühere Testsatz ist kein unberührter Abschlussprüfsatz; er wird
hier nicht geöffnet. Für einen abschließenden unabhängigen Nachweis sind später
neue, zuvor nicht ausgewertete Daten nötig. Alle zwölf Klassen bleiben, ebenso
das Ziel der späteren Drohnenprotokollerkennung. Keine Aussage zu Realempfang,
Coral-Einsatz oder unveränderter Literaturreproduktion aus diesem Vergleich.

Erst nach dem gemeinsamen Bericht wird entschieden: Budgetverlängerung,
weitergehende Diagnose oder Übernahme einer Einstellung. Kein ungesteuertes
Durchprobieren mehrerer Änderungen und keine Erfolgsmeldung allein anhand der
Gesamtquote.

## Technischer Prüfstand

Neun gezielte Tests bestanden unter Python3.12, PyTorch2.9.1+cpu und NumPy2.5.2:
bitgleiche Gewichte/Adam-Zustände/Batchgenerator nach unterbrochener gegenüber
durchgehender Optimierung, unveränderte Momente bei bloßem LR-Wechsel, Ablehnung
fehlender/resetter Momente, Abweichungen bei der Ausgangsvalidierung und veränderter
Checkpoint-Dateien. Ein vollständiger kurzer Ablauf führt alle vier Zweige mit
künstlichen Daten aus, prüft Gruppen-/Quellzuordnung, Paarung, zusätzliche Schritte,
Abschlussmarker und unveränderte Quelldateien. Weitere Modell-/Lerntests ergänzen dies.

Die I/O-Testdaten verwenden einen eigens registrierten synthetischen Testplan und
dessen Tensorhashes; der echte Produktionsplan samt historischen Hashes bleibt
unverändert. Die 50-Epochen-Fortsetzung auf Nutzer-Checkpoints/GPU ist noch nicht
ausgeführt. Der technische Test ist kein neuer Leistungsnachweis.
