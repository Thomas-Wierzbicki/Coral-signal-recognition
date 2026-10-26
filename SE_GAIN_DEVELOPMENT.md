# Größerer SE-Vergleich mit Entwicklungsvalidierung

Stand 26.10.2026. Vorab festgelegter Plan:
[SE_GAIN_DEVELOPMENT_PLAN_V1.json](SE_GAIN_DEVELOPMENT_PLAN_V1.json).

## Bisheriger Nachweis

Der Vorteil von Faktor 2 wurde auf denselben 120 Trainingsbeispielen in zwei neuen
Anfangszuständen richtungsgleich bestätigt: Seed 2071 72,5 →90,8 %, Seed 2072
75,8 →85,0 %. Beide Endverluste waren kleiner. Seed 2072 fiel allerdings vom
zwischenzeitlichen Maximum 96,7 % wieder zurück. Die primäre Auswertung verwendete
wie festgelegt den Endstand. Kein Nachweis unabhängiger Erkennungsleistung.
Zusammenfassung: [reports/se_gain_replication_summary.json](reports/se_gain_replication_summary.json).

## Neuer Vergleich

Vier Läufe: Seeds **2081 und 2082**, jeweils SE-Gain 1 gegen 2.

| Einstellung | Für alle vier Läufe |
|---|---|
| Trainingsausschnitt | 1.680 Beispiele, 140 je Klasse |
| SNR der Signale | −5, 0, 5, 10, 15, 20, 25 dB |
| Auswahl | 20 vollständige Störparameter-Gruppen pro SNR |
| Validierung | vorhandene 1.080 Validierungsbeispiele |
| Epochen | fest 100, kein Early Stopping |
| Mini-Batch | 32, letzter Batch 16; 53 Schritte je Epoche |
| Adam | LR 0,0001, eps1e-8, kein Scheduler oder Clipping |
| Checkpoint | kleinster mittlerer Validierungsverlust, bei Gleichstand frühester |

Alle zwölf Kategorien bleiben erhalten. Noise hat keinen Nutzsignal-SNR; es wird
mit der jeweils zugehörigen Störparameter-Gruppe ausgewählt. Jede Gruppe enthält
alle zwölf Kategorien. Die Auswahl hält somit Frequenzversatz, Bandbreitenparameter
und weitere gemeinsame Einstellungen zwischen Klassen gepaart. Ganze Gruppen
werden innerhalb des bisherigen Trainingssplits gewählt. Quell- und Gruppen-IDs
von train und validation müssen disjunkt sein. Datensatzhashes, Auswahl und die
vorab berechneten Anfangsgewichte werden geprüft/protokolliert.

Alle vier Fälle verwenden dieselbe Auswahl und dieselbe Batchreihenfolge. Die
Gewichte sind pro Seed gepaart. Explizite TensorFlow-kompatible SELU und explizite
experimentelle SE-Gain-Variante; historische Standardmodelle werden nicht geändert.
Gain wirkt weiterhin im gesamten Training. Initialisierungsursache nicht isoliert.

Der vorherige Lerntest nutzte nur gute SNR und 120 Beispiele im Full-Batch. Der
neue Versuch ist wegen größerer Datenmenge, aller SNR und Mini-Batches eine andere
Aufgabe. Seine Prozentwerte sind nicht direkt mit den Memorierungswerten vergleichbar.

## Validierung und Test sauber benennen

Die vorhandene Validierung wurde in früheren Entwicklungsversuchen bereits benutzt.
Sie dient hier der Checkpoint-Auswahl und dem Entwicklungsvergleich. Die primären
Werte sind Validierungsverlust und Macro-F1 des gewählten Checkpoints je Seed;
Genauigkeit, Klassenmatrizen und Signal-SNR-Auswertungen ergänzen die Diagnose.
Keine nachträgliche Auswahl nach höchster Genauigkeit und kein Entfernen ungünstiger
Seeds. Faktor 2 gilt nur dann als richtungsgleich besser, wenn beide primären
Kennzahlen in beiden Seeds besser sind. Andernfalls Befund gemischt.

`test.npz` wird nicht geöffnet. Der vorhandene Testsatz wurde aber in früheren
Projektphasen bereits ausgewertet. Er ist kein unberührter Abschlussprüfsatz.
Vor einem unabhängigen Abschlussnachweis sind neue, bislang nicht ausgewertete
Daten nach festgelegtem Plan nötig. Diese vier Läufe liefern Entwicklungswerte,
keine unabhängige Testgenauigkeit, Realempfangs- oder Drohnenprotokollleistung.
Das historische FP32-Gate bleibt unverändert; keine Literaturreproduktion.

## Start auf Schokobaer

```bash
cd ~/rf-learning/project
git pull --ff-only
cd ~/rf-learning
source .venv/bin/activate
python project/train_se_gain_development.py
```

Keine Paketinstallation. Das sind vier längere Trainingsläufe; gegenüber den
vorigen kleinen Diagnosen wird wesentlich mehr gerechnet. Das Skript zeigt nach
jeder Epoche den Validierungsverlust und die Zuordnungsrate. Für den festgelegten
Versuch keine Parameter zur Änderung von Epochen oder Gruppenzahl verwenden.

## Ausgaben, Sicherung und Abbruch

Neuer Ordner `models/se-gain-development-ZEITSTEMPEL`:

- Haupt-`protocol.json`: Plan, Quell-/Datenhashes, gewählte Gruppen und Umgebung.
- `training_selection.npz`: IDs, Labels, SNR und Gruppen des Trainingsausschnitts;
  die Originaldaten bleiben im bestehenden Datensatz, daher für Wiederherstellung
  auch `data/modulation12-v1` erhalten.
- Je Seed Anfangsgewichte, je Gain `history.json`, `best_checkpoint.pt`,
  `last_checkpoint.pt` und Fallbericht. Checkpoints tragen Gain, Aktivierung,
  Optimiererzustand, Epoche und Protokollverweis. Loader wie beim SE-Versuch.
- Haupt-`report.json` nach jedem abgeschlossenen Fall; unvollständige Paare werden
  nicht als vollständiger Vergleich bezeichnet.

Verläufe/Checkpoints werden pro Epoche atomar ersetzt. Kein automatisches Resume,
kein automatischer Ersatz eines fehlgeschlagenen Falls. Bei Abbruch bleiben
Teilstände erhalten; ein Neustart erzeugt einen neuen Ordner. Vorhandene Ausgaben
werden nicht überschrieben. `DEVELOPMENT_COMPLETE` erst nach allen vier Fällen.
Keine automatische Sicherung auf einen anderen Datenträger und kein Upload.

Danach bitte die **report.json direkt aus dem Hauptordner** hochladen.

Prüfung vor Veröffentlichung: 15 automatisierte Tests bestanden (zwei neue,
13 bestehende) unter Python 3.12 / PyTorch 2.9.1+cpu / NumPy 2.5.2. Neue Tests
prüfen vollständige gruppierte SNR-Auswahl, unvollständige/inkonsistente Gruppen,
vier kurze Optimierungsfälle, gezielt kontrollierte Validierungsverluste zur
Auswahl eines früheren statt des letzten Checkpoints, erneutes Laden, Train/Val-
Gruppenüberschneidung und Ausgabeschutz. Dabei werden künstliche kurze I/Q-Blöcke
und simulierte Quelldateien genutzt; kein Nutzer-Datensatz-/100-Epochen-Lauf.
