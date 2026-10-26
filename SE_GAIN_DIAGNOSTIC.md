# Kontrollierter SE-Skalierungsversuch – 26.10.2026

## Ausgangsbefund und Hypothese

Die gespeicherten Modelle reproduzierten beide früheren Verwechslungsmatrizen
exakt (75,0 % / 73,3 % Trainings-Wiedererkennung). Zu Beginn fiel die gemessene
Aktivierungs-RMS von erster Faltung zu letzter ReLU von 0,122 auf 0,000125,
etwa Faktor 974. Die sechs SE-Gewichtungen lagen dabei jeweils ungefähr bei 0,5.
Nach Training ist diese starke Abschwächung nicht mehr vorhanden.
Zusammenfassung: [reports/training_feature_summary.json](reports/training_feature_summary.json).

Das ist ein Ansatzpunkt, kein Nachweis einer alleinigen Ursache. Insbesondere
war die Beschreibung vor dem Pooling bereits identisch zu dessen Mittelwerten
und Varianzen. Gleiche Trennwerte dort sind konstruktionsbedingt und kein
Nachweis für korrektes oder fehlerhaftes Pooling.

## Festgelegter Eingriff

Plan vor Ausführung: [SE_GAIN_PLAN_V1.json](SE_GAIN_PLAN_V1.json).

| Einstellung | Kontrolle | Versuchsvariante |
|---|---:|---:|
| SE-Ausgabe | x × sigmoid(...) | 2 × x × sigmoid(...) |
| Lernrate Adam | 0,0001 | 0,0001 |
| Full-Batch | 120 | 120 |
| Schritte | 400 | 400 |
| Anfangsgewichte, Auswahl, Permutationen | identisch | identisch |

Alle zwölf Klassen, zehn Beispiele je Klasse aus train, Signal-SNR >=15 dB,
Rauschen ohne SNR. Anfangsgewichte und Quelldaten müssen die Planprüfsummen
erfüllen. Die identische Auswahlfunktion bleibt unverändert. Die Standardmodelle
und alten Trainingsprogramme bleiben ebenfalls unverändert; der neue Eingriff
liegt ausdrücklich in `se_gain_experiment.py` und `diagnose_training_se_gain.py`.
Explizite TensorFlow-kompatible SELU, keine Scheduler, Clipping oder Augmentation.
Validierung und Test werden nicht geöffnet. Keine Änderung des historischen Gates.

Gain 2 wirkt während des gesamten Trainings, nicht nur bei der Initialisierung.
Ein besseres Ergebnis würde deshalb die Wirkung dieser Variante stützen, aber
noch keinen isolierten Beweis für eine reine Initialisierungsursache liefern.
Die effektiven SE-Gates können jetzt 0 bis 2 statt 0 bis 1 annehmen. Gleiche
Gewichte bedeuten bei verschiedener Skalierung bewusst verschiedene Funktionen.
Diese Variante ist keine unveränderte Harper-Literaturreferenz.

Primär zählen Endverlust und Trainings-Wiedererkennung nach exakt 400 Schritten.
Erste Schwellenüberschreitungen von 50 % / 90 %, Verlustsprünge und Zwischenminima
werden zusätzlich protokolliert; keine nachträgliche Bestmodell-Auswahl.
Vor dem ersten Schritt werden auf denselben zwölf Beispielen (eins pro Klasse)
Aktivierungs-RMS-Werte erfasst. Eine bloße schnellere Anlaufphase löst verbleibende
Klassenverwechslungen nicht automatisch. Ein Vorteil erfordert weitere geplante
Seeds vor größerem Training. Keine Aussage über Generalisierung oder Realempfang.

## Start auf Schokobaer

```bash
cd ~/rf-learning/project
git pull --ff-only
cd ~/rf-learning
source .venv/bin/activate
python project/diagnose_training_se_gain.py
```

Keine neue Installation. Neuer Ordner `models/training-se-gain-ZEITSTEMPEL` mit
Protokoll, Auswahl, Anfangszustand, Verlauf und letztem Checkpoint je Variante
sowie `report.json`. `SE_GAIN_DIAGNOSTIC_COMPLETE` erscheint erst nach beiden
Läufen. Alte Ausgaben werden nicht überschrieben; kein automatisches Resume.
Für den geplanten Vergleich keine CLI-Parameter zur Änderung von Epoche,
Beispielzahl je Klasse oder Aktivierung verwenden.

Anschließend bitte die neue **report.json** aus diesem Ordner hochladen.

## Modelle später eindeutig laden

Der Gain ist eine Einstellung des Rechenwegs, kein zusätzlicher Gewichtstensor.
Der Checkpoint enthält deshalb `se_gain`, `activation` und einen ausdrücklichen
Architekturnamen. Nicht allein das `state_dict` in ein Standard-Harper-Modell laden.
Für spätere Auswertung gibt es `restore_gain_checkpoint` in `se_gain_experiment.py`:

```python
import torch
from se_gain_experiment import restore_gain_checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
model = restore_gain_checkpoint(checkpoint)
model.eval()
```

Das aktuelle alte Schichtdiagnoseskript unterstützt diese Versuchsordner nicht;
eine Übertragung benötigt bewusstes Laden der Gain-Variante.

Prüfung vor Veröffentlichung: elf automatisierte Tests bestanden (drei neue und
acht bestehende) unter Python 3.12 / PyTorch 2.9.1+cpu / NumPy 2.5.2. Dazu gehören
bitgleiche Kontrollausgaben und Gradienten bei 4096 Abtastwerten, die lokale
Faktor-2-Ausgabe samt Gradienten, gleiche Anfangsgewichte und Schrittbudgets,
Training-only-Zugriff, Checkpoint-Gain/Restore und Überschreibschutz. Die kurze
Ablaufprüfung ist kein 400-Schritte-Lauf auf Nutzerdaten; Nutzer-GPU-Ergebnis steht aus.
