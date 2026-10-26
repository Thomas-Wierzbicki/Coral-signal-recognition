# Full-Batch-Lernratenvergleich V1 – 26.10.2026

## Befund des vorigen Versuchs

Beide hochgeladenen Verläufe enthalten Epoche 0 bis 100 und stimmen am Ende
mit dem Bericht überein. Full-Batch: maximal 50 % Wiedererkennung bei Epoche 95,
Endstand 26,7 %. Mini-Batches: maximal 62,5 % bei Epoche 87, Endstand 53,3 %;
minimaler Verlust 0,7701 bei Epoche 93, dann 1,4708 bei Epoche 95.
Das sind Trainingswerte auf denselben 120 Beispielen, keine Testgenauigkeiten.

Bei gleichen 100 Optimierungsschritten: Full-Batch 26,7 %, Mini-Batches 8,3 %
(Epoche 25). Full-Batch hat dabei viermal so viele Beispiele verarbeitet. Weder
der Endvergleich nach Epochen noch dieser Vergleich isoliert die Batchgröße.
In allen erfassten ersten Batches waren sämtliche Gradientennormen positiv.
Das beweist weder gute Konditionierung noch ausreichende Signalunterscheidung.

Die Verläufe zeigen verzögertes Lernen mit Rückschlägen; die vollständige
Ursache des früheren Stillstands bleibt offen. Dokumentierte Zusammenfassung
mit SHA256 der Uploads: [reports/training_collapse_history_summary.json](reports/training_collapse_history_summary.json).

## Neuer kontrollierter Versuch

[TRAINING_LR_PLAN_V1.json](TRAINING_LR_PLAN_V1.json) wurde vor der Ausführung
separat festgeschrieben. Zwei neue Läufe ab identischen Anfangsgewichten:

| Merkmal | Fall 1 | Fall 2 |
|---|---:|---:|
| Lernrate | 0,001 | 0,0001 |
| Full-Batch-Größe | 120 | 120 |
| Schritte/Epochen | 400 | 400 |
| Verarbeitete Beispiele | 48.000 | 48.000 |

Alle zwölf Klassen, je zehn Beispiele, dieselbe deterministische Auswahl aus
train, SNR >=15 dB für Signale, Rauschen ohne SNR. Der Datensatz und die
Anfangsgewichte müssen die im Plan festgeschriebenen Hashes erfüllen.
Die unveränderte Auswahlfunktion und ihre Prüfsumme werden protokolliert.
Explizite TensorFlow-kompatible SELU; Adam eps1e-8; keine Scheduler,
Gradientenbegrenzung, Augmentation oder vorzeitige Beendigung.
Validierungs-/Testdaten werden nicht geöffnet. Historisches FP32-Gate unverändert.

Primärvergleich: finaler Trainingsverlust und Wiedererkennung nach 400 Schritten.
Zwischenminima und Verlustsprünge sind beschreibende Diagnosen, keine nachträgliche
Modellauswahl. Eine niedrigere Lernrate kann bei gleichem Budget langsamer lernen;
weniger Schwankung allein beweist daher keine bessere Trainingsstrategie.
Ein Seed und Trainingsmemorierung liefern keinen Generalisierungsnachweis.

## Auf Schokobaer ausführen

```bash
cd ~/rf-learning/project
git pull --ff-only
cd ~/rf-learning
source .venv/bin/activate
python project/diagnose_training_lr.py
```

Keine neuen Pakete nötig. Neuer Ordner `models/training-lr-ZEITSTEMPEL`.
`protocol.json`, `selection.npz`, `initial_state.pt`, pro Fall `history.json`
und letzter Modell-/Optimierercheckpoint; `report.json` mit Endständen und
Verlaufskennzahlen. `LR_DIAGNOSTIC_COMPLETE` erscheint erst nach beiden Fällen.
Bestehende Ausgaben werden nicht überschrieben. Kein automatisches Resume;
Checkpoints dienen hier der Sicherung, nicht einer geprüften Fortsetzung.
`--epochs`/`--per-class`/`--activation` sind technische Abweichungen, im Protokoll
als solche markiert; für den geplanten Vergleich keine davon ändern.

Anschließend zunächst die neue `report.json` hochladen. Explorer öffnen:

```bash
explorer.exe "$(wslpath -w /home/thwie/rf-learning/models)"
```

## Prüfstatus

Der neue Versuch auf Nutzerdaten/RTX steht aus. Technische Tests verwenden kurze
künstliche I/Q-Blöcke und simulierte Quelldateien, keine Empfangs-/Leistungsdaten.

Acht automatisierte Tests bestanden unter PyTorch 2.9.1+cpu, NumPy 2.5.2,
Python 3.12. Der neue Ablauf prüft unter anderem beide tatsächlichen
Optimierer-Lernraten, gleiche Anfangsgewichte und Schrittzahlen, Training-only-Zugriff,
Protokoll/Checkpoint-Ausgaben und Überschreibschutz. Kein Nutzer-GPU-Lauf ausgeführt.
