# Kontrollierter Vergleich: IQ mit Huelle und Phasendifferenz

Plan: `IQ_FEATURE_COMPARISON_PLAN_V1.json`, vor Implementierung und echtem
Trainingslauf im Repository registriert. Alle zwoelf Kategorien bleiben erhalten.
Die bestehende Validierung ist ein mehrfach verwendeter Entwicklungsdatensatz.
Der Testsatz wird nicht geoeffnet; er ist historisch bereits ausgewertet worden
und wird daher nicht als versiegelter abschliessender Test bezeichnet.

## Ausfuehren

Aus `~/rf-learning` mit aktivierter bestehender PyTorch-Umgebung:

```bash
python project/train_iq_feature_comparison.py
```

Der Lauf sucht genau eine abgeschlossene urspruengliche
`models/se-gain-development-...`-Auswertung. Er braucht daraus
`training_selection.npz` und `seed2081/initial_state.pt` sowie
`seed2082/initial_state.pt`. Das sind die gespeicherten **Anfangsgewichte**,
keine trainierten Checkpoints. Bei mehreren Quelllaeufen:

```bash
python project/train_iq_feature_comparison.py --source models/DEIN-URSPRUENGLICHER-DEVELOPMENT-LAUF
```

Optional: `--data data/modulation12-v1`, `--output NEUER-ORDNER`,
`--device cpu`. Standard ist CUDA; kein stiller CPU-Fallback.
Die registrierten Epochen und Auswahlgroessen sind nicht per CLI veraenderbar.
Vier Trainingslaeufe werden vollstaendig gerechnet: zwei Varianten je Seed,
jeweils 100 Epochen, 53 Updates pro Epoche. Die Laufzeit wird protokolliert;
es gibt noch keine belastbare Laufzeitprognose fuer diesen Versuch.

Neue Ausgabe: `models/iq-feature-comparison-ZEITSTEMPEL/report.json`.
Bitte diesen **report.json** hochladen, auch wenn ein Fehler gemeldet wurde.
Bestehende Modelle und Daten bleiben unveraendert. Bei Fehler/Abbruch keine
automatische Fortsetzung; unvollstaendige Ausgaben bleiben bestehen.

## Was gleich bleibt

- Dieselben 1680 Trainingsbeispiele: 20 komplette Bedingungsgruppen je SNR,
  jeweils alle zwoelf Kategorien, Auswahlseed2080.
- Dieselben 1080 Validierungsbeispiele; getrennte Quell-IDs und Bedingungen.
- Dieselben gespeicherten Anfangstensoren der Seeds2081/2082, per Digest geprueft.
- Experimenteller Harper-Port mit tf-kompatibler SELU und SE-Gain2.
- Frischer Adam je Variante, LR0.0001, eps1e-8; Batch32, identische Permutationen,
  letzte Batch16, 5300 Updates pro Variante; kein Scheduler, Clipping oder Augmentieren.
- Bestes Modell nach strikt kleinstem mittlerem Validierungs-CE, bei Gleichstand
  fruehere Epoche; alle100 Epochen ohne Early Stopping.

Beide Modelle verwenden vier Eingangskanaele und dieselbe Parameterzahl.
Die erste Faltung uebernimmt die urspruenglichen IQ-Kernel und den Bias;
die beiden neuen Kernel starten exakt mit Null. Alle anderen Tensoren bleiben
identisch. Zu Beginn werden identische Ausgaben beider Varianten geprueft.

## Was sich aendert

| Kanal | `iq_zero` | `iq_envelope_phase` |
|---|---|---|
| 0 | I unveraendert | I unveraendert |
| 1 | Q unveraendert | Q unveraendert |
| 2 | Null | Betrag des komplexen IQ-Werts |
| 3 | Null | Hauptwert der Phasendifferenz aufeinanderfolgender Werte, geteilt durch pi |

Phasendifferenz am ersten Zeitpunkt und bei einem Nullprodukt: Null.
Merkmalsberechnung in complex128/float64, dann float32 als Modelleingabe.
Keine Frequenzzentrierung, Huellennormalisierung, Phasenentfaltung oder
an den Labels angepasste Skalierung. Die vorhandene Block-RMS0.5 bleibt.
Inferenz auf spaeteren Empfangsdaten muss dieselbe Vorverarbeitung verwenden.

Die Zusatzmerkmale sind deterministische Funktionen des IQ-Signals. Sie liefern
keine neue Messinformation, koennen aber die Lernaufgabe anders darstellen.
Der Versuch prueft beide Merkmale gemeinsam, nicht ihre isolierten Einzelwirkungen.
Die Nullkanaele der Kontrolle bleiben inaktiv: gleiche nominelle Parameterzahl
bedeutet keine gleiche effektive Kapazitaet oder gleiche Laufzeit.

## Entscheidung und Nachvollziehbarkeit

Eine gerichtete Verbesserung wird nur gemeldet, wenn die Merkmalsvariante
bei **beiden** Seeds niedrigeren ausgewaehlten Validierungs-CE **und** hoeheren
Macro-F1 ueber alle zwoelf Kategorien erreicht. Sonst gemischt/nicht bestaetigt.
Keine Auswahl nur des besten Seeds und kein Nachjustieren anhand des Testsatzes.

Zusaetzlich werden Trainingsmetriken, Klassenrecalls, FSK-Mittelrecall,
SNR-Auswertungen sowie das letzte Epochenresultat gezeigt. In Signal-SNR-Gruppen
ist Noise nicht enthalten; dessen Recall ist dort ausdruecklich undefiniert.
Die Daten bleiben synthetisch und liefern keinen Nachweis fuer reale RF-,
Drohnenprotokoll- oder Coral-Erkennung.

Gesichert werden Protokoll/Codehashes, exakte Trainingsauswahl, Anfangsgewichte,
History jeder Epoche sowie beste und letzte Checkpoints mit Optimizer,
Permutationsgenerator und Torch-RNG-Zustaenden. Architektur, Variantenname,
Vorverarbeitung und SE-Gain stehen explizit im Checkpoint. Zum Wiederherstellen
`restore_feature_checkpoint` verwenden; historische Zweikanal-Loader passen nicht.
Ein Erfolgsmarker entsteht erst nach allen vier abgeschlossenen Varianten.

Die historischen FP32-Paritaetsabweichungen und Toleranzen bleiben unveraendert.
Der Vergleich ist eine experimentelle Port-Adaption, keine Literaturreproduktion.
Deterministische Torch-Algorithmen und cuDNN werden eingeschaltet, TF32 aus;
CUDA-BLAS-Workspace wird vor CUDA-Verwendung gesetzt. Keine Behauptung
bitidentischer Ergebnisse zwischen unterschiedlichen Rechnern oder Backends.

## Lokale Implementierungspruefung

```bash
python -m unittest test_iq_feature_comparison test_modulation_reference test_modulation_learning
```

Die kurzen Fixture-Laeufe testen Merkmalswerte, gepaarte Anfangsausgaben und
Batchfolgen, Zusatzkanal-Gradienten, Checkpoint-Wiederherstellung, Entscheidung
ueber beide Seeds, serialisierbare SNR-Metriken, den kompletten Vier-Varianten-
Ablauf und die Ablehnung einer manipulierten Auswahl. Sie ersetzen den echten
GPU-Trainingslauf auf den vorhandenen Daten nicht.
