# Restabweichungen der Gradienten: Ursachen lokalisiert

Stand: 24.10.2026. Die Aufgabe war die Lokalisierung der zwei verbleibenden CPU-Gradientenabweichungen. **Beide Fälle sind nun durch gemessene Mechanismen erklärt.** Das ist kein Nachweis einer behobenen Ursache des früheren Trainingsstillstands und keine neue Erkennungsleistung.

## Versuchstreue

Der Untersuchungsplan wurde vor Ausführung im Repository festgehalten: `GRADIENT_LOCALIZATION_PLAN.json`, Commit `1068df0e45f16c082c2c2b92eb9af5a34c49c019`. Verwendet wurden dieselben sechs ursprünglichen Prüffälle, dieselben Gewichte und Eingaben, Original-TensorFlow 2.4.1 und die explizite SELU-Kompatibilitätsvariante. Beide CPU-Faltungsprofile wurden geprüft; keine Auswahl des günstigeren Profils je Seed.

Die Float32-Wiederholung reproduziert alle 192 TensorFlow-Parametergradienten der bisherigen sechs Fälle bitidentisch. Für Float64 wurden genau dieselben Float32-Werte hochgestuft; es wurden keine anderen Signale oder Anfangsgewichte erzeugt. An den Kriterien des bisherigen FP32-Vergleichs wurde nichts geändert.

Die Diagnose ergänzt Gradienten je Eingabe, die tatsächlichen Rückwärtsgewichte an Conv5 und Conv7, ReLU-Masken sowie isolierte Bias-Additionsableitungen. Die ursprüngliche Batch-Kreuzentropie wird dabei in drei Beiträge mit jeweils Faktor 1/3 zerlegt.

## Fall A: eine ReLU-Schaltentscheidung bei fast null

Betroffen: 4096 Samples, zwölf Ausgaben, Seed 903, oneDNN-aktiviertes CPU-Profil, Bias von Conv5. Der Unterschied entsteht im Rausch-Prüfsignal; Nullsignal und Sinussignal bestehen diesen Bias-Vergleich.

Genau eine ReLU-Entscheidung unterscheidet sich nahe null. Die TensorFlow-Ausgabe an der betreffenden Stelle liegt bei rund 9,45e-10. Die Zerlegung am am stärksten abweichenden Bias-Kanal ergibt:

| Anteil | Vorzeichenbehafteter Beitrag |
|---|---:|
| Gemessene Gesamtabweichung PyTorch minus TensorFlow | -3,20957e-6 |
| Allein durch unterschiedliche ReLU-Masken vorhergesagt | -3,21106e-6 |
| Änderung des ankommenden Rückwärtssignals | +6,13e-10 |
| TensorFlow-Summierungsanteil | -2,02e-10 |
| PyTorch-Summierungsanteil | +1,08e-9 |

Die Summe rekonstruiert den beobachteten Unterschied innerhalb der Rechengenauigkeit.

Eine zusätzliche, ausdrücklich nachgelagerte Kausalprobe übernimmt nur für die Rückwärtsrechnung dieser ReLU die TensorFlow-Maske. Die Vorwärtsausgaben bleiben dabei **bitidentisch** zum unveränderten PyTorch-Lauf. Der Conv5-Bias-Fehler fällt von 3,20957e-6 auf 7,45058e-9; alle 32 Parametergradienten dieses Diagnosefalls bestehen danach die bisherige Toleranz.

Diese Maske ist kein neues produktives Modell und wird nicht in das Training eingebaut. Die Kausalprobe erklärt einen numerischen Knickpunkt; sie wird nicht zum Bestehen des ursprünglichen Vergleichs umetikettiert. Bericht: `reports/gradient_mask_replay.json`.

## Fall B: Float32-Summierung des Bias-Gradienten

Betroffen: 4096 Samples, zwölf Ausgaben, Seed 901, natives CPU-Faltungsprofil, Bias von Conv7. Hier unterscheiden sich keine Conv-ReLU-Entscheidungen. Auch die getrennten Eingabebeiträge bestehen jeweils die Toleranz; der zusammengefasste Batch-Gradient verfehlt sie.

Bei identischen festgehaltenen Rückwärtssignalen wurde die Bias-Ableitung isoliert ausgeführt. TensorFlows isolierte Bias-Additionsableitung reproduziert den ursprünglichen TensorFlow-Modellgradienten exakt. Gegenüber der Float64-Summe derselben Float32-Werte ergibt sich:

| Messung | Maximale absolute Abweichung |
|---|---:|
| Gesamter ursprünglicher Unterschied | 1,01998e-5 |
| TensorFlow-Bias-Ableitung gegenüber genauerer Summe | 1,01809e-5 |
| PyTorch-Kanäle-zuerst-Ableitung gegenüber derselben Summe | 1,93861e-8 |

Die Zerlegung am kritischsten Kanal weist fast den gesamten Unterschied der TensorFlow-Float32-Summierung zu; die verbleibenden Beiträge aus Rückwärtssignal und PyTorch-Summierung sind deutlich kleiner. Damit wurde die Abweichung unabhängig von Faltung, Klassifikation und Lernvorgang bereits am isolierten Summierungsoperator reproduziert.

Das bedeutet ausdrücklich nicht, dass PyTorch diesem alten TensorFlow-Rundungsverhalten künstlich angepasst werden sollte. Der Referenzwert selbst ist hier eine endlich genaue Float32-Berechnung.

## Kontrolle mit höherer Präzision

| Präzision | Sechs Formen/Seeds × zwei CPU-Profile | Bestanden | Fehlgeschlagene Einzelgradienten je Eingabe |
|---|---:|---:|---:|
| Float32 | 12 | 10 | 1 |
| Float64, identische Ausgangswerte hochgestuft | 12 | 12 | 0 |

Jeder vollständige Diagnosevergleich enthält Verlust, Logits, sieben Faltungsaktivierungen und sämtliche 32 Parametergradienten; die drei Einzel-Eingaben liefern zusätzlich 96 Parametergradienten je Profilfall. Bei den untersuchten Conv5-/Conv7-Bias-Gradienten liegen die Float64-Abweichungen in der Größenordnung 1e-16 bis 1e-15.

Die Float64-Kontrolle erklärt die Präzisionsabhängigkeit. Sie ersetzt nicht nachträglich die ursprünglich festgelegte Float32-Freigabe und ist keine Empfehlung, das spätere GPU-Training pauschal auf Float64 umzustellen.

## Zulässige Schlussfolgerung

Die zwei konkreten Restfälle liefern **keinen Hinweis auf einen weiteren falschen Achsentausch oder eine andere fehlerhafte Schichtgleichung als Ursache dieser Abweichungen**. Sie sind durch eine nichtglatte ReLU-Grenze beziehungsweise die gemessene Bias-Summierung lokalisiert. Der vorher bereits bestätigte SELU-Ableitungsunterschied bleibt eine separate semantische Kompatibilitätskorrektur.

Unverändert offen: Die Ursache, warum das vollständige Zwölfklassen-Training auf Zufallsniveau stagnierte. Weder diese Diagnose noch die frühere kleine Wiedererkennung ist eine Generalisierungsprüfung.

Der alte FP32-Freigabestatus bleibt formal nicht bestanden. Eine weitere Qualifikation muss deshalb als eigener, vorab festgelegter Schritt zwischen mathematischer Implementierungsgleichheit und endlicher Legacy-Float32-Übereinstimmung unterscheiden. Es werden weder die bisherigen Fehlschläge gelöscht noch die Toleranz stillschweigend erhöht. Erst danach sind weitere vollständige Trainingsvergleiche sachgerecht.

## Reproduzierbarkeit und Artefakte

- `reports/gradient_localization_full.json`: alle 24 Profilfälle einschließlich Fehler, Masken, Einzel-Eingaben und Operatorzerlegung.
- `reports/gradient_localization_summary.json`: kompakte Zusammenfassung.
- `reports/gradient_mask_replay.json`: nachgelagerte Kausalprobe mit unveränderter Vorwärtsausgabe.
- `tools/export_harper_gradient_localization.py`: Original-TensorFlow-Prüfdaten und Gradientenzerlegung.
- `localize_reference_gradients.py`: Vergleich, isolierte Bias-Ableitungen und Summierungszerlegung.
- `tools/replay_reference_relu_mask.py`: reine Diagnoseintervention, keine produktive Aktivierung.

Zwölf gezielte Prüftests bestanden, darunter die Summierungszerlegung, beide Achsenlayouts der isolierten Bias-Ableitung und die Trennung von unveränderter Vorwärtsrechnung und Diagnose-Rückwärtsmaske. Die Original-Quelldateien und Trainingsskripte wurden nicht geändert.

Eine Wiederholung setzt die getrennte Referenzumgebung aus `REFERENCE_PARITY_RESULTS.md` voraus. In `~/rf-learning` mit bestehenden ursprünglichen TF-Prüfdaten beispielsweise:

```bash
.venv-reference-tf241/bin/python project/tools/export_harper_gradient_localization.py \
  --fixtures models/reference-parity-default-v1/tf-fixtures --output models/gradient-tf-v1

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python project/localize_reference_gradients.py \
  --tf-export models/gradient-tf-v1 --output models/gradient-localization-v1
```

Die Pfade sind Beispiele für neue Ausgabeordner. Der erste Befehl benötigt die bereits erzeugten ursprünglichen Prüfdaten; er greift nicht auf Funkaufnahmen oder den RadioML-Datensatz zu. Für die hier dokumentierten Befunde ist kein zusätzlicher Lauf auf dem Nutzerrechner erforderlich.
