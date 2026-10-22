# RadioML-Ergebnisstand – 22. Oktober 2026

Diese Ergebnisse stammen aus den vom Projektinhaber ausgefuehrten Windows-
Trainings-/Coral-Laeufen und dem Ubuntu-Compilerbericht. Modellbytes und
Originalprotokolle sind lokal, nicht Bestandteil dieses GitHub-Dokuments.

## Daten und Vergleich

Original GOLD_XYZ_OSC.0001_1024.hdf5: X=(2555904,1024,2), Y=(2555904,24),
Z=(2555904,1). Vom Nutzer ermittelter SHA256:
`e3dd0bef66a3426959ee66a1709a8c0a95d4f8395d18aaf6f1214bdbc763bd38`.
Die Datei liegt in Downloads in einem gleichnamigen Unterordner.
24-Spalten-Klassenfolge mit classes-fixed.json abgeglichen.
Auswahl: AM-DSB-WC, FM, BPSK, QPSK, 16QAM (Originalspalten 19,21,3,4,12).
SNR 0 bis 20 dB in 2-dB-Schritten, 100 Beispiele je Klasse/SNR.
3850 Training, 825 Validierung, 825 Test (165 je Klasse; 75 je SNR im Test).
Spektrogramm und direkte I/Q verwenden identische Quellzeilen, Labels, SNR
und Splitreihenfolgen. Beide normalisieren komplexes Block-RMS auf 0,5.

| Messung | Spektrogramm | Direkte I/Q |
|---|---:|---:|
| Eingang ohne Batch | 29 × 128 × 1 | 1 × 1024 × 2 |
| Float CPU | 71,6 % | 97,0 % |
| INT8 CPU | 71,6 % | 97,2 % |
| INT8 Coral | nicht dokumentiert | 97,2 % (802/825) |
| CPU-INT8/Coral Vorhersageuebereinstimmung | – | 100,0 % |

Der Gewinn von 25,6 Prozentpunkten betrifft diesen Testsatz. Die I/Q-Variante
erhaelt Phaseninformation, hat aber auch eine andere Architektur. Kein isolierter
Kausalnachweis fuer den Einfluss der Phase; keine neue unabhaengige Testkohorte.

## Coral-Konfusionsmatrix

Zeilen Soll, Spalten vorhergesagt:

| Soll / erkannt | AM-DSB-WC | FM | BPSK | QPSK | 16QAM |
|---|---:|---:|---:|---:|---:|
| AM-DSB-WC | 165 | 0 | 0 | 0 | 0 |
| FM | 0 | 165 | 0 | 0 | 0 |
| BPSK | 0 | 0 | 165 | 0 | 0 |
| QPSK | 0 | 0 | 1 | 152 | 12 |
| 16QAM | 0 | 0 | 0 | 10 | 155 |

Recall: AM-DSB-WC/FM/BPSK je 100 %, QPSK 92,1 %, 16QAM 93,9 %.
23 Fehler: 22 QPSK/16QAM-Verwechslungen, einmal QPSK als BPSK.

| SNR dB | Richtig | Genauigkeit |
|---|---:|---:|
| 0 | 67/75 | 89,3 % |
| 2 | 69/75 | 92,0 % |
| 4 | 74/75 | 98,7 % |
| 6 | 74/75 | 98,7 % |
| 8 | 74/75 | 98,7 % |
| 10 | 73/75 | 97,3 % |
| 12 | 72/75 | 96,0 % |
| 14 | 75/75 | 100,0 % |
| 16 | 75/75 | 100,0 % |
| 18 | 75/75 | 100,0 % |
| 20 | 74/75 | 98,7 % |

## Compiler und Laufzeit

Compiler 14.1.317412892, erfolgreich in 53 ms. 82,38 KiB Eingangsmodell,
124,56 KiB kompiliertes Modell. Ein Subgraph, alle 12 Operationen auf TPU:
3 Conv2D, 3 MaxPool2D, 2 FullyConnected, 2 Quantize, 1 Reshape, 1 Softmax.
Parametercache 154,50 KiB, verbleibend 7,51 MiB; kein Off-Chip-Parameterstreaming.

Ein geladenes Modell, ein ungemessenes Warmup, 825 Testaufrufe.
invoke-Median 0,457 ms, p95 0,811 ms. Enthalten sind Python/USB-Kosten von
invoke; nicht enthalten sind RMS, Quantisierung, Ergebnisabruf oder Argmax.
Kein Ende-zu-Ende-SDR-Durchsatzbenchmark.

Lokale Originaldateien: models/radioml2018-iq-v1/report.json,
compiled/coral_report.json, compiled/coral_predictions.npz,
test_predictions.npz und preprocessing.json.

## Grenzen und naechste Schritte

RadioML ist synthetisch; keine echte Empfangsvalidierung. Keine noise-/carrier-
oder unknown-Klasse in dieser Fuenfklassenvariante. Ein unbekanntes Signal kann
also trotzdem einer bekannten Klasse zugeordnet werden. Modellbewertungen
sind keine garantierten Wahrscheinlichkeiten. Disjunkte Zeilen belegen keine
unabhaengigen Aufnahmen. Der Testsatz wurde zur Variantenbewertung angesehen;
fuer die abschliessende Bewertung ist ein neuer unangetasteter Testsatz noetig.
Zuerst diesen Stand sichern, dann weitere Quellzeilen ausschliesslich fuer einen
neuen Test auswaehlen; echte SDR-Aufnahmen anschliessend separat pruefen.

## Zusaetzlicher Holdout: 11000 neue Quellzeilen

Vom Nutzer auf Coral ausgefuehrt, unveraendertes Modell. Alle bisherigen 5500
Train/Val/Test-Zeilen ausgeschlossen. 200 Beispiele je Klasse/SNR, insgesamt
2200 je Klasse und 1000 je SNR. 10789/11000 richtig (98,1 % gerundet).

| Soll / erkannt | AM-DSB-WC | FM | BPSK | QPSK | 16QAM |
|---|---:|---:|---:|---:|---:|
| AM-DSB-WC | 2200 | 0 | 0 | 0 | 0 |
| FM | 0 | 2200 | 0 | 0 | 0 |
| BPSK | 0 | 0 | 2196 | 1 | 3 |
| QPSK | 0 | 0 | 5 | 2097 | 98 |
| 16QAM | 0 | 0 | 1 | 103 | 2096 |

Recall: 100 %, 100 %, 99,8 %, 95,3 %, 95,3 % in dieser Reihenfolge.
201 von 211 Fehlern betreffen QPSK/16QAM. invoke-Median 0,424 ms, p95 0,514 ms.
Kein gesicherter Geschwindigkeitsgewinn gegenueber frueherem Lauf.

| SNR dB | Genauigkeit |
|---|---:|
| 0 | 91,5 % |
| 2 | 96,8 % |
| 4 | 98,3 % |
| 6 | 98,6 % |
| 8 | 99,4 % |
| 10 | 99,2 % |
| 12 | 99,5 % |
| 14 | 99,4 % |
| 16 | 98,3 % |
| 18 | 98,8 % |
| 20 | 99,1 % |

Protokoll lokal unter models/radioml2018-iq-v1/holdout-v1/coral_report.json.
Neue Zeilen derselben synthetischen Quelle, weiterhin keine bewiesene Aufnahme-
Unabhaengigkeit und keine SDR-Validierung. Erneute Sicherung nach dem Holdout
wurde empfohlen, aber deren Ausfuehrung bisher nicht bestaetigt.
