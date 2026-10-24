# Zwölfklassen-Vergleich mit veröffentlichter Referenzarchitektur

Stand: 24. Oktober 2026. Ziel bleibt die Erkennung aller zwölf Kategorien; Drohnenprotokollerkennung bleibt das anschließende eigene Arbeitspaket. Neue Hardware ist nicht erforderlich.

## Was dieser Versuch klärt

Bisher: eigenes CNN, 56,0 % auf dem synthetischen Entwicklungsprüfsatz, Macro-F1 0,531. GFSK/GMSK bleiben auch bei gutem SNR schlecht. Die Drei-Klassen-Diagnose mit 89,2 % gegen 5,0 % ist kein fairer Architekturvergleich: unterschiedliche Klassenanzahl, Trainingsbedingungen und Darstellung. Der Grenzfall identischer GFSK-/GMSK-Parameter beweist keine falsch beschrifteten Duplikate im normalen Datensatz. Pooling als alleinige Fehlerursache wurde nicht nachgewiesen.

Dieser Versuch vergleicht zwei Architekturpakete bei identischen Daten, Eingaben und Trainingsregeln. Er isoliert nicht einzelne Komponenten wie Pooling oder Kanalgewichtung. Es gibt keine zugesagte Verbesserung.

## Primärquellen und Herkunft

- Harper, Thornton, Larson (2023), *Automatic Modulation Classification with Deep Neural Networks*: https://doi.org/10.3390/electronics12183962 und https://arxiv.org/abs/2301.11773
- Autoren-Code: https://github.com/caharper/Automatic-Modulation-Classification-with-Deep-Neural-Networks
- Geprüfter Commit: `902cca45c3bf04b5da9f78030372b5a03f730f42`.
- Architektur: https://github.com/caharper/Automatic-Modulation-Classification-with-Deep-Neural-Networks/blob/902cca45c3bf04b5da9f78030372b5a03f730f42/experiments/radioml2018.01a/models.py
- Kanalgewichtung: gleiches Repository, `experiments/radioml2018.01a/layers.py`.
- Varianzdefinition: gleiches Repository, `amc_neural_networks/layers.py`.

Die MIT-Lizenz des Autoren-Codes liegt unter `third_party/harper2023_LICENSE.txt`. Die PyTorch-Portierung steht in `modulation_reference.py`. Commit, Quellblob-IDs und Anpassungen stehen außerdem in jedem Referenz-Trainingsprotokoll.

Verwendet werden die **Standardoptionen des veröffentlichten create_model-Codes**: SE an, letzte ReLU an, Dilatation an, keine residuale SE-Verbindung und keine Selbstaufmerksamkeit. Das ist nicht die Behauptung, jede Variante oder das beste Ergebnis der Publikation zu reproduzieren.

## Architektur und Änderungen

Sieben Conv1D-Stufen mit Kanälen 32/48/64/72/84/96/108, Kernelgrößen 7/5/7/5/3/3/3 und Dilatationen 1/2/3/2/2/2/1. Alle Faltungen mit Schrittweite eins und längenerhaltendem Padding. Nach den ersten sechs Stufen erfolgt SE-Kanalgewichtung; nach der siebten werden zeitlicher Mittelwert und Populationsvarianz verkettet. Danach folgen zweimal 128 Einheiten mit SELU und zwölf Ausgabelogits. Insgesamt 201332 Parameter gegenüber 287812 im bisherigen Netz.

Dokumentierte Anpassungen gegenüber dem Autoren-Code:

- PyTorch statt TensorFlow; dieselben Schichtgleichungen, kein Übertragen vortrainierter Gewichte.
- 4096 statt 1024 komplexe Samples, zwölf statt 24 Ausgaben.
- Weiterhin rohe I/Q-Werte und unveränderte Block-RMS-Normalisierung auf 0,5. Der NCHW-Eingabeadapter bleibt `[N,2,1,4096]`.
- Logits statt abschließendem Softmax, da PyTorch CrossEntropyLoss dieses intern berücksichtigt.
- Glorot-Uniform-Initialisierung der Faltungen und normalen dichten Schichten, korrigiert abgeschnittene He-Normalverteilung der SE-Schichten entsprechend den Initialisiererfamilien des Originals. Zufallszahlen sind nicht frameworkübergreifend identisch.
- Unser vorher festgelegtes Trainingsprotokoll statt der Optimierer-/Callback-Einstellungen der Publikation. Insbesondere ist die Lernrate weiterhin 0,001, nicht die 0,0001 im Autoren-Trainingsskript. Dieser Versuch prüft beide Netze unter **unserem** Protokoll; er liefert keine optimale Abstimmung beider Netze.

Numerische Prüftests vergleichen den vollständigen Vorwärtslauf mit einer unabhängigen NumPy-Auswertung der Quellgleichungen. Das ist keine Ausführung des originalen TensorFlow-Programms und kein Nachweis identischer Trainingsergebnisse.

## Auf Schokobaer starten

```bash
cd ~/rf-learning
source .venv/bin/activate
git -C project pull --ff-only
python project/compare_modulation12.py
```

Es laufen nacheinander sechs Trainingsläufe: bisheriges Netz und Referenznetz jeweils mit Seeds 2051, 2052 und 2053. Die Referenz behält die volle Zeitauflösung durch sieben Stufen und benötigt deshalb mehr Rechenzeit und Aktivierungsspeicher als das alte Netz, trotz geringerer Parameterzahl. Währenddessen den Prozess auslaufen lassen. CUDA wird standardmäßig verlangt.

Unverändert bleiben:

| Einstellung | Wert |
|---|---|
| Datensatz | vorhandenes `data/modulation12-v1` |
| Klassen | am-dsb, fm, bpsk, qpsk, 16qam, 64qam, 2fsk, 2gfsk, 2gmsk, ofdm-64, ook, noise |
| Training / Validierung / Entwicklungstest | 5040 / 1080 / 1080 |
| Optimierer | Adam, Lernrate 0,001, eps 1e-8, kein Weight Decay |
| Batchgröße | 32, gleiche Batchreihenfolge je Seed |
| Begrenzung | maximal 40 Epochen, acht Epochen Geduld |
| Modellwahl pro Lauf | geringster Validierungsverlust |
| Zusätzliche Trainingsaugmentation | keine |
| Frequenzkontrollen | dieselben gepaarten Verschiebungen -1600/+1600 Hz |

Der Test wird innerhalb jedes Laufes erst nach der Modellwahl geladen. Weil wir diesen Prüfsatz bereits zur Weiterentwicklung verwendet haben, ist er eine **Entwicklungsreferenz**. Die Auswertung darf nicht nachträglich zur Auswahl des besten Seeds verwendet werden. Drei Seeds beschreiben eine begrenzte Trainingsstreuung, keine allgemeine statistische Sicherheit. Das ursprüngliche Modell in `models/modulation12-v1` bleibt erhalten.

## Ausgaben

Neuer Ordner `models/modulation12-reference-v1`:

- `protocol.json`: vor dem Training festgelegte Regeln, Daten- und Quellcode-Prüfsummen.
- `environment-freeze.txt`: automatisch gesicherte Python-Paketversionen.
- Je Architektur/Seed ein Unterordner mit Gewichten, erneut geladenem TorchScript-Inferenzexport, Vorhersagen, Konfusionsmatrix, Lernverlauf und Bericht.
- `runs.json`: nach jedem abgeschlossenen Lauf aktualisierte Einzelresultate.
- `comparison.json`: Mittelwerte, Stichprobenstandardabweichung der Genauigkeit, Macro-F1, Recall aller zwölf Klassen, SNR- und Frequenzkontrollen.
- `COMPARISON_COMPLETE`: Abschlussmarker nach allen Läufen.

Bestehende Ausgabeordner werden nicht überschrieben. Bei einem abgebrochenen Versuch bleiben fertige Teilergebnisse erhalten; dieser einfache Runner unterstützt keine Fortsetzung mitten im Training. Für einen neuen Versuch einen neuen `--output` wählen. Ein einzelner vorab gewünschter Seed ist mit `--seeds 2051` möglich, liefert jedoch keine Streuung über mehrere Läufe. Einstellungen nicht nach Betrachtung der Testergebnisse variieren und anschließend nur den besten Lauf melden.

## Geprüft und noch offen

Lokal geprüft: Schichtgleichungen gegen NumPy, Populationsvarianz, Gradienten bei vollen 4096 Samples, TorchScript mit anderer Batchgröße, bestehende Baseline-Tests und vollständiger CPU-Ablauf beider Architekturen auf einem kleinen echten Generator-Datensatz. Die Kurzläufe prüfen nur den Ablauf. GPU-Leistung und Erkennungsleistung auf dem vollständigen Nutzerdatensatz sind damit nicht nachgewiesen.

Nach dem Vergleich werden zuerst Lernverläufe und alle Klassen betrachtet. Danach ist ein separater Versuch zur Datenmenge bzw. frischer Augmentation möglich, ohne gleichzeitig Architektur und Darstellung zu ändern. Unabhängige Generatoren, reale Aufnahmen und ein unberührter Abschlussprüfsatz bleiben erforderlich. Kein Coral-Export und keine Drohnenprotokollerkennung in diesem Schritt.

Weitere Grundlagen: TorchSig/Sig53 https://arxiv.org/abs/2207.09918 ; Vergleich synthetischer und aufgezeichneter RF-Daten https://arxiv.org/abs/2010.00178 ; Hinweise des RadioML-Anbieters https://www.deepsig.ai/datasets/ .
