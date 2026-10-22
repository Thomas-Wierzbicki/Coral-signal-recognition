# RadioML: direkte I/Q-Variante

Der Spektrogramm-Basistest erreichte auf 825 Testbeispielen Float/INT8 jeweils
71,6 %. Recall: AM-DSB-WC 100 %, FM 100 %, BPSK 69,7 %, QPSK 12,7 %,
16QAM 75,8 %. Die I/Q-Variante bewahrt die Phase. Ein besseres Ergebnis ist
noch nicht nachgewiesen. Auch die Architektur aendert sich; ein Unterschied
waere deshalb kein isolierter Beweis fuer den Einfluss der Phase.

## Import (Anaconda Prompt im Projektordner)

```bat
git pull --ff-only
conda activate radioml
python radioml_iq_import.py "C:\Users\thwie\Downloads\GOLD_XYZ_OSC.0001_1024.hdf5\GOLD_XYZ_OSC.0001_1024.hdf5"
```

Der vorhandene Import unter data/radioml2018-v1 ist erforderlich. Dessen
source_index, Label, SNR und Zeilenreihenfolge bleiben fuer alle drei Splits
erhalten. Der Import prueft Quellgroesse und jedes ausgewaehlte Label/SNR;
dies ersetzt keinen vollstaendigen SHA256-Vergleich. IQ kommt aus der
Original-HDF5, nicht aus Spektrogrammen. Ausgang: data/radioml2018-iq-v1,
3850/825/825 Beispiele der Form (N,1,1024,2), Kanalreihenfolge I,Q.
Komplexes Block-RMS wird auf 0,5 normiert, ohne Clipping, FFT oder
getrennte Kanalnormalisierung. Negative Werte bleiben erhalten.

## Training

```bat
conda activate coral-train
python train_radioml_iq.py
```

Conv2D-Kerne (1,7)/(1,5) verarbeiten die Zeitachse und beide Kanaele;
Pooling, Flatten, Dense und Softmax bilden ein eigenes kleines CNN.
50 Epochen maximal, Batch 32, Seed 2031, Early Stopping nach 8 Epochen
ohne bessere Validierungs-Loss. Kalibrierung verwendet nur Trainingsdaten.
Vollinteger-TFLite mit uint8-Ein-/Ausgang: die Quantisierung bildet auch
negative Float-Eingaenge mittels Skala und Nullpunkt ab.

Export: models/radioml2018-iq-v1 mit radioml_float.h5,
radioml_int8.tflite, preprocessing.json, labels.txt, report.json und
test_predictions.npz. Bericht enthaelt Float/INT8-Matrix, Recall und SNR.
Vorhandene Ausgabeordner werden abgelehnt; fuer Wiederholung --output nutzen.
Die alten Modelle und Imports bleiben erhalten.

Vergleich mit dem Spektrogramm nur auf denselben Testzeilen. Der bereits
angesehene Testsatz ist jetzt ein Entwicklungsvergleich; eine abschliessende
Bewertung braucht spaeter einen weiteren unangetasteten Testsatz.
Disjunkte Zeilen beweisen keine unabhaengigen Aufnahmen. RadioML bleibt
synthetisch; kein Nachweis fuer echte Empfangssignale.

## Validierung und naechster Schritt

```bat
python -m unittest test_radioml_iq test_train_radioml -v
```

Fuenf lokale Tests bestanden: RMS-/Skalierungsverhalten, Phasendrehung,
Eingabevalidierung, Split-Ueberschneidungen und bisherige Datenpruefungen.
TensorFlow-Training, Konvertierung und vollstaendiger HDF5-Import wurden hier
nicht ausgefuehrt. Nach Nutzer-Training folgt der Compilerbericht und ein
passender Coral-Test. Bestehende Drei-Klassen-Inferenzskripte passen nicht
zu diesem Eingang. TPU-Abdeckung und Genauigkeit sind noch offen.
