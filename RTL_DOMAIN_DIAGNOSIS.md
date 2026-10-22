# Diagnose: RadioML und reales NFM

## Ausgangspunkt

Realer 438,800-MHz-Kanal, 60 s, 32 kS/s. Vortrag nach FM-Demodulation vom Nutzer
als gut verstaendlich bestaetigt. Unveraendertes RadioML-I/Q-Modell:
FM 192/1874 (10,2 %), QPSK 1682/1874 (89,8 %), andere Klassen null.
Quantisierungs-Clipping null. invoke-Median 0,422 ms, p95 0,680 ms.
Dies ist kein Nachweis fuer einen QPSK-Sender und keine belastbare Accuracy-
Schaetzung. Die Hoerreferenz gilt der Aufnahme, nicht einzeln annotierten Bloecken.

## Diagnose ausfuehren

```bat
git pull --ff-only
conda activate coral
python diagnose_rtl_domain.py "data\recordings\rtl-20261022-171509-339614\channel-analysis-v1"
```

Kein Coral-Zugriff, Training, Resampling oder Modellwechsel. Keine neuen Pakete.
Liest den realen Kanal und ausschliesslich train.npz unter data/radioml2018-iq-v1.
Kein Zugriff auf Test-/Validierungsarrays. Alle verfuegbaren Trainingsbeispiele
und alle bisherigen 1874 realen Bloecke werden untersucht. RMS-Normalisierung
ist wie beim Modell; die Kanaldatei bleibt unveraendert.

Ausgabe channel-analysis-v1/domain-diagnosis-v1: features.csv und summary.json.
Bestehende Ordner werden abgelehnt; fuer Wiederholung --output verwenden.
Bericht enthaelt Kanal-/Trainingsdateihashes sowie Metadaten.
Falls coral-real-v1/CLASSIFICATION_COMPLETE vorhanden ist, werden dessen
Vorhersagen nach Kanalhash, Blockanzahl, Startindex und Zeitfolge geprueft und
fuer Gruppen real/predicted_FM und real/predicted_QPSK verwendet.
Es erfolgt dabei keine neue Inferenz. Modell unveraendert lassen.

## Kennwerte

| Kennwert | Bedeutung / Einheit |
|---|---|
| envelope_cv | Standardabweichung / Mittelwert der komplexen Amplitude |
| phase_offset_cycles_per_sample | Winkel des mittleren Einheits-Phasenschrittvektors, geteilt durch 2 pi |
| phase_spread_rad | Standardabweichung zirkulaer um diesen Winkel zentrierter Phasenschritte |
| phase_step_change_rms_rad | RMS der gewrappten Aenderung benachbarter Phasenschritte |
| lag1_phase_coherence | Betrag des mittleren Einheits-Phasenschrittvektors, 0..1 |
| dc_power_fraction | Leistung des komplexen Mittelwerts / mittlere Gesamtleistung |
| spectral_width90_cycles_per_sample | Abstand der 5/95-%-Leistungsquantile nach Phasenschritt-Zentrierung, Hann-FFT |

Fuer jede Gruppe p10, Median und p90. Konsole: real/all sowie RadioML-FM/QPSK
bei SNR >=10 dB. JSON enthaelt zusaetzlich alle fuenf Klassen, alle SNR und
reale Vorhersagegruppen. SNR >=10 dB ist eine beschreibende Auswahl,
keine SNR-Schaetzung fuer die reale Aufnahme.

RadioML hat hier keine physikalische Hz-Abtastrate. Deshalb Vergleiche in
Zyklen/Sample und Radiant; keine erfundene Hz-Gleichsetzung. Die Kreisstatistik
wird bei stark verteilten Phasenschritten instabil. Die Spektrumszentrierung
dient nur dieser Kennzahl, veraendert keine Modelleingaenge und beweist keinen
Traegerfrequenzfehler. 90-%-Bandbreite ist eine FFT-Schaetzung, keine kalibrierte
Kanalbandbreite. Rauschen, Gleichanteil und Filterung beeinflussen die Kennwerte.

Die Diagnose beschreibt Unterschiede, isoliert keine Ursache. Insbesondere
kann man daraus weder ableiten, dass die Aufnahme QPSK sei, noch eine nachtraeglich
angepasste Vorverarbeitung bereits als validierte Verbesserung bezeichnen.
Erst den unveraenderten Vergleich festhalten, danach kontrollierte Einzeltests.

## Tests und Sicherung

```bat
python -m unittest test_rtl_domain test_rtl_classify -v
python checkpoint_radioml.py
```

Fuenf Tests bestanden: bekannte Traegerschritte, Amplituden-/Phasenrotations-
invarianz, veraenderliche FM-Phasenschritte, ungueltige Bloecke und kompletter
Diagnosebericht mit ausschliesslich Trainings-NPZ. Klassifikationstest mit
simuliertem Interpreter. Echte Diagnose erfolgt am Nutzerrechner.
Die erneute Sicherung nach Realtest wurde noch nicht bestaetigt; sie nimmt
auch neue Diagnoseausgaben unter data/ mit auf.
