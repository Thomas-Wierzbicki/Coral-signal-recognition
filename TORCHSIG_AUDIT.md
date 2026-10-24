# Generatoraudit fuer TorchSig 2.2.0

## Durchfuehrung unter Ubuntu/WSL

```bash
cd ~/rf-learning
source .venv/bin/activate
git -C project pull --ff-only
python project/audit_torchsig_generators.py --output data/torchsig-audit-v1
```

352 saubere Signale: elf Signalklassen x vier angeforderte Bandbreiten
(2/4/8/12 kHz) x acht unabhaengige Seeds. Rauschen ist bewusst nicht als
Modulationsklasse dieser Messung enthalten. Keine AWGN-Zugabe, kein Frequenz-
versatz, keine Klassifikation und kein Training. Generatoren sind unveraendert.
Wie bei der Probe 32 kS/s, 16384 Samples erzeugen, mittlere 4096 auswerten;
RRC-Rolloff fuer Konstellationen fest 0,35, OFDM 64 Untertraeger/CP16.
Die Generatoren variieren weitere interne Parameter weiterhin gemaess ihrem Code.

Ausgabe: cases.csv mit allen Messungen, summary.json mit p10/Median/p90 je
Klasse/Bandbreite sowie FM-internen Groessen, bandwidth_audit.png und
AUDIT_COMPLETE. Neuer Ausgabeordner erforderlich; Wiederholungen ueber --output.
Kein kompletter Trainingsdatensatz und keine I/Q-Archivierung in diesem Audit.

## Messung und FM-Instrumentierung

90-%-Bandbreite aus Hann-FFT, Amplituden-CV und Statistik der benachbarten
Phasendifferenzen. FM-Hub-p95 ist das 95-%-Quantil des Betrags der um ihren
Median zentrierten Momentanfrequenz. Kein kalibrierter Spitzenhub und keine
Carson-Bandbreitenmessung. Digitale Phasenspruenge und Amplitudennullstellen
erschweren die Interpretation derselben Momentanfrequenzstatistik bei PSK/QAM.
Angeforderte 3-dB-Bandbreite ist eine andere Definition als gemessene 90-%-Breite.

Nur waehrend des FM-Aufrufs wird TorchSigs convolve-Funktion mit einem Wrapper
versehen, der das unveraenderte Ergebnis zurueckgibt und RMS vor/nach dem Filter
protokolliert. Keine dauerhafte Aenderung des installierten Pakets. Ein Test
bestaetigt bitidentische FM-Samples mit und ohne Wrapper. Die internen Parameter
Modulationsindex/fdev/Nachrichtencutoff werden aus dem zuvor geprueften
2.2.0-Code und dem ersten identischen RNG-Zug rekonstruiert. Versionsbindung und
SHA256 der installierten FM-Quelldatei werden im Bericht gespeichert.
RMS-Werte beziehen sich auf den vollstaendigen gefilterten Nachrichtenblock,
Phasendifferenzen auf den mittleren Signalausschnitt.

## Hier bestaetigte Ergebnisse

Vollstaendiger 352-Signal-Lauf mit TorchSig 2.2.0 und PyTorch 2.9.1 CPU.
Drei Tests bestanden: bekannte FM-Phasendifferenzen, unveraenderte Samples
bei Instrumentierung und vollstaendiges balanciertes Messraster mit simuliertem
Generator/Ueberschreibschutz. Grafik visuell kontrolliert.

| Angeforderte Bandbreite | FM: gemessene 90-%-Breite, Median | FM: Nachrichten-RMS nach Filter, Median | FM: Hub-p95, Median |
|---|---:|---:|---:|
| 2000 Hz | 289,1 Hz | 0,1041 | 172,9 Hz |
| 4000 Hz | 726,6 Hz | 0,1318 | 442,5 Hz |
| 8000 Hz | 2289,1 Hz | 0,2093 | 1357,3 Hz |
| 12000 Hz | 4043,0 Hz | 0,2537 | 2442,4 Hz |

Die Nachricht ist vor Filterung auf RMS 1 normiert. Im geprueften Generator
folgt nach der Filterung keine erneute Nachrichtennormierung vor der FM-
Integration; dadurch wirkt fdev nicht als garantierter realisierter Spitzenhub.
Das erklaert einen konkreten Skalierungseffekt, ist aber allein weder ein
universeller Fehlernachweis noch der Beweis fuer das Scheitern eines Klassifikators.
Auch interne Modulationsindizes und Nachrichtenbandbreiten beeinflussen die Breite.
PSK/QAM/OFDM liegen im selben Lauf mit 4-kHz-Vorgabe bei rund 3,6..3,8 kHz
belegter 90-%-Breite. FSK-Varianten zeigen wiederum eigene Verhaeltnisse.

Naechste Entscheidung: Fuer physikalisch kontrollierte analoge Referenzen
Nachrichtenband und Hub explizit definieren und protokollieren; TorchSig-Original
als eigene Generatorvariante erhalten. Nicht alle Klassen anhand derselben
angeforderten Zahl als gleichbandig behandeln. Erst danach breitere Trainings-
verteilungen mit ueberlappenden realisierten Merkmalen aufbauen. Modulations-
und Drohnenerkennung bleiben gleichrangige Ziele; dieser Audit validiert nur
Generatorverhalten, keine Drohnenprotokolle oder Praxiserkennungsleistung.

## Tests und Aufbewahrung

```bash
python -m unittest discover -s project -p test_torchsig_audit.py -v
python -m pip freeze > data/torchsig-audit-v1/environment-freeze.txt
```

Auditdaten und Probe getrennt behalten. Die alten Windows-Sicherungen enthalten
diese spaeter erzeugten WSL-Dateien nicht automatisch.
