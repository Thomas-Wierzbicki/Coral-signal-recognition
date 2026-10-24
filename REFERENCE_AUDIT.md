# Referenzaudit und verbindliche Einordnung

Stand: 24.10.2026. Gegenstand ist die Quellen- und Protokollprüfung, nicht eine bereits erfolgreiche Reproduktion der Literaturergebnisse. Das Projektziel bleibt die Erkennung aller zwölf vereinbarten Kategorien; Drohnenprotokollerkennung bleibt das anschließende eigene Arbeitspaket.

## Ergebnis

Unsere bisherigen Harper-Läufe sind **eine Anpassung einer veröffentlichten Architektur**, keine Reproduktion des Autorenexperiments. Die Anpassung lernt auf einer kleinen Auswahl aller zwölf Klassen, stagniert jedoch im vollständigen bisherigen Trainingsprotokoll. Eine Ursache dafür ist noch nicht isoliert.

Auch die Originalquellen definieren kein widerspruchsfreies Einzelrezept: Preprint und öffentliches Trainingsskript unterscheiden sich bei Batchgröße und Steuergröße. Eine Reproduktion des festgelegten Codes muss deshalb von einer Untersuchung nach der Beschreibung im Text unterschieden werden. Keine der beiden darf als Nachweis realer RF-Erkennung gelten.

## Quellenbasis

Geprüfter Autoren-Commit: `902cca45c3bf04b5da9f78030372b5a03f730f42` im [Autoren-Repository](https://github.com/caharper/Automatic-Modulation-Classification-with-Deep-Neural-Networks/tree/902cca45c3bf04b5da9f78030372b5a03f730f42).

Textquelle für die nachfolgenden Literaturangaben: [arXiv 2301.11773v1, Abschnitte III, IV und Tabelle III](https://arxiv.org/html/2301.11773v1). Die finale Zeitschriftenfassung wird separat durch [DOI 10.3390/electronics12183962](https://doi.org/10.3390/electronics12183962) identifiziert. Ihr vollständiger Methodenabschnitt konnte in diesem Audit nicht unabhängig abgerufen werden; eine Übereinstimmung mit dem Preprint wird nicht vorausgesetzt.

Datei-URLs und Git-Blob-IDs sind in `reference_audit_evidence.json` festgehalten. Die Feststellungen stammen aus tatsächlich gelesenen Dateien: `models.py`, beide `layers.py`, `run.py`, `loader.py`, `process_dataset.py`, Sweep-Konfiguration, `requirements.txt` und `setup.py`.

## Abgleich

| Gegenstand | Preprint bzw. Autoren-Code | Unser bisheriger Versuch | Bewertung |
|---|---|---|---|
| Daten | RadioML 2018.01A | eigenes TorchSig-/FM-Gemisch | Andere Aufgabe und andere Verteilung |
| Kategorien | 24 | 12 | Keine direkt vergleichbare Genauigkeit |
| Eingabe | 1024 I/Q-Paare, TensorFlow channels-last | 4096 I/Q-Paare, PyTorch-Adapter channels-first | Form und Layout geändert; Layout muss numerisch geprüft werden |
| Amplitude | Konverter schreibt X-Werte unverändert; Loader zeigt keine weitere RMS-Skalierung | jeder komplexe Block auf RMS 0,5 | Explizite Änderung; tatsächliche Originalpegel müssen gemessen werden |
| Datenaufteilung | veröffentlichte Indexdateien | eigene Seed-getrennte 5040/1080/1080 Beispiele | Andere Aufteilung und Größenordnung |
| Adam-Lernrate | 0,0001 | 0,001 | Zehnfach verändert; Ursache des Stillstands damit nicht bewiesen |
| Optimierer | TensorFlow 2.4.1 Adam laut Requirements; `clipnorm=1` | PyTorch 2.9.1 Adam; kein Clipping | Nicht dieselbe Aktualisierungsvorschrift |
| Batchgröße | Preprint: 32; Sweep-Datei: 128 | 32 | Quellenwiderspruch; nicht stillschweigend auflösen |
| Lernratensteuerung | Preprint: Validierungsverlust; Code: Trainingsverlust, Faktor 0,1 nach 12 Epochen, Minimum 1e-7 | keine Reduktion | Quellenwiderspruch plus eigene Änderung |
| Abbruch / Gewichtsauswahl | Preprint: Validierungsverlust; Code: Trainingsverlust, Geduld 20, beste Gewichte zurückladen | Validierungsverlust, Geduld 8 | Unterschiedliche Konvergenz- und Auswahlregel |
| Epochenobergrenze | Sweep-Datei: 1000 | 40 | Stark verändertes Budget; nicht alleinige Erklärung nachgewiesen |
| Auswertung | Code übergibt `test_ds` als `validation_data` und wertet denselben Split erneut aus | separater Validierungssplit, danach Test; Test inzwischen für Entwicklung genutzt | Historisches Benchmarkverfahren ist kein versiegelter Projekt-Abschlusstest |
| Metriken | Genauigkeit, Top-2, SNR-, Familienwerte, Maximum über SNR | Genauigkeit, Macro-F1, Klassenwerte, SNR und Frequenzkontrollen | Metriken und Mittelungen müssen zur Fragestellung passen |
| Umgebung | TF 2.4.1 / NumPy 1.19.2; kein vollständiger Environment-Lock | Python 3.12 / Torch 2.9.1 / NumPy 2.5.2 beim Nutzer | Reproduktion benötigt eine getrennte alte Umgebung |

Die Übergabe des Testsplits zur Validierungsanzeige im Originalcode bedeutet nicht automatisch, dass dessen Gradienten verwendet werden: Die sichtbaren Callbacks überwachen `loss`, nicht `val_loss`. Dennoch ist der angezeigte Benchmarksplit kein unangetasteter Abschlussnachweis. Dies ist präziser als die pauschale Behauptung eines nachgewiesenen Trainingslecks.

Der Loader mischt Dateinamen und verschachtelt TFRecords; er zeigt keinen zusätzlichen elementweisen Shuffle nach dem Einlesen der Beispiele. Der Konverter mischt allerdings die Indizes vor dem Schreiben. Die tatsächliche Batchverteilung muss aus den erzeugten Batches gemessen werden, nicht aus dem Wort „shuffle“ abgeleitet werden.

## Architekturprüfung: bestanden, aber begrenzt

Die geprüfte Standardkonfiguration verwendet SE, Dilatation und letzte ReLU, keine Selbstaufmerksamkeit und keine residuale SE-Verbindung. Mit 24 Ausgaben besitzt unsere Portierung 202880 Parameter; das stimmt mit der entsprechenden Größenordnung in der Literaturtabelle exakt überein. Mit zwölf Ausgaben sind es 201332. Die Differenz beträgt 1548 = 12 × (128 + 1), entsprechend der geänderten letzten Schicht.

Lokal wurde zusätzlich der Vorwärtslauf bei 1024 Samples und 24 Ausgaben geprüft. Vorhanden sind NumPy-Gleichungsprüfung, Gradienten-/Exporttests und der erfolgreiche kleine GPU-Lerntest. **Nicht vorhanden ist ein direkter TensorFlow–PyTorch-Vergleich mit übertragenen identischen Gewichten.** Parameterzahl und NumPy-Prüfung ersetzen ihn nicht.

## Belegte Ergebnisse aus den bereitgestellten Nutzerprotokollen

| Versuch | Ergebnis | Zulässige Aussage |
|---|---|---|
| Eigenes CNN, drei Seeds | 55,4 % mittlere Entwicklungsgenauigkeit, Streuung 0,65 Prozentpunkte, Macro-F1 0,529 | Unter bisherigem Protokoll reproduzierbar begrenzte Leistung |
| Harper-Portierung, drei Seeds | jeweils 8,33 %, jeweils konstante Klasse | Vollständiges bisheriges Training stagniert |
| Harper-Verlauf, Seed 2051 | Verlust rund ln(12), Validierung stets 8,33 % | Problem besteht vor Export und Abschlussauswertung |
| GPU-Lerntest, bisheriges CNN | 24/24 richtig nach 75 Schritten, Verlust 0,0162 | Kleine Trainingsauswahl lernbar |
| GPU-Lerntest, Harper 0,001 | 24/24 nach 225 Schritten, Verlust 0,0646 | Höhere Lernrate verhindert Lernen nicht grundsätzlich |
| GPU-Lerntest, Harper 0,0001 | 24/24 nach 275 Schritten, Verlust 0,0464 | Auch niedrigere Lernrate lernt diese Auswahl |

Die beiden Harper-Lerntests starteten mit identischen Gewichten. Sie sagen nichts über neue Beispiele oder reale Signale aus. Ein Upload von Gewichten oder Rohdaten war für diese Einordnung nicht erforderlich. Die Originaldateien bleiben unverändert; die hier veröffentlichten Feststellungen sind Zusammenfassungen, keine Veröffentlichung der hochgeladenen Dateien.

## Offene Reproduktionsvoraussetzungen

1. TensorFlow 2.4.1 hat veröffentlichte Binärpakete für Python 3.6–3.8, nicht für Python 3.12 ([Paketdateien](https://pypi.org/project/tensorflow/2.4.1/)). Die funktionierende TorchSig-Umgebung wird deshalb nicht umgebaut. Eine historische Referenzumgebung ist separat aufzubauen und zu prüfen; GPU-Kompatibilität unter der aktuellen WSL-/Treiberkombination wird nicht behauptet.
2. Die Requirements enthalten nicht alle unmittelbar importierten Werkzeuge: unter anderem `bocas`, `ml_collections`, `termcolor` und `pandas` fehlen als explizite Einträge; `setup.py` hat leere `install_requires`. Ein vollständiger reproduzierbarer Lock fehlt. Versionen müssen ermittelt und eingefroren werden, statt beliebige aktuelle Pakete zu installieren.
3. `config.version` wird im Runner verwendet, ist in der sichtbaren Sweep-Konfiguration aber nicht gesetzt. Ob der Launcher es ergänzt, ist vor Ausführung zu klären. Daraus wird noch kein sicherer Laufzeitfehler abgeleitet.
4. Vor einer historischen Datenreproduktion sind Original-HDF5, Klassenreihenfolge und Autoren-Indexdateien zu prüfen. Die früher gemeldete HDF5-Prüfsumme identifiziert die Datei, beweist aber allein nicht die Aufteilung der Studie. Deren Quellindizes und tatsächliche Batchverteilung wurden hier noch nicht ausgeführt.
5. RadioML bleibt eine historische Kontrollaufgabe. Der [Anbieter nennt bekannte Errata](https://www.deepsig.ai/datasets/); eine erfolgreiche Reproduktion ersetzt weder zwölfklassenbezogene Validierung noch reale Empfangstests.

## Entscheidung und nächster ausführbarer Auftrag

Die nächsten Arbeitspakete sind in `EXPERIMENT_PROTOCOL_V1.json` festgelegt. Zuerst wird die getrennte Referenzumgebung hergestellt und geprüft, danach der direkte numerische Frameworkvergleich. Erst dann wird eine historische Code-Reproduktion sinnvoll; eine vollständige Literaturleistungsreproduktion darf erst nach Klärung der Quellenunterschiede so bezeichnet werden.

Für die zwölf Projektklassen folgt nach den technischen Prüfungen ein explizit als neue Aufgabe gekennzeichneter Transfer. Er ist kein isolierter Architekturvergleich zur RadioML-Aufgabe. Ein Batch-Vergleich bleibt ein nachgelagerter Diagnoseversuch, nicht die bereits feststehende Lösung. Die zwölf Klassen werden im Projekt nicht reduziert.

Aktueller Status: Quellenprüfung abgeschlossen; historische Umgebung, direkter Frameworkvergleich und Literaturleistungsreproduktion **nicht durchgeführt**. Es wird derzeit kein weiterer vollständiger Trainingslauf zur Ausführung empfohlen. Die fehlenden Nachweise werden nicht durch neue Genauigkeitsversprechen ersetzt.
