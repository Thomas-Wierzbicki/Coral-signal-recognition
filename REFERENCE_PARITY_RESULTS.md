# Direkte Prüfung der Originalreferenz: Umgebung und numerische Gleichwertigkeit

**Aktualisierung:** Die zwei Restfälle wurden inzwischen nach Eingabe, Operator und Präzision lokalisiert: [GRADIENT_LOCALIZATION_RESULTS.md](GRADIENT_LOCALIZATION_RESULTS.md). Ein einzelner ReLU-Maskenwechsel und eine isoliert reproduzierte TensorFlow-Float32-Bias-Summierung erklären die konkreten Befunde. Die historischen FP32-Fehlschläge und Toleranzen bleiben bestehen; die Ursache des vollständigen Trainingsstillstands ist weiterhin nicht nachgewiesen.

Stand: 24.10.2026. **Ergebnis: Originalumgebung läuft; die Gesamtfreigabe für die Zwölfklassen-Portierung ist noch nicht erreicht.** Es wurde kein neues Erkennungsmodell auf dem vollständigen Datensatz trainiert.

## Tatsächlich ausgeführte Umgebung

Eine separate Linux-CPU-Umgebung wurde hier eingerichtet: Python 3.8.20, TensorFlow 2.4.1, NumPy 1.19.2 und die ursprünglichen Paketvorgaben. Fehlende direkte Abhängigkeiten wurden explizit ergänzt und alle aufgelösten Versionen in `reference-tf241-linux-py38.lock.txt` festgehalten. `pip check` meldet keine gebrochenen Abhängigkeiten. Der Original-Runner und seine Abhängigkeiten importieren; das Originalmodell baut mit 202880 Parametern, Eingabe `(None,1024,2)` und Ausgabe `(None,24)`.

Die relevanten Autoren-Dateien liegen unverändert unter `third_party/harper2023_original`, mit MIT-Lizenz und `SOURCE_MANIFEST.json`. Dies ist ein Manifest der benötigten Quelldateien, keine vollständige Kopie des historischen Datensatzes oder sämtlicher Repository-Dateien. Commit: `902cca45c3bf04b5da9f78030372b5a03f730f42`.

Eine konkrete Launcher-Lücke wurde bestätigt: `config.version` fehlt und wird auch von der geprüften Bocas-Version 0.0.3 nicht ergänzt. Der Exportadapter löst die Sweep-Optionen explizit auf und ergänzt ausschließlich den Namen `reference-validation-v1`. Er ändert weder Original-Modellcode noch wissenschaftliche Trainingsparameter. Ein vollständiger Lauf des ursprünglichen Trainingsrunners wurde damit noch nicht durchgeführt. Details: `reports/reference_environment_cpu.json`.

## Vorab festgelegte Prüfung

Geprüft wurden Originalform 1024 Samples / 24 Ausgaben und Projektform 4096 Samples / zwölf Ausgaben, jeweils mit Seeds 901, 902, 903. Jede Prüfgruppe enthält Nullsignal, amplitudenveränderlichen Chirp und komplexes gaußsches Rauschen. Das sind technische Eingabefälle, keine drei Erkennungsklassen und kein Ersatz für die zwölf Projektkategorien.

Der unveränderte TensorFlow-Code erzeugt die tatsächlichen Gewichte und Eingaben. Diese werden anschließend in PyTorch übertragen; Kernelachsen werden explizit transponiert. Verglichen werden Zwischenausgaben, Statistik-Pooling, Logits, Softmax, Verlust, sämtliche 32 Parametergradienten und die Formel des ersten Adam-Schritts. Der Gradient stammt aus einer festen mittleren Kreuzentropie über dieselben drei Eingaben; keine Gewichte werden zwischen den Fällen weitertrainiert.

Regel unverändert aus Protokoll V1: Für jedes Arrayelement gilt `abs(Abweichung) <= 1e-6 + 1e-4 * abs(TensorFlow-Wert)`, außerdem gleiche Form und endliche Werte. Es wurden keine Seeds entfernt, keine Eingaben nachträglich ausgetauscht und keine Toleranzen gelockert.

## Ergebnisse einschließlich Fehlschlägen

| Kandidat / CPU-Profil | Prüfgruppen bestanden | Auswertung |
|---|---:|---|
| Bisherige PyTorch-SELU, Standard-Faltung | 0/6 | Systematischer SELU-Grenzfall; zusätzlich ein weiterer Gradientengrenzfall |
| TensorFlow-kompatible SELU, Standard-Faltung | 5/6 | Ein Conv5-Bias-Gradient in 4096/12/Seed903 überschreitet die Grenze |
| TensorFlow-kompatible SELU, native Faltung ohne oneDNN | 5/6 | Ein Conv7-Bias-Gradient in 4096/12/Seed901 überschreitet die Grenze |

Pro Profil wurden 774 Einzelvergleiche durchgeführt. Nach SELU-Anpassung bestehen jeweils 773; die fehlende Prüfung verhindert trotzdem das Bestehen des Gesamtprofils. Für die ursprüngliche 1024/24-Form bestehen nach SELU-Anpassung in beiden CPU-Profilen alle drei Gruppen. Alle Vorwärtsprüfungen bestehen nach der Anpassung, die verbleibenden Fehler betreffen Gradienten.

### Bestätigter semantischer Unterschied: SELU bei exakt null

Bei `x=0` liefert TensorFlow 2.4.1 eine Ableitung von ungefähr 1,050701, PyTorch 2.9.1 ungefähr 1,758099. Für `x=-0,1` und `x=+0,1` stimmen die überprüften Ableitungen überein. Das betrifft am Nullsignal die Bias-Gradienten der beiden dichten SELU-Stufen, obwohl die Vorwärtswerte gleich sind.

Die Korrektur ist als `Harper2023(tensorflow_selu=True)` ausdrücklich auswählbar. `TensorFlowSELU` verwendet bei null den positiven Zweig; eine begrenzte Exponentialberechnung verhindert unnötige Überläufe im nicht gewählten Zweig. Der historische Standard bleibt `tensorflow_selu=False`, damit alte Versuche nicht stillschweigend ein anderes Modellverhalten erhalten. Die vollständigen Trainingsskripte wurden nicht auf den neuen Kandidaten umgestellt.

Die Korrekturabsicht wurde vor ihrer vollständigen Bewertung öffentlich festgehalten: `EXPERIMENT_PROTOCOL_AMENDMENT_01.json`, Commit `4f12856808edb46c66ff51c09e2699f1e1528d11`. Präzisierung des damaligen Begründungstextes: Der vollständige Originalbericht enthielt neben zwölf SELU-bezogenen Bias-Abweichungen bereits einen zusätzlichen Conv5-Gradientengrenzfall. Die Formulierung „only two ... per fixture batch“ in der Änderungsbegründung war insofern unvollständig. Der unveränderte Bericht enthält alle 13 fehlgeschlagenen Einzelvergleiche und ist maßgeblich.

### Verbleibende Float32-/Backendabhängigkeit

Im Standardprofil unterscheiden sich bei 4096/12/Seed903 die Aktivitätsentscheidungen genau einer Conv5-ReLU nahe null; der TensorFlow-Ausgabewert an dieser Stelle liegt bei etwa 9,45e-10. Der maximale Conv5-Bias-Gradientenfehler beträgt 3,21e-6 und überschreitet die zugehörige kombinierte Toleranz.

Ein explorativer Einzeltest ohne oneDNN beseitigte diesen konkreten Conv5-Befund. Danach wurde das vollständige native Profil vorab als eigener Versuch registriert: `EXPERIMENT_PROTOCOL_AMENDMENT_02.json`, Commit `e0ef62849b8c16c23995163edc75645da5673915`. Das native Profil besteht insgesamt ebenfalls nicht: Bei Seed901 bleibt ein Conv7-Bias-Gradientenfehler von 1,02e-5. Im zugehörigen zusätzlichen Maskenvergleich wurden keine unterschiedlichen Conv-ReLU-Aktivitätsentscheidungen gefunden. Dessen genaue numerische Entstehung ist damit noch nicht isoliert.

Diese Ergebnisse werden nicht durch die Auswahl des jeweils günstigeren Backends je Seed zusammengeführt. Es gibt keine Gesamtfreigabe aus einem solchen Mischprofil. Float32-Grenzfälle und ein systematischer Ableitungsunterschied sind außerdem nicht automatisch die Ursache des beobachteten vollständigen Trainingsstillstands.

### Adam: tatsächlicher Unterschied nachgewiesen, Ursache des Lernstillstands offen

Beim ersten Optimiererschritt wurde mit denselben exportierten Gradienten gearbeitet, einschließlich Clipping jeder Parametertensor-Norm auf eins. Die separat geprüfte TensorFlow-Erstschrittformel besteht die Toleranzen. Natives PyTorch-Adam weicht selbst bei derselben Lernrate und demselben numerischen epsilon ab; auch die Variante mit dem PyTorch-Standard-epsilon wurde separat erfasst. Das zeigt eine tatsächliche Optimiererabweichung und ersetzt keinen kontrollierten Trainingsvergleich.

Die Erstschrittformel ist ausschließlich ein Prüfhelfer, kein neuer einsatzfähiger Optimierer. Zustandsentwicklung über viele Schritte und ein vollständiges Training mit angeglichener Optimierung sind noch nicht geprüft.

## Reproduzieren ohne bestehende Umgebung umzubauen

Die folgenden Befehle dokumentieren den lokal geprüften Aufbau für Linux x86_64. Sie setzen `uv` voraus und sind für eine neue separate Umgebung bestimmt; ihre Ausführung auf dem Nutzerrechner ist bisher nicht bestätigt. Der Lock ist kein allgemeiner Windows-/GPU-Kompatibilitätsnachweis.

```bash
cd ~/rf-learning
uv python install 3.8
uv venv --python 3.8 .venv-reference-tf241
uv pip install --python .venv-reference-tf241/bin/python -r project/reference-tf241-linux-py38.lock.txt
.venv-reference-tf241/bin/python -m pip check

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python project/verify_reference_parity.py \
  --tf-python "$PWD/.venv-reference-tf241/bin/python" \
  --output models/reference-parity-default-v1
```

Das Werkzeug erzeugt die TF-Prüfdaten selbst in einem Unterordner und nutzt die vorhandene PyTorch-Umgebung nur für den Vergleich. Mit `--native-convolution` und einem anderen Ausgabeordner wird das getrennte native CPU-Profil geprüft. Bestehende Ordner werden abgewiesen. Die Fixture-Dateien und ausführlichen `details.json` können deutlich größer als die Zusammenfassung sein.

**Wichtig:** Für den derzeit dokumentierten Stand ist ein nicht bestandener Gesamttest mit Exitcode 1 zu erwarten. Das ist ein wissenschaftlicher Befund, kein Anlass, die Grenze nachträglich zu erhöhen. Zur bloßen Bestätigung dieses bekannten Ergebnisses ist kein weiterer Nutzerlauf erforderlich.

## Artefakte und aktueller Status

- `reports/reference_environment_cpu.json`: tatsächliche Umgebung und Abhängigkeitsprüfung.
- `reports/reference_parity_original_candidate.json`: ursprünglicher Fehlschlag.
- `reports/reference_parity_selu_corrected_default_backend.json`: Korrektur mit Standardprofil.
- `reports/reference_parity_native_backend.json`: getrennt geprüftes natives Profil.
- `REFERENCE_VALIDATION_STATUS.json`: Status gegenüber dem eingefrorenen Versuchsplan.

Neun Prüftests bestanden, darunter Kernelzuordnung, Ableitung bei null, endliche Berechnung bei großen positiven SELU-Eingaben, Gradienten und TorchScript des expliziten Kandidaten. Erfolgreiche Prüftests ersetzen den noch fehlenden Gesamt-Pass nicht.

Nächster begründeter Schritt ist die numerische Lokalisierung des verbleibenden Gradientenfehlers, bevor der G2-Gesamtnachweis oder neue vollständige Trainingsläufe freigegeben werden. Historisches RadioML-Training, neue zwölfklassige Leistungsbewertung, reale RF-Erkennung und Drohnenprotokollerkennung wurden in diesem Arbeitsschritt nicht nachgewiesen.
