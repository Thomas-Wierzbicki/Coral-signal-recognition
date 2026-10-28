# Exakte Generator-Rekonstruktion

Der vorab registrierte Plan steht in `GENERATOR_REPLAY_PLAN_V1.json`.
Dieser Lauf erzeugt keine neuen Trainingsdaten und trainiert kein Modell.
Er rekonstruiert alle vorhandenen Beispiele aus Training und Validierung,
einschliesslich aller zwoelf Klassen. Der Testsatz wird nicht geoeffnet.

Aus `~/rf-learning` in der bestehenden TorchSig-Umgebung:

```bash
python project/replay_modulation_generators.py
```

Standardquelle: `data/modulation12-v1`. Optional `--data PFAD`.
Die Ausgabe entsteht in einem neuen `data/generator-replay-ZEITSTEMPEL`;
ein bestehender Ausgabeordner wird abgelehnt.

Der Lauf prueft die gespeicherten NPZ-/Parameterdatei-Hashes, Versionen,
Builder-Codehash, Quell-IDs, Labels, Bedingungs-IDs, SNR und IQ-Hashes.
Die aktuelle Wrapper- und FSK-Quelle werden gehasht protokolliert;
deren historische Identitaet wird nicht allein aus Versionsnummern behauptet.
Entscheidend ist der bitgenaue Vergleich der tatsaechlichen IQ-Inhalte.

Bei FSK wird zuerst unveraendert rekonstruiert. Anschliessend beobachtet
ein Wrapper den echten Rueckgabewert von `fsk_random_params`, ohne einen
zusaetzlichen Zufallswert zu ziehen. Auch diese zweite Rekonstruktion muss
bitidentisch sein. Die Originalfunktion wird selbst bei Fehlern wiederhergestellt.
Pro Beispiel werden h, BT und Gaussian-Span aufgezeichnet, soweit geliefert.

Nur wenn **alle** Beispiele exakt passen, setzt der Bericht
`parameter_interpretation_allowed: true` und entsteht `REPLAY_COMPLETE`.
Abweichungen werden im `report.json` festgehalten; die Rekonstruktionen
duerfen dann nicht als bestaetigte Parameter der Originaldaten interpretiert werden.
Keine Toleranzanpassung, kein stiller Austausch von Daten.

Ausgaben: `report.json`, `train_replay.jsonl`, `validation_replay.jsonl`.
Zum Besprechen genuegt zunaechst **report.json**. Ein Erfolg bestaetigt
Reproduzierbarkeit, keine Modulationserkennungsleistung oder Fehlerursache.
Parameterueberschneidungen sind anschliessend deskriptiv zu untersuchen;
Klassen werden deshalb nicht automatisch zusammengelegt oder entfernt.

Lokale Tests mit synthetischen Ersatzgeneratoren:

```bash
python -m unittest test_generator_replay
```

Die Tests decken vollstaendigen Lauf ohne vorhandenen Testsatz, absichtliche
Rekonstruktionsabweichung mit gesperrter Interpretation sowie RNG-Neutralitaet
und Wiederherstellung nach einer Ausnahme ab. Der echte TorchSig-Lauf auf
den vorhandenen Daten muss auf Schokobaer stattfinden.
