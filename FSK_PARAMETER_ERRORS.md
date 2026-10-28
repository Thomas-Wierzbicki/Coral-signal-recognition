# FSK-Parameter und vorhandene Fehlvorhersagen

Plan: `FSK_PARAMETER_JOIN_PLAN_V1.json`. Die Grenzen sind vor dem echten
Datenlauf festgelegt. Der Lauf benoetigt nur die Python-Standardbibliothek.
Keine GPU-Inferenz, kein Training und kein Oeffnen des Testsatzes.

Aus `~/rf-learning`:

```bash
python project/join_fsk_parameter_errors.py
```

Der Lauf sucht genau einen abgeschlossenen Fehleraudit unter `models` und
genau eine abgeschlossene Generator-Rekonstruktion unter `data`.
Bei mehreren Treffern bitte explizit angeben:

```bash
python project/join_fsk_parameter_errors.py --audit models/DEIN-LAUF/error-audit-ZEITSTEMPEL --replay data/generator-replay-ZEITSTEMPEL
```

Optional `--data data/modulation12-v1`, `--output NEUER-ORDNER`.
Standardausgabe: `data/fsk-parameter-errors-ZEITSTEMPEL/report.json`.
Zum Besprechen diesen **report.json** hochladen. Der Lauf ueberschreibt nichts.

Alle vier Modelle (Seeds 2081/2082, keep_lr/lower_lr) bleiben getrennt.
Die Trainingsauswahl ist die bestehende Teilmenge; die Validierung muss
vollstaendig sein. Alle Modelle muessen dieselben Quell-IDs verwenden.
Vollstaendige CSV-Matrizen aller zwoelf Klassen werden mit dem Audit verglichen.
Nur die zusaetzlichen Parametertabellen beschraenken sich auf die FSK-Familie.

Die Verknuepfung prueft gespeicherte Datei-Hashes, Replay-Erfolg und IQ-Digests,
Quell-IDs, Labels, Bedingungen und FSK-SNR. Sie liest keine IQ-NPZ-Datei erneut;
deren exakter Inhalt wurde im vorangegangenen Replay geprueft.
Sie fuehrt keine unabhaengige neue Checkpoint-/Modellvalidierung durch.

Tabellen je Modell, Split und fuer alle SNR sowie separat SNR>=15 dB:

- Klasse nach h-Bereich;
- Klasse nach BT-Bereich;
- Klasse nach Gaussian-Span;
- sekundaer Klasse nach h- und BT-Bereich gemeinsam.

Jede Zelle zeigt Anzahl, korrekte Vorhersagen, Trefferquote, vorhergesagte
Klassen und Anzahl verschiedener Bedingungen. Zellen mit weniger als 20
Beispielen sind `sparse`. h=0.5 wird ausdruecklich getrennt von h<0.5 gehalten.
Bei 2FSK sind BT und Gaussian-Span nicht anwendbar.

Es handelt sich um deskriptive Entwicklungsauswertung. Unterschiedliche
h-Gruppen koennen sich zugleich in SNR, Bandbreite und Offset unterscheiden.
Ein Zusammenhang ist keine isolierte Ursache; vier Modelle auf denselben
Beispielen sind keine vier unabhaengigen Datensaetze. Keine automatische
Klassenentfernung, Parametersuche oder weitere Trainingsrunde.

Erst wenn alles besteht, entsteht `FSK_PARAMETER_JOIN_COMPLETE`.
Fehler bleiben im neuen Ausgabeordner als Bericht erhalten.

Pruefung der Implementierung mit synthetischen Fixtures:

```bash
python -m unittest test_fsk_parameter_join
```
