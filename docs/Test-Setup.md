# Test-Setup — LatinLearner

## Test-Framework

**Gewählt:** pytest
**Begründung:** De-facto-Standard, weniger Boilerplate, bessere Ausgabe, Fixtures, Parametrisierung. Bestehende unittest-Tests laufen ohne Änderung auch unter pytest.

## Vorhandene Tests

### Übersicht

| Datei | Modul unter Test | Anzahl Tests | Status |
|---|---|---|---|
| `text_unit_tests.py` | `clean_data.py`, `text_handling.py`, `embedding.py` | 12 | Framework-unabhängig (bis auf `TestWordEmbedding`) |
| `LSTM_unit_tests.py` | `LSTM_model.py` | 3 | TF 1.x-abhängig, nicht lauffähig |
| `word2vec_test.py` | `word2vec.py` | 1 | TF 1.x-abhängig, nicht lauffähig |

### Detailbewertung

**`text_unit_tests.py`**
- `TestCleaning` (8 Tests): Testen die Textbereinigungsfunktionen in `clean_data.py`. **Sofort portierbar** — keine TF-Abhängigkeit. Benötigen nur die Testdaten in `test_library/`.
- `TestReadingData` (4 Tests): Testen die Datenlade- und Aufbereitungsfunktionen in `text_handling.py`. **Sofort portierbar** — abhängig von `nltk` und `numpy`, aber nicht von TensorFlow.
- `TestWordEmbedding` (1 Test): Testet `embedding.py`. **Nicht portierbar** ohne Modell-Migration, da `embedding.py` `tensorflow.contrib.keras` importiert.

**`LSTM_unit_tests.py`**
- `TestLSTMModel` (3 Tests): Testen Batch-Generierung, Sampling und ein vollständiges Mini-Training. **Nicht portierbar** — komplett an TF 1.x-API gebunden (`tf.placeholder`, `tf.Session`, `tf.contrib.rnn`).

**`word2vec_test.py`**
- `Word2VecTest` (1 Test): TensorFlow-Tutorial-Testcode. **Nicht portierbar** — setzt kompilierte Custom Ops voraus.

## Geplante Test-Struktur nach Portierung

```
tests/
├── conftest.py              # Gemeinsame Fixtures (Testdaten laden etc.)
├── test_clean_data.py       # Tests für Textbereinigung (portiert aus text_unit_tests.py)
├── test_text_handling.py    # Tests für Datenaufbereitung (portiert aus text_unit_tests.py)
├── test_model.py            # Tests für das neue Modell (LSTM oder Transformer)
└── test_scraper.py          # Tests für den Scraper (neu)
```

## Portierungsplan für Tests

### Phase 1 — Sofort (framework-unabhängige Tests)
1. `tests/`-Verzeichnis erstellen mit `conftest.py`
2. `TestCleaning` und `TestReadingData` nach `tests/test_clean_data.py` und `tests/test_text_handling.py` portieren
3. Auf pytest-Stil umschreiben (assert statt self.assertEqual, Fixtures statt setUp)
4. Sicherstellen, dass `nltk` und `numpy` in aktueller Version funktionieren

### Phase 2 — Nach Modell-Portierung
1. Modell-Tests für das neue Framework schreiben (PyTorch/TF2)
2. Scraper-Tests mit gemockten HTTP-Responses hinzufügen
3. Integrationstests für die gesamte Pipeline (Scrape → Clean → Train → Sample)

## Testdaten

- **`test_library/`**: Drei kurze lateinische Texte (Augustus, Caesar, Catullus) — insgesamt ~750 Zeichen. Dienen als Miniatur-Datensatz für schnelle Unit-Tests.
- **`small_library/`**: Größere Texte für manuelle Experimente und ggf. Integrationstests.

## Externe Dienste

Aktuell werden keine externen Dienste für Tests benötigt (kein Datenbankzugriff, kein Message-Broker). Der Scraper greift auf thelatinlibrary.com zu — für Tests sollte das über Mocking (`responses` oder `pytest-httpserver`) gelöst werden, um nicht von der Webseite abhängig zu sein.

Eine `docker-compose.yml` ist derzeit nicht erforderlich.

## Tests ausführen

### Voraussetzungen
- Python ≥ 3.10 (nach Modernisierung)
- `pip install pytest numpy nltk`
- NLTK-Daten: `python -c "import nltk; nltk.download('punkt')"`

### Befehle

```bash
# Alle Tests
pytest

# Nur Textbereinigung
pytest tests/test_clean_data.py

# Mit ausführlicher Ausgabe
pytest -v

# Mit Coverage
pip install pytest-cov
pytest --cov=. --cov-report=term-missing
```
