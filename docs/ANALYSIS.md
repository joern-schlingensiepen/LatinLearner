# Projektanalyse — LatinLearner

## 2.1 Projektzweck

Das Projekt **LatinLearner** ist ein Deep-Learning-Projekt für die lateinische Sprache. Das übergeordnete Ziel ist der Aufbau eines lateinischen Chatbots sowie die Verbesserung des Verständnisses der lateinischen Sprache durch maschinelles Lernen.

Das Projekt verfolgt drei Hauptzwecke:

1. **Lernprojekt:** Technologien für ein Deep-Learning-System erlernen, das lateinische Sprache verarbeiten kann.
2. **Praxisübung:** Coding-Skills, GitHub-Workflows und Testing üben.
3. **Werkzeug:** Ein Sprachmodell bereitstellen, das in Schule, Universität oder als Web-/App-Tool genutzt werden kann.

### Abgleich Dokumentation ↔ Code

Die README-Dokumentation beschreibt den tatsächlichen Codestand insgesamt korrekt. Folgende Abweichungen bestehen:

- **`embedding.py`**: Die Methode `process()` ruft `self.train()` auf, obwohl die Methode `optimize()` heißt — das führt zu einem `AttributeError` zur Laufzeit. Die README erwähnt die Keras-Embedding-Variante als „nicht überzeugend", aber der Code ist tatsächlich fehlerhaft und nicht lauffähig.
- **`_compute_similarity`**: Als `@staticmethod` deklariert, nimmt aber `self` als Parameter entgegen — ein Designfehler, der den Aufruf verkompliziert (wird über `self._compute_similarity(self, ...)` umgangen).
- **Umgebung**: Die `environment.yml` referenziert TensorFlow 1.3 und Python 3.6.2 — beides ist stark veraltet und auf modernen Systemen nicht ohne Weiteres installierbar.
- **word2vec.py / word2vec_kernels.cc / word2vec_ops.cc**: Stammen direkt aus dem TensorFlow-Tutorial und setzen eine kompilierte `.so`-Datei voraus. Die Build-Anleitung in der README ist korrekt, aber die Abhängigkeit von TensorFlow 1.x macht sie auf aktuellen Systemen nicht nutzbar.

---

## 2.2 Einstiegspunkte

| Einstiegspunkt | Dateipfad | Beschreibung |
|---|---|---|
| Hauptprogramm | `latin_learner.py` | CLI-Entry-Point mit `argparse`. Steuert über `--mode` (LSTM/embedding), `--inference`, `--checkpoint` und `--datadir` den gesamten Ablauf. |
| Datenbereinigung | `clean_data.py` | Eigenständiges Skript (`if __name__ == "__main__"`). Bereinigt Texte aus dem `library/`-Verzeichnis und schreibt `cleaned_text.txt`. |
| Web-Scraper | `scrape_latin_texts.py` | Eigenständiges Skript. Scrapt lateinische Texte von thelatinlibrary.com und speichert sie als `.txt`-Dateien in `library/`. |
| Word2Vec (TF-Tutorial) | `word2vec.py` | Eigenständiges Skript via `tf.app.run()`. TensorFlow-Tutorial-basiertes word2vec-Training mit Custom Ops. |
| Jupyter Notebook | `LatinLeaRNNr.ipynb` | Interaktives Notebook mit dem Char-RNN-Code, vermutlich als Experimentierumgebung gedacht. |

---

## 2.3 Technologie-Stack

### Programmiersprache
- **Python 3.6.2** (laut `environment.yml`)
- **C++11** für die TensorFlow Custom Ops (`word2vec_kernels.cc`, `word2vec_ops.cc`)

### Frameworks
- **TensorFlow 1.3** — Kern-Framework für alle Modelle (LSTM, word2vec, Embedding)
- **Keras** (über `tensorflow.contrib.keras`) — nur für das Embedding-Modell in `embedding.py`

### Externe Bibliotheken (aus `environment.yml`)

| Bibliothek | Version | Verwendung |
|---|---|---|
| `tensorflow` | 1.3.0 | Deep-Learning-Framework (LSTM, word2vec, Embeddings) |
| `tensorflow-tensorboard` | 0.1.5 | Visualisierung von Training und Embeddings |
| `numpy` | 1.13.1 | Numerische Berechnungen, Array-Operationen |
| `nltk` | 3.2.4 | Tokenisierung (word_tokenize), Frequenzverteilung |
| `protobuf` | 3.4.0 | Serialisierung (TensorFlow-Abhängigkeit) |
| `bleach` | 1.5.0 | HTML-Sanitizing (TensorBoard-Abhängigkeit) |
| `html5lib` | 0.9999999 | HTML-Parsing (TensorBoard-Abhängigkeit) |
| `markdown` | 2.6.9 | Markdown-Rendering (TensorBoard-Abhängigkeit) |
| `werkzeug` | 0.12.2 | WSGI-Utilities (TensorBoard-Abhängigkeit) |
| `six` | 1.10.0 | Python 2/3-Kompatibilität |

### Nicht in `environment.yml`, aber im Code verwendet
- **`beautifulsoup4`** (`bs4`) — HTML-Parsing für den Web-Scraper (`scrape_latin_texts.py`)
- **`tqdm`** — Fortschrittsbalken im LSTM-Training (`LSTM_model.py`)

### Technologie-Mix
Es handelt sich um ein reines Python-Backend-Projekt. Es gibt kein separates Frontend. Die einzige nicht-Python-Komponente sind die C++-Dateien für die TensorFlow Custom Ops (word2vec), die zu einer `.so`-Shared-Library kompiliert werden müssen.

---

## 2.4 Architektur

### Verzeichnisstruktur

| Verzeichnis/Datei | Funktion |
|---|---|
| `/` (Wurzel) | Alle Python-Quelldateien liegen flach im Wurzelverzeichnis — keine Paketstruktur |
| `assets/` | Bilder und Diagramme für die README-Dokumentation |
| `checkpoints/` | Gespeicherte Modell-Checkpoints (TensorFlow Saver) |
| `grammar/` | Enthält `verbs.txt` — eine Liste lateinischer Verben (Zweck im Code nicht referenziert) |
| `library/` | Zielverzeichnis für gescrapte Texte (in `.gitignore`, muss generiert werden) |
| `small_library/` | Kleine Textsammlung für lokale Experimente (3 Dateien: Augustus, Caesar, Catullus) |
| `test_library/` | Testdaten für Unit-Tests (dieselben 3 Dateien) |
| `models/` | Vortrainiertes LSTM-Modell (ein Checkpoint) |
| `logs/` | TensorBoard-Logs (in `.gitignore`) |
| `docs/` | Dokumentation (neu erstellt) |

### Architekturmuster

Das Projekt folgt keinem etablierten Architekturmuster. Es ist ein **monolithisches Script-basiertes Projekt** mit folgender impliziter Schichtung:

1. **Datenbeschaffung**: `scrape_latin_texts.py` (Web-Scraping)
2. **Datenbereinigung**: `clean_data.py` (Text-Cleaning-Pipeline)
3. **Datenaufbereitung**: `text_handling.py` (Tokenisierung, Encoding, Splitting)
4. **Modelle**: `LSTM_model.py` (Char-RNN), `embedding.py` (Keras-Embedding), `word2vec.py` (TF-Tutorial)
5. **Orchestrierung**: `latin_learner.py` (CLI-Entry-Point)

Die Abhängigkeiten sind geradlinig und nicht zirkulär. Es gibt keine abstrakte Modell-Schnittstelle — jedes Modell hat seine eigene API.

---

## 2.5 Stilanalyse

### PEP 8-Konformität

Das Projekt hält PEP 8 **größtenteils ein**, mit folgenden Verstößen:

| Verstoß | Beispiel | Datei:Zeile |
|---|---|---|
| Vergleich mit `== True` statt `is True` oder direkter Bool-Auswertung | `if sample_only == True:` | `latin_learner.py:52` |
| Vergleich mit `== None` statt `is None` | `if checkpoint == None:` | `LSTM_model.py:241` |
| Leerzeichen um `=` in Keyword-Argumenten | `checkpoint = None` | `LSTM_model.py:189` |
| Inkonsistente Einrückung (Mischung aus Spaces) | Verschiedene Ausrichtungen bei mehrzeiligen Aufrufen | `LSTM_model.py:288-291` |
| Fehlende Leerzeile vor Funktionsdefinition | `def build_loss` direkt nach `return` | `LSTM_model.py:121` |
| `CamelCase` statt `snake_case` für Variablen | `FLAGS`, `HAPAX_INDEX`, `LATIN_BASE_URL` (als Konstanten akzeptabel), aber `TextData` Methoden verwenden korrektes snake_case | diverse |
| Bare `except` ohne spezifische Exception | `except:` | `text_handling.py:27` |
| Nicht verwendete Variable `filename` referenziert, bevor sie definiert ist | `print("ATTENTION: ... ", filename)` — `filename` ist im Scope nicht definiert | `scrape_latin_texts.py:35` |

### word2vec.py / word2vec_test.py
Diese Dateien stammen aus dem TensorFlow-Tutorial und verwenden Google-Python-Stil (2-Space-Einrückung, abweichende Namenskonventionen). Das ist akzeptabel, da es sich um Fremdcode handelt.

---

## 2.6 Pattern-Analyse

### Verwendete Patterns

| Pattern | Wo | Beschreibung |
|---|---|---|
| **Generator-Pattern** | `LSTM_model.py:get_batches()` | Yield-basierter Batch-Generator — sauber implementiert |
| **Builder-Pattern (funktional)** | `LSTM_model.py:build_inputs/build_lstm/build_output/build_loss/build_optimizer` | Funktionale TF-Graph-Konstruktion — typisch für TF 1.x |
| **God-Class** | `LSTM_model.py:CharRNN` | Vereint Graph-Konstruktion, Training, Evaluation und Sampling in einer Klasse |
| **Pipeline-Pattern** | `clean_data.py:clean_text()` | Sequentielle Anwendung von Transformationen — klar und verständlich |
| **Data-Class** | `text_handling.py:TextData` | Reine Datenverarbeitungsklasse ohne Zustand — könnte auch ein Modul mit Funktionen sein |
| **Scraper-Pattern** | `scrape_latin_texts.py:LatinScraper` | Rekursiver Web-Scraper mit Besuchsliste — solide Grundstruktur |

### Pattern-Verletzungen

| Verletzung | Wo | Vermutliche Ursache |
|---|---|---|
| **`@staticmethod` mit `self`-Parameter** | `embedding.py:59-69` | Vermutlich ein Verständnisfehler bei `@staticmethod` vs. regulärer Methode. Die Methode wird dann über `self._compute_similarity(self, ...)` aufgerufen (Zeile 76) — funktioniert, ist aber ein Anti-Pattern. |
| **Fehlender Methodenname** | `embedding.py:105` — `self.train()` sollte `self.optimize()` sein | Wahrscheinlich Tippfehler oder Umbenennung der Methode ohne Aktualisierung des Aufrufs. |
| **Fehlende Ressourcen-Verwaltung** | `LSTM_model.py:239,246` — TF-Session wird nicht geschlossen | Das `TODO`-Kommentar zeigt, dass es dem Entwickler bewusst war. Vermutlich wurde es aufgeschoben. |
| **Hardcoded Konfiguration** | `latin_learner.py:25-31` — Modellparameter als lokale Variablen | Das `TODO`-Kommentar (Zeile 24) bestätigt: geplante, aber nicht umgesetzte Verlagerung in die Modellklasse. |
| **Bare except** | `text_handling.py:27` | Wahrscheinlich schnelle Fehlerbehandlung während der Entwicklung — sollte spezifische Exceptions abfangen und den Fehler loggen. |
| **Undefinierte Variable** | `scrape_latin_texts.py:35` — `filename` im Scope nicht definiert | Vermutlich Copy-Paste-Fehler oder Refactoring-Überbleibsel. |
| **Mixed Concerns in `process_reference`** | `scrape_latin_texts.py:97-134` | Vermischt URL-Konstruktion, HTTP-Request, Parsing und Dateischreiben. Gewachsener Code ohne Refactoring. |
| **Auskommentierter interaktiver Code** | `scrape_latin_texts.py:89,123` — `input()` auskommentiert und durch `'y'` ersetzt | Deutet auf eine Entwicklungsphase hin, in der manuelles Durchgehen nötig war, dann aber automatisiert wurde — ohne Aufräumen. |
