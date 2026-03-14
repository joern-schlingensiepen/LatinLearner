# LatinLearner — Projektstatus

**Status:** Aktiv — Modernisierung geplant
**Datum der Entscheidung:** 2026-03-14

---

## Zusammenfassung der Analyse

Das Projekt ist ein Deep-Learning-Projekt für die lateinische Sprache mit dem Ziel, einen lateinischen Chatbot zu bauen und das Verständnis der Sprache durch maschinelles Lernen zu verbessern. Es umfasst:

- **Web-Scraper** für lateinische Texte (thelatinlibrary.com)
- **Datenbereinigungspipeline** für die gesammelten Texte
- **Char-RNN (LSTM)** zur Textgenerierung auf Zeichenebene
- **Word-Embeddings** (TF-Tutorial-basiert + Keras-Variante)
- **Einfacher Chatbot** auf Basis des Char-RNN

### Technischer Zustand

- **Technologie-Stack veraltet:** TensorFlow 1.3, Python 3.6 — auf modernen Systemen nicht lauffähig
- **Codeumfang gering:** ~500 Zeilen eigenständiger Code + TF-Tutorial-Code (word2vec)
- **Bugs vorhanden:** `embedding.py` nicht lauffähig (falscher Methodenaufruf), undefinierte Variable in `scrape_latin_texts.py`
- **Keine Paketstruktur:** Alle Dateien flach im Wurzelverzeichnis
- **Vortrainiertes Modell:** TF 1.x Checkpoint vorhanden, aber nicht portierbar — Neutraining wird bevorzugt

### Wertvolle, framework-unabhängige Bestandteile

- Gesammelte und bereinigte lateinische Trainingsdaten (~24 MB)
- Scraping-Pipeline für thelatinlibrary.com
- Validierungsfragen (`latin_questions.txt`)
- Domänenwissen aus den Experimenten (dokumentiert in README)

Die vollständige Analyse befindet sich in `docs/ANALYSIS.md`.

---

## Offene TODOs

- [ ] **Johannes Staub anrufen** — Abstimmung, wie das Projekt gemeinsam weiterentwickelt werden soll

## Geplante nächste Schritte

1. **Entscheidung Ziel-Framework:** Festlegen, ob Modernisierung auf PyTorch, TensorFlow 2.x oder ein anderes Framework erfolgt
2. **Entscheidung Modellarchitektur:** LSTM beibehalten oder auf Transformer-Architektur wechseln
3. ~~**Test-Framework klären:**~~ Entschieden: **pytest**
4. **Portierung durchführen:**
   - Python-Version und Dependencies aktualisieren
   - Scraper und Datenbereinigung modernisieren (framework-unabhängig, geringer Aufwand)
   - Modellcode auf Ziel-Framework portieren
   - Neutraining mit vorhandenen Daten
5. **`docs/Test-Setup.md` erstellen** nach Klärung des Test-Frameworks
6. **Paketstruktur einführen** (saubere Verzeichnisstruktur statt flacher Ablage)
