# ⚖️ GDPR Assistant: an offline RAG chatbot with an ontology

Ask questions about the EU General Data Protection Regulation (GDPR) and get answers **with article citations**, running **100% on your own computer**. A hand-built **OWL ontology** (rights, obligations, deadlines, fines) adds structured knowledge on top of plain RAG, and the project includes an evaluation harness to compare the two.

> ⚠️ Educational project. This is **not legal advice**.

<img width="1593" height="773" alt="image" src="https://github.com/user-attachments/assets/fa1a2f33-f1c5-4ba2-8c7d-44bd5dd563d8" />


## Features

- **Fully offline:** local LLM (Ollama) and local vector database (ChromaDB). Internet is only needed once, to download the models.
- **Three modes, switchable in the app:**
  - **Plain RAG:** vector search + local LLM.
  - **Ontology + RAG (hybrid):** the ontology adds structured facts and the right articles, then the LLM answers.
  - **Ontology only (no LLM):** the answer is built directly from the ontology, plus the official article text.
- **Cited answers:** every answer points to the source article, for example `[Art. 33]`.
- **Refuses out-of-scope questions:** answers "does not cover this" instead of making things up.
- **Ontology you can open in Protégé:** `ontology/gdpr.owl` (RDF/XML), read by the app with no extra library.
- **Evaluation harness:** 24 test questions with expected articles and key facts, plus a script that compares the three modes.

## How it works

![How it works](how-it-works.png)

1. **Index (once):** `ingest.py` splits the regulation per article (99 articles + recitals → 328 chunks), embeds each chunk, and stores it in ChromaDB.
2. **Ontology:** `ontology.py` reads `gdpr.owl`, applies two small inference steps (subclass closure, inverse properties), recognises concepts in the question through `rdfs:label` synonyms, and walks relations such as `triggeredBy`, `hasObligation`, `hasPenaltyTier` and `definedIn`.
3. **Answer:** in hybrid mode the ontology facts and the articles they point to are added to the vector results, and a local LLM answers using only that evidence, at temperature 0, with citations.

Why the ontology helps: "List all the rights of a data subject" needs several articles at once (15-18, 20-22, 82). Top-k vector search can miss some; the ontology lists all of them.

## The ontology

14 classes, 7 object properties, 6 data properties and 61 individuals, covering about 24 key articles.

| Part | Content |
|---|---|
| **Classes** | Actor (DataSubject, Controller, Processor, DataProtectionOfficer, SupervisoryAuthority), Right, Obligation, Principle, LegalBasis, Rule, Penalty, Event, Article |
| **Object properties** | `hasRight` (inverse `rightOf`), `hasObligation`, `definedIn`, `triggeredBy`, `notifies`, `hasPenaltyTier` |
| **Data properties** | `articleNumber`, `deadlineHours`, `deadlineMonths`, `maxFineEUR`, `maxFinePercent`, `description` |
| **Examples** | `RightToErasure definedIn Art_17` · `BreachNotificationToAuthority deadlineHours 72, triggeredBy PersonalDataBreach` · `FineTier2 maxFineEUR 20000000, maxFinePercent 4` |

Facts in the ontology were checked against the official text. Penalty tiers follow Art. 83(4) and 83(5).

### Explore it in Protégé
Open `ontology/gdpr.owl` in [Protégé](https://protege.stanford.edu), start the HermiT reasoner, and try these in the **DL Query** tab (tick "Instances"):

| Query | Returns |
|---|---|
| `Right` | all 8 rights |
| `Right and definedIn value Art_17` | RightToErasure |
| `Obligation and triggeredBy value PersonalDataBreach` | the two breach notifications |
| `deadlineHours value 72` | BreachNotificationToAuthority |
| `hasPenaltyTier value FineTier1` | 6 items (Art. 83(4) tier) |
| `hasPenaltyTier value FineTier2` | 18 items (Art. 83(5) tier) |
| `definedIn value Art_6` | the 6 legal bases |
| `inverse hasObligation value TheController` | the controller's 5 obligations |
| `rightOf value AnyDataSubject` | all 8 rights (via the inverse property) |

When you edit the ontology, save it as **RDF/XML** and keep `description` and `rdfs:label` on new individuals, because the app uses them to recognise concepts.

## Tech stack

Python · [Ollama](https://ollama.com) (`llama3.2:3b`, `nomic-embed-text`) · [ChromaDB](https://www.trychroma.com) · [Streamlit](https://streamlit.io) · OWL / RDF-XML (Protégé) · pypdf

## Quick start

**Requirements:** Python 3.10+, [Ollama](https://ollama.com), about 8 GB RAM. No GPU needed.

```bash
# 1. Get the code
git clone https://github.com/Nishaahsan/gdpr-offline-rag.git
cd gdpr-offline-rag

# 2. Download the local models (once, needs internet)
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 3. Python environment
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

# 4. Add the GDPR text (see below), then build the index
mkdir docs
python ingest.py

# 5. Check the ontology on its own (no Ollama needed)
python ontology.py "List all the rights of a data subject"

# 6. Run the chatbot
streamlit run app.py
```

### Getting the GDPR text

The regulation text is **not included** in this repository. Download it from the official source:

1. Open <https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng>
2. Select all text on the page, copy it, and paste it into `docs/gdpr.txt`. Save the file.
3. Each article must start on its own line, such as `Article 17`, followed by its title on the next line.

`ingest.py` should print `99 articles found`.

## Project structure

```
.
├── app.py                    # Streamlit chatbot (3 modes)
├── rag_core.py               # shared pipeline: plain, hybrid, ontology-only
├── ontology.py               # OWL reader, concept matching, light reasoning
├── chunker.py                # article-aware chunking
├── ingest.py                 # builds the ChromaDB index from docs/
├── requirements.txt
├── ontology/
│   └── gdpr.owl              # the GDPR ontology (open in Protégé)
├── evaluation/
│   ├── eval_questions.json   # 24 questions with expected articles and key facts
│   └── run_eval.py           # compares plain / hybrid / ontology-only
├── .streamlit/config.toml    # UI theme
├── assets/screenshot.png
└── docs/                     # put gdpr.txt here (not tracked)
```

## Evaluation

`evaluation/eval_questions.json` has 24 questions: 19 direct, 2 multi-article ("list all rights of a data subject"), and 3 out-of-scope traps. Expected articles and key facts were checked against the official text.

```bash
python evaluation/run_eval.py --config plain
python evaluation/run_eval.py --config hybrid
python evaluation/run_eval.py --config ontology
python evaluation/run_eval.py --compare      # prints a Markdown table
```

| Metric | Meaning |
|---|---|
| hit@k / recall@k / MRR | the right articles were retrieved |
| recall (list questions) | all articles of a multi-article question were found |
| fact coverage | key facts such as "72 hours" appear in the answer |
| citation OK | the answer cites an expected article |
| refused out-of-scope | the bot says it cannot answer |

> **Note on fairness:** the ontology was extended while looking at these 24 questions, so scores on this set would be optimistic, especially for "Ontology only". For an honest comparison, write held-out questions after the ontology is finished.

## Limitations

- The ontology is hand-built and covers about 24 articles. Questions outside it fall back to vector search (hybrid) or are refused (ontology only).
- Concept matching is keyword and synonym based, not semantic. Unusual wording can be missed.
- A 3B model is small. It can miss details or paraphrase imperfectly, so always check the cited article.
- Only the GDPR text is indexed. National laws, court decisions and guidance are not covered.

## Roadmap

- [x] GDPR ontology (OWL): actors, rights, obligations, rules, penalty tiers
- [x] Ontology + RAG hybrid mode and ontology-only mode
- [x] Evaluation harness for three configurations
- [ ] Held-out test questions and published comparison results
- [ ] More articles in the ontology (Chapters IV-VII)
- [ ] Semantic (embedding-based) concept matching

## Troubleshooting

- **`SKIPPED ... only N characters`:** the file in `docs/` is empty. Paste the text and save it.
- **"articles missing" warning:** `Article 17` must be on its own line. Split it if it is joined with the title.
- **`Index not found`:** run `python ingest.py` first.
- **Ollama connection error:** make sure Ollama is running (`ollama list`).
- **`pip install` times out:** use `pip install -r requirements.txt --default-timeout=200 --retries 10`.
- **Ontology not found:** `ontology/gdpr.owl` must exist. Without it only Plain RAG is available.

## Licenses and attribution

- Code: MIT, see [LICENSE](LICENSE).
- GDPR text: Regulation (EU) 2016/679, from [EUR-Lex](https://eur-lex.europa.eu). Check the EUR-Lex legal notice before redistributing it.
- Models are downloaded separately through Ollama and keep their own licenses (Llama 3.2 Community License, nomic-embed-text).
