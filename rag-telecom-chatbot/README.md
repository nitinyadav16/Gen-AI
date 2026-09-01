# NovaCell RAG Telecom Customer-Care Chatbot

A Retrieval-Augmented Generation assistant that answers Tier-1 telecom support
questions using **only** NovaCell's own knowledge: a published FAQ, resolved
support tickets, and the official user guide. Built to the spec in `PRD.md`.

```
question ─▶ merged retriever ─┬─ chroma:faq      top-3
                              ├─ chroma:tickets  top-3
                              └─ chroma:guides   top-3
                                      │  9 source-labelled documents
                                      ▼
                    ChatPromptTemplate (grounding rules)
                                      ▼
                qwen/qwen3.6-27b on Groq (temperature=0)
                                      ▼
                     StrOutputParser ─▶ streamed answer + Sources
```

* **Embeddings:** `all-MiniLM-L6-v2`, run locally — no embedding API cost.
* **Vector store:** ChromaDB persisted to `chroma_store/` — no re-ingest on start.
* **LLM:** Groq-hosted Qwen, `temperature=0`, `reasoning_effort="none"`.
* **UI:** Streamlit (streaming, sample questions, sources, 👍/👎) + a CLI REPL.

---

## Setup

Everything installs into the project-local `.venv`. Never install into a global
interpreter.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

Add your Groq API key (free tier at <https://console.groq.com/keys>):

```bash
copy .env.example .env          # cp .env.example .env on macOS/Linux
```

Then edit `.env` and set `GROQ_API_KEY=...`.

## Ingest the knowledge base (once)

```bash
python ingest_all.py
```

Or one source at a time — each script is safe to re-run:

| Script | Source | Documents |
|---|---|---|
| `python ingest_faq.py` | `data/faq.csv` | 1 per row, upserted on `FAQ-<id>` |
| `python ingest_tickets.py` | `data/tickets.db` | 1 per resolved ticket, upserted on `ticket_id` |
| `python ingest_guide.py` | `data/telecom_guide.pdf` | 600-char chunks, 100-char overlap (collection rebuilt) |

Support ops can edit `faq.csv` or seed new tickets and re-run the matching
script — no redeployment, no code change.

## Run

Browser UI:

```bash
streamlit run app.py
```

CLI REPL (`quit` to exit, `samples` to list example questions):

```bash
python main.py
```

---

## Project layout

| File | Role |
|---|---|
| `config.py` | Paths, model names, retrieval constants, `.env` loading |
| `embeddings.py` | Cached local `all-MiniLM-L6-v2` embedder |
| `vectorstore.py` | Chroma open / count / upsert / reset helpers |
| `ingest_faq.py`, `ingest_tickets.py`, `ingest_guide.py`, `ingest_all.py` | Knowledge ingestion |
| `retriever.py` | Collection registry + parallel merged retriever + context formatting |
| `chain.py` | Grounding prompt, Groq LLM, LCEL chain, `RagPipeline` |
| `app.py` | Streamlit interface (behaviour) |
| `ui.py` | Interface styling: design tokens, CSS, and HTML render helpers |
| `main.py` | CLI REPL |
| `interaction_log.py` | Appends answers and 👍/👎 votes to `logs/interactions.jsonl` |
| `smoke_test.py` | Offline check of ingestion + retrieval (no API calls, no cost) |
| `live_test.py` | Live check against Groq: grounding, escalation, latency |
| `.streamlit/config.toml` | Disables Streamlit's source watcher (see below) |

### Adding a knowledge source

1. Write `ingest_<source>.py` that upserts documents into a new collection.
2. Append a `CollectionSpec` to `COLLECTIONS` in `retriever.py`.

The retriever, prompt context, Sources panel, and status banner pick it up
automatically.

## Grounding behaviour

The system prompt forbids answering from model-internal knowledge. When the
retrieved context is insufficient, the assistant says so and points the customer
to **611 or the MyTelecom app**. It has no access to live billing or CRM data,
so account-specific questions ("my balance") are answered with the documented
self-serve route rather than a fabricated figure.

Every answer ships with an expandable **Sources** panel listing each retrieved
document as `FAQ 16`, `TICKET TK-006`, `GUIDE p.8 #31`.

## Tests

```bash
python smoke_test.py    # offline: collection counts, 9-doc retrieval, context labels
python live_test.py     # live: calls Groq, prints answers + latency for 5 cases
```

`live_test.py` covers three in-scope questions plus two boundary cases — an
account-specific question ("what is my balance") and a question with no answer in
the knowledge base — to confirm the bot declines instead of inventing an answer.

## Note on `.streamlit/config.toml`

Streamlit's source watcher walks the `__path__` of every imported module. Because
`transformers` resolves its submodules lazily on attribute access, that walk
imports optional vision models and fails on a missing `torchvision` — hundreds of
tracebacks and a stalled first response. The config sets
`server.fileWatcherType = "none"`, which the model files never needed anyway.

**Trade-off:** editing a `.py` file no longer auto-reloads the app. Restart
`streamlit run app.py` after code changes.
