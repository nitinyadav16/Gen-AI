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
| `llm.py` | Shared Groq model factory, reused by every route |
| `conversation.py` | Session history window shared by both routes |
| `router.py` | Intent routing: RAG pipeline or Refund Agent |
| `intent.py` | Classifies `requirement_inquiry` vs `refund_request` |
| `accounts.py` | Hard-coded demo account and recharges |
| `refund_tools.py` | The four refund tools + deterministic validation rules |
| `refund_agent.py` | Tool-using agent inside a deterministic state machine |
| `app.py` | Streamlit interface (behaviour) |
| `ui.py` | Interface styling: design tokens, CSS, and HTML render helpers |
| `main.py` | CLI REPL |
| `interaction_log.py` | Appends answers and 👍/👎 votes to `logs/interactions.jsonl` |
| `smoke_test.py` | Offline check of ingestion + retrieval (no API calls, no cost) |
| `live_test.py` | Live check against Groq: grounding, escalation, latency |
| `test_refund.py` | Offline tests: the 7 refund scenarios and every guardrail |
| `live_agent_test.py` | Live check of intent routing and the agent tool loop |
| `.streamlit/config.toml` | Disables Streamlit's source watcher (see below) |

### Adding a knowledge source

1. Write `ingest_<source>.py` that upserts documents into a new collection.
2. Append a `CollectionSpec` to `COLLECTIONS` in `retriever.py`.

The retriever, prompt context, Sources panel, and status banner pick it up
automatically.

## Agentic refund capability

The assistant classifies every message, then routes it:

```
                        message
                           |
                    Intent detection
              /                             requirement_inquiry              refund_request
           |                              |
    existing RAG pipeline           Refund Agent
                                          |
                          get_user_account / validate_refund
                                          |
                                   eligible?  -- no --> rejected, reason given
                                          |
                                   amount <= Rs 499 ?
                                    /                                              yes             no
                                   |               |
                          confirm / cancel   submitted to Support Team
                                   |
                            process_refund
```

### What the model decides, and what it cannot

The agent genuinely calls tools - it chooses when to read the account and which
amount to validate. But every decision that matters is application code:

| Decided by the model | Decided by code |
|---|---|
| Which tool to call | Whether the refund is eligible |
| Which amount to propose | The Rs 499 approval limit |
| Wording of the intent label | Whether a reply counts as confirmation |
| | Whether a refund executes at all |

`get_user_account` and `validate_refund` are bound to the model.
`process_refund` and `submit_to_support` **are never exposed to it** - they are
reachable only through the state machine, after validation and an explicit yes.

The guardrails that follow from that:

* A refund request is not itself a confirmation - the user must say yes.
* An ambiguous reply asks again rather than executing.
* An amount the customer never typed is never validated, so the agent cannot
  invent one.
* Amounts above Rs 499 go to the Support Team, and the agent never claims what
  the Support Team decided.
* What the customer asserts about their recharges never overrides the account.

### Demo account

One hard-coded user (`user_123`), treated as if it arrived from an
authenticated session - the customer is never asked for an id. It has two
recharges: **Rs 499** (refundable here) and **Rs 999** (above the limit, goes to
Support). The sidebar shows both and their live refund status; **Reset demo
recharges** restores them.

### Conversation context

The last few exchanges are passed to both routes, so "should I do that before I
travel?" resolves against the previous answer. Retrieval is deliberately left
alone: the search query is always the question exactly as asked.

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
python test_refund.py      # offline: 29 tests, refund scenarios + guardrails
python smoke_test.py       # offline: collection counts, 9-doc retrieval, context labels
python live_test.py        # live: grounded answers + latency for 5 cases
python live_agent_test.py  # live: intent routing and the agent tool loop
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
