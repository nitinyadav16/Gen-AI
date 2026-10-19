# OSA Root-Cause Copilot

An AI copilot for consumer-health retail execution managers. It flags at-risk store-SKUs, ranks them by estimated lost sales, explains the likely root cause of each on-shelf availability (OSA) decline with cited evidence, and recommends the next action and owner from an approved playbook. A manager approves every item, and the tool never sends anything.

Built from `OSA_Copilot_AI_PRD.docx` (v1.0, 19 Oct 2026). The MVP runs on seeded synthetic data. Python · LangChain · Groq (`gpt-oss-120b` / `gpt-oss-20b`) · Streamlit.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env          # then add your GROQ_API_KEY
.venv\Scripts\streamlit run app.py
```

The synthetic data is already in `data/`. To rebuild it (deterministic, fixed seed):

```bash
.venv\Scripts\python -m osa.datagen
```

Without a key, the worklist, lost-sales ranking, evidence packs and data-issue checks still work; diagnosis shows as "unavailable" (PRD 9.2).

## How it works (PRD 9.3)

| Step | What happens | Where | LLM? |
|---|---|---|---|
| 1-2 | Load feeds, scope to the manager's retailer, drop excluded fields, check freshness and completeness | `osa/data.py` | no |
| 3-4 | Flag OSA < target or drop > X pts vs the 4-week baseline; lost sales = baseline units x price x weeks x (1 - OSA); rank | `osa/detect.py` | no |
| 5 | Evidence pack: numbered signals, one rule check per cause, conflict checks | `osa/evidence.py` | no |
| 6 | Model ranks 1-3 causes from the fixed cause list as JSON, citing signals | `osa/diagnose.py` | yes |
| 7 | Validator: schema, cause names, every cited value, every number in the prose; one retry | `osa/diagnose.py` | no |
| 8 | Confidence gate: low confidence, conflicts, no matching pattern, or an unsupported top cause gives "Needs investigation" | `osa/diagnose.py` | no |
| 9 | Playbook action, owner and turnaround with the entry ID | `osa/playbook.py`, `data/playbook.json` | no |
| 10-12 | Accept / reject / edit (reason required); escalation or brief drafted; manager approves and sends; everything logged | `osa/copilot.py`, `logs/` | no |
| FR-10 | Plain-language Q&A with read-only metric tools, filters shown, out-of-scope declined | `osa/qa.py` | yes |

**Cause list** (`osa/causes.py`): DC shortage, order cuts (both upstream, escalated to the Supply Chain Planner), store ordering miss, phantom inventory, shelf execution gap, delisting, promo demand spike.

**Guardrails (PRD 10):**
- All numbers are computed in code. The validator blocks any number the model writes that is not in the evidence pack.
- Retailer scoping is applied as each feed is read.
- Rep names, store-manager names, free-text notes and margins are dropped on load. The test data includes a prompt-injection note in the audit feed to prove it never reaches the model.
- The tool has no send or write-back capability.

## Data (`data/`)

Synthetic feeds at store-SKU-week grain: 2 retailers (50 + 12 stores), 20 SKUs, 12 weeks ending 2026-10-18.

| File | Contents |
|---|---|
| `osa.csv` | OSA |
| `pos.csv` | POS units and price |
| `store_inventory.csv` | Store on-hand |
| `shipments.csv` | Store orders and shipments |
| `dc.csv` | DC stock, plus orders and receipts from the manufacturer |
| `audits.csv` | Execution audits |
| `promo.csv` | Promo calendar |
| `stores.csv`, `skus.csv` | Master data |
| `feed_manifest.json` | Refresh dates and data owners |
| `playbook.json` | The action playbook |

`eval_data/scenarios.json` is the answer key for 60 seeded scenarios: 7 per cause, 6 ambiguous and 5 data gaps. The copilot never reads it.

## Tests and evals

```bash
.venv\Scripts\python -m unittest discover -s tests      # offline, fake models, ~2 min
.venv\Scripts\python evals.py                            # PRD section 11 against the live model
```

`evals.py` writes `eval_results/report.md` and `results.json`. It measures top-1 and top-3 cause accuracy, evidence grounding (cited values re-computed straight from the CSVs), numeric fidelity, abstention, playbook adherence, scope and safety (20 adversarial prompts plus a cross-retailer leak check), consistency (5 runs of the same input) and p95 latency. On Groq's free tier (8k tokens/min), the run paces itself and takes about 25 minutes.

## Configuration

`osa/config.py` holds the thresholds (target 92%, drop 8 pts, 4-week baseline, 6 weeks of minimum history, 7 days to stale), with per-retailer overrides in `RETAILER_RULES`, plus the models and prompt versions. Logs go to `logs/`: `decisions.jsonl` (FR-9 audit), `diagnoses.jsonl`, `requests.jsonl` and a diagnosis cache.

## Not built yet

- **Vercel deployment.** The PRD names Vercel for the prototype, but Streamlit needs a long-running server. Use Streamlit Community Cloud, or add a thin FastAPI layer over `osa.copilot.Copilot`.
- **Real sign-in.** The retailer scope is a sidebar picker that stands in for account-based scoping.
- **Tracking OSA after an action.** The decision log records what is needed, but there is no recovery dashboard yet (PRD O11, deferred).
