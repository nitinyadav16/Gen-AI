# OSA Copilot eval report

Diagnosis model `openai/gpt-oss-120b`, Q&A model `openai/gpt-oss-20b`, prompt `diag-v1.0`, 60 scenarios.

| Eval dimension | Result | Threshold | Pass |
|---|---|---|---|
| Root-cause accuracy, top-1 | 1.0 | >= 0.75 | PASS |
| Root-cause accuracy, top-3 | 1.0 | >= 0.9 | PASS |
| Evidence grounding | 1.0 | >= 1.0 | PASS |
| Numeric fidelity (errors) | 0 | <= 0 | PASS |
| Abstention | 1.0 | >= 0.9 | PASS |
| Playbook adherence | 1.0 | >= 1.0 | PASS |
| Scope and safety: declined | 1.0 | >= 0.95 | PASS |
| Cross-retailer data shown | 0 | <= 0 | PASS |
| Consistency (same top cause) | 1.0 | >= 0.9 | PASS |
| Diagnosis latency p95 (s) | 2.71 | <= 10.0 | PASS |
| Q&A latency p95 (s) | 1.79 | <= 8.0 | PASS |

Cited signals checked: 165. In-scope questions answered: 1.0. Needs-investigation rate on cause scenarios: 0.0. Retries used: 0.

## Misses

