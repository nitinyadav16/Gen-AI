"""Live end-to-end check against Groq: grounding, escalation, latency."""
from __future__ import annotations

import sys, time
from chain import RagPipeline

CASES = [
    ("in-scope / data",      "Why is my mobile internet so slow?"),
    ("in-scope / roaming",   "How do I activate roaming before I travel abroad?"),
    ("in-scope / SIM",       "My new SIM is not being recognised by my phone."),
    ("account-specific",     "What is my current account balance right now?"),
    ("out-of-knowledge",     "Does NovaCell offer home mortgage loans and what is the interest rate?"),
]

def main() -> int:
    pipeline = RagPipeline()
    for label, question in CASES:
        t0 = time.perf_counter()
        docs = pipeline.retrieve(question)
        t_retrieved = time.perf_counter() - t0

        first_token = None
        parts = []
        for token in pipeline.stream(question, docs):
            if first_token is None and token.strip():
                first_token = time.perf_counter() - t0
            parts.append(token)
        total = time.perf_counter() - t0
        answer = "".join(parts).strip()

        print("=" * 78)
        print(f"[{label}] {question}")
        print(f"  retrieval {t_retrieved:.2f}s | first token {first_token:.2f}s | total {total:.2f}s")
        print(f"  sources: {', '.join(d.citation for d in docs)}")
        print("-" * 78)
        print(answer)
        print()
    return 0

if __name__ == "__main__":
    sys.exit(main())
