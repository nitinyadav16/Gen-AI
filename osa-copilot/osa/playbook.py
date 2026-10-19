"""Playbook lookup (FR-8) and draft briefs / escalations (FR-10 stretch, section 9.3 step 11).

Actions come verbatim from the product-owned playbook and always carry the
entry ID, so playbook adherence is structural rather than something the model
has to get right. Drafts are filled from templates with computed numbers
only; the tool has no send capability, the manager copies and sends them.
"""

from __future__ import annotations

from osa import causes as C
from osa.detect import FlaggedItem
from osa.evidence import EvidencePack, fmt


def lookup(playbook: dict, cause: str) -> dict | None:
    return next((e for e in playbook["entries"] if e["cause"] == cause), None)


def _evidence_lines(pack: EvidencePack, ids: list[str]) -> list[str]:
    return [f"  - {pack.signal(i).text()}" for i in ids if pack.signal(i)]


def draft_message(pack: EvidencePack, cause: str, playbook: dict, worklist: list[FlaggedItem],
                  cited_ids: list[str] | None = None) -> dict:
    """Escalation to the Supply Chain Planner for upstream causes, otherwise a field brief."""
    it = pack.item
    entry = lookup(playbook, cause)
    check = pack.check(cause)
    ids = list(dict.fromkeys((cited_ids or []) + (check.signal_ids if check else [])))
    header = (f"{it.retailer} | {it.region} | Store {it.store_id} | {it.sku} {it.sku_name}\n"
              f"OSA {fmt(it.osa_latest, '%')} (target {fmt(it.osa_target, '%')}, 4-week baseline "
              f"{fmt(it.osa_baseline, '%')}), {it.weeks_affected} week(s) affected. "
              f"Estimated lost sales ${it.lost_sales:,.2f} (estimate).")
    if cause in C.UPSTREAM_CAUSES:
        stores = sorted({w.store_id for w in worklist if w.sku == it.sku and w.dc_id == it.dc_id})
        dc_lines = _evidence_lines(pack, ["S9", "S10", "S8", "S7"])
        body = (f"Subject: [OSA escalation] {C.label(cause)} - {it.sku} {it.sku_name} at {it.dc_id}\n\n"
                f"Hi Supply Chain team,\n\n"
                f"The OSA copilot diagnosed {C.label(cause).lower()} for this SKU and I have reviewed it.\n\n"
                f"{header}\n\nDC and shipment evidence:\n" + "\n".join(dc_lines) + "\n\n"
                f"Flagged stores served by {it.dc_id} for this SKU: {', '.join(stores)}\n\n"
                f"Requested action ({entry['entry_id']}): {entry['action']}\n"
                f"Target turnaround: {entry['turnaround']}\n\nThanks")
        return {"kind": "escalation", "to": entry["owner"], "entry_id": entry["entry_id"], "text": body}
    body = (f"Subject: [OSA action] {C.label(cause)} - Store {it.store_id}, {it.sku} {it.sku_name}\n\n"
            f"Hi,\n\n{header}\n\nWhat the data shows:\n" + "\n".join(_evidence_lines(pack, ids)) + "\n\n"
            f"Action ({entry['entry_id']}): {entry['action']}\n"
            f"Owner: {entry['owner']}\nTarget turnaround: {entry['turnaround']}\n\nThanks")
    return {"kind": "brief", "to": entry["owner"], "entry_id": entry["entry_id"], "text": body}


def draft_analyst_note(pack: EvidencePack, reason: str) -> str:
    it = pack.item
    lines = "\n".join(f"  - {s.text()}" for s in pack.signals)
    checks = "\n".join(f"  - {C.label(c.cause)}: {c.result}" for c in pack.checks)
    return (f"Subject: [OSA needs investigation] Store {it.store_id}, {it.sku} {it.sku_name}\n\n"
            f"The copilot could not name a cause with confidence: {reason}\n\n"
            f"Signals checked:\n{lines}\n\nRule checks:\n{checks}\n\n"
            f"Conflicts: {'; '.join(pack.conflicts) or 'none'}\n\nNo action has been recommended.")


def draft_data_owner_note(retailer: str, issues) -> str:
    rows = "\n".join(f"  - {r.feed}: store {r.store_id}, {r.sku} ({r.issue})" for r in issues.itertuples())
    owners = ", ".join(sorted(set(issues["owner"])))
    return (f"Subject: [OSA data issue] {retailer} - missing or stale feed data\n\nHi {owners},\n\n"
            f"These items are excluded from this week's OSA worklist until the data arrives:\n{rows}\n\n"
            f"They have not been treated as out-of-stock. Please confirm when the feed is backfilled.\n\nThanks")
