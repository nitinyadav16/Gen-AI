"""The OSA Root-Cause Copilot: one manager session, scoped to one retailer.

Ties the section 9.3 decision flow together:

    load + check feeds -> flag -> rank by lost sales -> evidence pack
    -> LLM diagnosis -> validator -> confidence gate -> playbook
    -> manager accept / reject / edit -> draft (manager sends) -> audit log

The copilot never sends or writes to any source system. "Approve" only
records the decision and releases a draft for the manager to send.
"""

from __future__ import annotations

from pathlib import Path

from osa import audit
from osa import causes as C
from osa import config
from osa.data import Dataset, available_retailers, load_dataset
from osa.detect import FlaggedItem, build_worklist
from osa.diagnose import CONFIDENT, NEEDS_INVESTIGATION, Diagnosis, DiagnosisCache, diagnose
from osa.evidence import EvidencePack, build_pack
from osa.llm import GroqJsonLLM, JsonLLM, chat_model
from osa.playbook import draft_analyst_note, draft_data_owner_note, draft_message, lookup
from osa.qa import QAAnswer, answer

ACCEPT, REJECT, EDIT, ROUTE = "accepted", "rejected", "edited", "routed_to_analyst"
REASON_CODES = [
    "Wrong cause",
    "Evidence insufficient",
    "Known local issue (e.g. store refit, recall)",
    "Data looks wrong",
    "Cause correct, action not suitable",
    "Other",
]


class Copilot:
    def __init__(self, retailer: str, user: str = "manager", data_dir: Path = config.DATA_DIR,
                 llm: JsonLLM | None = None, qa_model=None, log_dir: Path = config.LOG_DIR,
                 use_cache: bool = True):
        self.retailer = retailer
        self.user = user
        self.data_dir = data_dir
        self.ds: Dataset = load_dataset(retailer, data_dir)
        self.worklist: list[FlaggedItem] = build_worklist(self.ds)
        self._by_key = {i.key: i for i in self.worklist}
        self._packs: dict[str, EvidencePack] = {}
        self.diagnoses: dict[str, Diagnosis] = {}
        self._llm = llm
        self._qa_model = qa_model
        self.log_dir = log_dir
        self.cache = DiagnosisCache(log_dir / config.DIAGNOSIS_CACHE.name) if use_cache else None

    # --- models (built lazily so the rules-only worklist works without a key) ---
    @property
    def llm(self) -> JsonLLM:
        if self._llm is None:
            self._llm = GroqJsonLLM()
        return self._llm

    @property
    def qa_model(self):
        if self._qa_model is None:
            self._qa_model = chat_model(config.QA_MODEL)
        return self._qa_model

    # --- worklist and evidence -----------------------------------------------------
    def item(self, key: str) -> FlaggedItem:
        if key not in self._by_key:
            raise KeyError(f"{key} is not on the {self.retailer} worklist")
        return self._by_key[key]

    def evidence(self, key: str) -> EvidencePack:
        if key not in self._packs:
            self._packs[key] = build_pack(self.ds, self.item(key))
        return self._packs[key]

    # --- diagnosis ---------------------------------------------------------------
    def diagnose(self, key: str, use_cache: bool = True) -> Diagnosis:
        pack = self.evidence(key)
        try:
            llm = self.llm
        except Exception as e:      # no key configured: same fallback as an outage
            d = Diagnosis(key, "UNAVAILABLE", reason=f"model unavailable: {e}")
        else:
            d = diagnose(pack, self.ds.playbook, llm, self.cache if use_cache else None)
        self.diagnoses[key] = d
        if not d.cached:
            audit.append(self.log_dir / config.DIAGNOSIS_LOG.name, {
                "event": "diagnosis", "user": self.user, "retailer": self.retailer, "item_key": key,
                **d.to_dict(), "evidence": pack.for_model()})
        return d

    def diagnose_top(self, n: int = 10) -> list[Diagnosis]:
        return [self.diagnose(i.key) for i in self.worklist[:n]]

    # --- manager decisions (FR-9) -------------------------------------------------
    def decide(self, key: str, decision: str, reason_code: str = "", note: str = "",
               edited_cause: str | None = None) -> dict:
        d = self.diagnoses.get(key)
        if d is None:
            raise ValueError("diagnose the item before recording a decision")
        if decision not in (ACCEPT, REJECT, EDIT, ROUTE):
            raise ValueError(f"unknown decision {decision!r}")
        if decision in (REJECT, EDIT) and not reason_code:
            raise ValueError("a reason is required to reject or edit a diagnosis")
        if decision == ACCEPT and d.status != CONFIDENT:
            raise ValueError("only a confident diagnosis can be accepted; edit it or route it to an analyst")
        if decision == EDIT and edited_cause not in C.CAUSES:
            raise ValueError("pick the corrected cause from the cause list")

        final_cause = edited_cause if decision == EDIT else (d.top_cause if decision == ACCEPT else None)
        entry = lookup(self.ds.playbook, final_cause) if final_cause else None
        pack = self.evidence(key)
        draft = None
        if final_cause:
            cited = [e["signal_id"] for c in d.causes if c["cause"] == final_cause for e in c["evidence"]]
            draft = draft_message(pack, final_cause, self.ds.playbook, self.worklist, cited)
        elif decision == ROUTE or (decision == REJECT and d.status == NEEDS_INVESTIGATION):
            draft = {"kind": "analyst_note", "to": "Sales analyst", "entry_id": None,
                     "text": draft_analyst_note(pack, d.reason or reason_code)}

        record = audit.append(self.log_dir / config.DECISION_LOG.name, {
            "event": "decision", "user": self.user, "retailer": self.retailer, "item_key": key,
            "diagnosis_id": d.diagnosis_id, "diagnosis_status": d.status, "model_top_cause": d.top_cause,
            "decision": decision, "final_cause": final_cause, "reason_code": reason_code, "note": note,
            "playbook_entry": entry["entry_id"] if entry else None,
            "est_lost_sales": self.item(key).lost_sales,
            "cited_evidence": [{"cause": c["cause"], "signals": c["evidence"]} for c in d.causes],
            "prompt_version": d.prompt_version, "model_version": d.model,
        })
        return {"record": record, "draft": draft}

    def approve_draft(self, key: str, draft: dict, edited_text: str) -> dict:
        """The manager approved the draft. The copilot logs it; the manager sends it themselves."""
        return audit.append(self.log_dir / config.DECISION_LOG.name, {
            "event": "draft_approved", "user": self.user, "retailer": self.retailer, "item_key": key,
            "kind": draft["kind"], "to": draft["to"], "playbook_entry": draft.get("entry_id"),
            "edited": edited_text.strip() != draft["text"].strip(), "sent_by_tool": False})

    # --- data issues -------------------------------------------------------------
    def data_owner_note(self) -> str | None:
        di = self.ds.data_issues
        return draft_data_owner_note(self.retailer, di) if len(di) else None

    # --- plain-language questions -------------------------------------------------
    def ask(self, question: str) -> QAAnswer:
        try:
            model = self.qa_model
        except Exception as e:
            return QAAnswer(question, f"The assistant is unavailable: {e}", note="no model")
        a = answer(self, question, model, available_retailers(self.data_dir))
        audit.append(self.log_dir / config.REQUEST_LOG.name, {
            "event": "question", "user": self.user, "retailer": self.retailer,
            **{k: v for k, v in a.to_dict().items() if k != "tool_results"}})
        return a

    def decisions(self) -> list[dict]:
        return [r for r in audit.read(self.log_dir / config.DECISION_LOG.name) if r.get("retailer") == self.retailer]
