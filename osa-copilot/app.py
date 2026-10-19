"""OSA Root-Cause Copilot - Streamlit UI (NFR Usability: one worklist screen).

Run:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from osa import causes as C
from osa import config
from osa.copilot import ACCEPT, EDIT, REASON_CODES, REJECT, ROUTE, Copilot
from osa.data import available_retailers
from osa.detect import worklist_frame
from osa.diagnose import CONFIDENT, NEEDS_INVESTIGATION, UNAVAILABLE
from osa.evidence import fmt

st.set_page_config(page_title="OSA Root-Cause Copilot", page_icon="🛒", layout="wide")

CONF_BADGE = {"high": "🟢 high", "medium": "🟡 medium", "low": "⚪ low"}


# --- session --------------------------------------------------------------------
with st.sidebar:
    st.title("OSA Root-Cause Copilot")
    user = st.text_input("Signed in as", value="retail.exec.manager")
    retailer = st.selectbox("Retailer scope (set by your account)", available_retailers(),
                            help="A manager is scoped to one retailer; other retailers' data is never loaded.")

sess_key = f"{retailer}|{user}"
if st.session_state.get("sess_key") != sess_key:
    st.session_state.sess_key = sess_key
    st.session_state.cp = Copilot(retailer, user=user)
    st.session_state.outcomes = {}
    st.session_state.chat = []
cp: Copilot = st.session_state.cp
outcomes: dict = st.session_state.outcomes

with st.sidebar:
    r = cp.ds.rules
    st.caption(f"Week ending **{cp.ds.latest_week}** · refreshed {cp.ds.as_of}")
    st.caption(f"Flag rule: OSA < {r.osa_target:g}% or drop > {r.drop_points:g} pts vs "
               f"{r.baseline_weeks}-week baseline (min {r.min_history_weeks} weeks history)")
    st.divider()
    n = st.slider("Diagnose the top N items", 1, max(1, len(cp.worklist)), min(10, len(cp.worklist)))
    if st.button("Run diagnosis", type="primary", width="stretch"):
        bar = st.progress(0.0, text="Diagnosing...")
        for j, item in enumerate(cp.worklist[:n], 1):
            cp.diagnose(item.key)
            bar.progress(j / n, text=f"Diagnosed {j}/{n}")
        bar.empty()
    st.divider()
    st.caption(f"Model `{config.DIAGNOSIS_MODEL}` · prompt `{config.DIAGNOSIS_PROMPT_VERSION}`")
    st.caption("Drafts only. Nothing leaves this tool: you review, approve and send every message yourself.")


def status_of(key: str) -> str:
    if key in outcomes:
        return f"✅ {outcomes[key]['record']['decision'].replace('_', ' ')}"
    d = cp.diagnoses.get(key)
    if d is None:
        return "Not diagnosed"
    if d.status == CONFIDENT:
        return f"{C.label(d.top_cause)} ({d.causes[0]['confidence']})"
    return {NEEDS_INVESTIGATION: "⚠️ Needs investigation", UNAVAILABLE: "❌ Diagnosis unavailable"}[d.status]


# --- header metrics ------------------------------------------------------------
diagnosed = [d for d in cp.diagnoses.values()]
ni = sum(d.status == NEEDS_INVESTIGATION for d in diagnosed)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("At-risk store-SKUs", len(cp.worklist))
c2.metric("Est. lost sales", f"${sum(i.lost_sales for i in cp.worklist):,.0f}")
c3.metric("Data issues (excluded)", cp.ds.data_issues.item_key.nunique())
c4.metric("Diagnosed", f"{len(diagnosed)}/{len(cp.worklist)}")
c5.metric("Needs investigation", f"{ni}" + (f" ({100 * ni / len(diagnosed):.0f}%)" if diagnosed else ""))

tab_work, tab_issues, tab_ask, tab_audit = st.tabs(["Worklist", "Data issues", "Ask about OSA", "Audit log"])


# --- worklist -----------------------------------------------------------------
def render_diagnosis(key: str) -> None:
    item, pack = cp.item(key), cp.evidence(key)
    d = cp.diagnoses.get(key)
    if d is None:
        if st.button("Diagnose this item", key=f"diag-{key}", type="primary"):
            with st.spinner("Weighing the evidence..."):
                cp.diagnose(key)
            st.rerun()
        return

    st.caption(f"Model `{d.model}` · prompt `{d.prompt_version}` · {d.latency_s}s · attempts {d.attempts}"
               + (" · cached" if d.cached else ""))
    if d.status == UNAVAILABLE:
        st.error("Diagnosis unavailable. The rule flags and lost-sales ranking above still apply.\n\n"
                 f"{d.reason}")
        if st.button("Retry diagnosis", key=f"retry-{key}"):
            cp.diagnose(key, use_cache=False)
            st.rerun()
        return
    if d.status == NEEDS_INVESTIGATION:
        st.warning(f"**Needs investigation.** {d.reason}. No action is recommended; route this to an analyst.")
    else:
        st.markdown("#### Likely causes")
        for i, c in enumerate(d.causes, 1):
            with st.container(border=True):
                st.markdown(f"**{i}. {c['label']}** · confidence {CONF_BADGE[c['confidence']]} · "
                            f"rule check *{c['rule_check']}*")
                if c["rationale"]:
                    st.write(c["rationale"])
                for e in c["evidence"]:
                    st.markdown(f"- `{e['signal_id']}` {e['text']}")
        if d.summary:
            st.info(d.summary)
        e = d.playbook_entry
        st.markdown("#### Recommended action")
        with st.container(border=True):
            st.markdown(f"**{e['action']}**")
            st.markdown(f"Owner: **{e['owner']}** · Target turnaround: **{e['turnaround']}** · "
                        f"Playbook entry `{e['entry_id']}`")
            if d.upstream:
                st.markdown("⬆️ Upstream cause: on approval an escalation with the evidence pack is drafted "
                            "for the Supply Chain Planner.")

    st.markdown("#### Your decision")
    if key in outcomes:
        render_outcome(key)
        return
    options = ([ACCEPT] if d.status == CONFIDENT else [ROUTE]) + [EDIT, REJECT]
    labels = {ACCEPT: "Accept", ROUTE: "Route to analyst", EDIT: "Edit cause", REJECT: "Reject"}
    with st.form(f"decide-{key}"):
        choice = st.radio("Decision", options, format_func=labels.get, horizontal=True)
        edited = st.selectbox("Corrected cause (for Edit)", C.CAUSE_KEYS, format_func=C.label)
        reason = st.selectbox("Reason (required for Reject or Edit)", [""] + REASON_CODES)
        note = st.text_input("Note (optional)")
        if st.form_submit_button("Record decision", type="primary"):
            try:
                outcomes[key] = cp.decide(key, choice, reason, note, edited if choice == EDIT else None)
                st.rerun()
            except ValueError as err:
                st.error(str(err))


def render_outcome(key: str) -> None:
    out = outcomes[key]
    rec, draft = out["record"], out["draft"]
    st.success(f"Decision **{rec['decision'].replace('_', ' ')}** logged at {rec['timestamp']}"
               + (f" · reason: {rec['reason_code']}" if rec["reason_code"] else ""))
    if not draft:
        return
    title = {"escalation": "Escalation to Supply Chain Planner", "brief": "Brief",
             "analyst_note": "Note to analyst"}[draft["kind"]]
    st.markdown(f"#### Draft: {title} (to {draft['to']})")
    text = st.text_area("Edit before you send", draft["text"], height=320, key=f"draft-{key}")
    if out.get("approved"):
        st.success("Approved. Copy the text and send it yourself; the copilot does not send messages.")
    elif st.button("Approve draft", key=f"approve-{key}", type="primary"):
        cp.approve_draft(key, draft, text)
        out["approved"] = True
        st.rerun()


with tab_work:
    if not cp.worklist:
        st.info("No store-SKUs are at risk this week.")
    else:
        df = worklist_frame(cp.worklist)
        df.insert(1, "status", [status_of(k) for k in df["key"]])
        view = df.drop(columns=["key"]).rename(columns={
            "osa_latest": "OSA %", "osa_baseline": "baseline OSA %", "weeks_affected": "weeks",
            "baseline_units": "baseline units/wk", "price": "price $", "est_lost_sales": "est. lost sales $"})
        sel = st.dataframe(view, hide_index=True, width="stretch", on_select="rerun",
                           selection_mode="single-row", key="worklist",
                           column_config={"est. lost sales $": st.column_config.NumberColumn(format="$%.2f"),
                                          "price $": st.column_config.NumberColumn(format="$%.2f")})
        rows = sel.selection.rows if sel and sel.selection else []
        key = df.iloc[rows[0]]["key"] if rows else df.iloc[0]["key"]
        st.caption("Select a row to review it. Lost sales = baseline weekly units x price x weeks affected x "
                   "(1 - OSA over the affected weeks); an estimate.")

        item, pack = cp.item(key), cp.evidence(key)
        st.divider()
        st.subheader(f"#{item.rank} · Store {item.store_id} ({item.region}) · {item.sku} {item.sku_name}")
        a, b, c = st.columns(3)
        a.metric("OSA latest week", fmt(item.osa_latest, "%"),
                 f"{item.osa_latest - item.osa_baseline:+.1f} pts vs baseline", delta_color="normal")
        b.metric("Target / 4-week baseline", f"{fmt(item.osa_target, '%')} / {fmt(item.osa_baseline, '%')}")
        c.metric("Est. lost sales (estimate)", f"${item.lost_sales:,.2f}")
        st.caption(f"Flag: {item.flag_reason}. Formula: {item.formula()}")

        left, right = st.columns([3, 2])
        with left:
            render_diagnosis(key)
        with right:
            st.markdown("#### Evidence pack")
            st.dataframe(pd.DataFrame([{"id": s.id, "signal": s.name, "value": fmt(s.value, s.unit),
                                        "compared to": "" if s.baseline is None else
                                        f"{s.basis}: {fmt(s.baseline, s.unit)}"} for s in pack.signals]),
                         hide_index=True, width="stretch")
            st.markdown("**Rule checks**")
            st.dataframe(pd.DataFrame([{"cause": C.label(c.cause), "result": c.result,
                                        "conditions met": "; ".join(c.met)} for c in pack.checks]),
                         hide_index=True, width="stretch")
            if pack.conflicts:
                st.warning("Conflicts: " + "; ".join(pack.conflicts))


# --- data issues -------------------------------------------------------------------
with tab_issues:
    di = cp.ds.data_issues
    stale = [f for f in cp.ds.feeds.values() if f.stale]
    if stale:
        st.error("Stale feeds: " + ", ".join(f"{f.feed} (last refreshed {f.last_refreshed})" for f in stale))
    if di.empty:
        st.success("All required feeds are fresh and complete for the latest week.")
    else:
        st.write("These store-SKUs are **excluded from ranking and diagnosis** until the data arrives. "
                 "They are not treated as out-of-stock.")
        st.dataframe(di.drop(columns=["item_key"]), hide_index=True, width="stretch")
        st.markdown("#### Draft note to the data owner")
        st.text_area("Edit before you send", cp.data_owner_note(), height=240, key="data-note")
    st.dataframe(pd.DataFrame([vars(f) for f in cp.ds.feeds.values()]), hide_index=True,
                 width="stretch")


# --- ask -----------------------------------------------------------------------------
with tab_ask:
    st.caption(f"Read-only questions about OSA for {retailer}, e.g. \"Why is OSA down in the South region?\" "
               "or \"Which 5 store-SKUs cost us the most sales?\"")
    for turn in st.session_state.chat:
        with st.chat_message("user"):
            st.write(turn["q"])
        with st.chat_message("assistant"):
            st.markdown(turn["a"])
            if turn["filters"]:
                with st.expander("Filters used"):
                    st.json(turn["filters"])
    q = st.chat_input("Ask about OSA...")
    if q:
        with st.spinner("Looking it up..."):
            ans = cp.ask(q)
        st.session_state.chat.append({"q": q, "a": ans.answer, "filters": ans.filters_used})
        st.rerun()


# --- audit ---------------------------------------------------------------------------
with tab_audit:
    log = cp.decisions()
    if not log:
        st.info("No decisions recorded yet.")
    else:
        cols = ["timestamp", "event", "user", "item_key", "decision", "final_cause", "reason_code", "note",
                "playbook_entry", "model_version", "prompt_version", "kind", "edited", "sent_by_tool"]
        st.dataframe(pd.DataFrame(log).reindex(columns=cols), hide_index=True, width="stretch")
