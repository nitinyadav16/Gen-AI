"""Presentation layer for the Streamlit interface.

All styling lives here so `app.py` stays about behaviour. The CSS targets
Streamlit's stable `data-testid` hooks (verified against the running app)
rather than generated class names.
"""

from __future__ import annotations

import html

import streamlit as st

# Colour tone per knowledge source, keyed by CollectionSpec.source_label.
SOURCE_TONES = {
    "FAQ": "faq",
    "TICKET": "ticket",
    "GUIDE": "guide",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --nc-brand:      #4f46e5;
  --nc-brand-dark: #4338ca;
  --nc-brand-soft: #eef0ff;
  --nc-ink:        #141625;
  --nc-ink-2:      #565d75;
  --nc-ink-3:      #8b91a6;
  --nc-line:       #e6e8f2;
  --nc-surface:    #f7f8fd;
  --nc-faq:        #2563eb;  --nc-faq-bg:    #eaf1ff;
  --nc-ticket:     #c2410c;  --nc-ticket-bg: #fff1e9;
  --nc-guide:      #0f766e;  --nc-guide-bg:  #e5f6f4;
}

html, body, [data-testid="stAppViewContainer"] {
  font-family: 'Inter', -apple-system, 'Segoe UI', system-ui, sans-serif;
  color: var(--nc-ink);
}
/* Streamlit's own heading/markdown rules are more specific than the block above,
   so name the elements. Icon spans are left alone - they carry a symbol font. */
h1, h2, h3, h4, p, li, label,
[data-testid="stHeading"], [data-testid="stMarkdown"],
[data-testid="stChatMessageContent"], [data-testid="stChatInputTextArea"],
[data-testid="stExpander"] summary, [data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"] {
  font-family: 'Inter', -apple-system, 'Segoe UI', system-ui, sans-serif;
}

/* --- chrome ------------------------------------------------------------ */
[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppDeployButton"] { display: none; }
[data-testid="stMainBlockContainer"] { max-width: 830px; padding-top: 2.4rem; }

/* --- hero -------------------------------------------------------------- */
.nc-hero { display:flex; align-items:center; gap:1rem; margin:.5rem 0 1.1rem; }
.nc-hero-mark {
  width:56px; height:56px; flex:none; border-radius:17px;
  background:linear-gradient(140deg, var(--nc-brand), #8b5cf6);
  display:grid; place-items:center; font-size:1.6rem;
  box-shadow:0 6px 18px -6px rgba(79,70,229,.55);
}
.nc-hero-title { font-size:1.85rem; font-weight:700; letter-spacing:-.02em; margin:0; line-height:1.15;
  font-family:'Inter', -apple-system, 'Segoe UI', system-ui, sans-serif; color:var(--nc-ink); }
.nc-hero p { color:var(--nc-ink-2); font-size:.95rem; margin:.28rem 0 0; line-height:1.45; }

.nc-bar { display:flex; align-items:center; gap:.7rem; margin:.2rem 0 1.4rem;
          padding-bottom:.9rem; border-bottom:1px solid var(--nc-line); }
.nc-bar-mark { width:32px; height:32px; flex:none; border-radius:10px;
  background:linear-gradient(140deg, var(--nc-brand), #8b5cf6);
  display:grid; place-items:center; font-size:.95rem; }
.nc-bar-name { font-weight:650; font-size:.98rem; }
.nc-bar-note { color:var(--nc-ink-3); font-size:.8rem; margin-left:auto; }

/* --- knowledge pills --------------------------------------------------- */
.nc-pills { display:flex; flex-wrap:wrap; gap:.5rem; margin:0 0 1.6rem; }
.nc-pill {
  display:inline-flex; align-items:center; gap:.45rem;
  padding:.34rem .72rem; border-radius:999px; font-size:.78rem; font-weight:500;
  background:var(--nc-surface); border:1px solid var(--nc-line); color:var(--nc-ink-2);
}
.nc-dot { width:7px; height:7px; border-radius:50%; flex:none; }
.nc-dot--faq { background:var(--nc-faq); }
.nc-dot--ticket { background:var(--nc-ticket); }
.nc-dot--guide { background:var(--nc-guide); }
.nc-pill b { color:var(--nc-ink); font-weight:650; }

/* --- chat -------------------------------------------------------------- */
[data-testid="stChatMessage"] { background:transparent; padding:0; margin-bottom:1.5rem; gap:.75rem; }
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] { padding:0; }

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { flex-direction:row-reverse; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
  [data-testid="stChatMessageContent"] {
  background:var(--nc-brand); color:#fff; padding:.65rem 1rem;
  border-radius:16px 16px 5px 16px; max-width:76%; margin-left:auto; width:fit-content;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p { color:#fff; margin:0; }

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stChatMessageContent"] {
  background:#fff; border:1px solid var(--nc-line); border-radius:5px 16px 16px 16px;
  padding:.9rem 1.15rem;
}
[data-testid="stChatMessageAvatarUser"] { background:var(--nc-brand-dark); color:#fff; }
[data-testid="stChatMessageAvatarAssistant"] {
  background:linear-gradient(140deg, var(--nc-brand), #8b5cf6); color:#fff;
}

/* --- sources ----------------------------------------------------------- */
[data-testid="stExpander"] { border:none; margin-top:.65rem; }
[data-testid="stExpander"] details {
  border:1px solid var(--nc-line); border-radius:12px; background:var(--nc-surface);
}
[data-testid="stExpander"] summary { font-size:.83rem; font-weight:600; color:var(--nc-ink-2); }
[data-testid="stExpanderDetails"] { padding-top:.2rem; }

.nc-src { border-top:1px solid var(--nc-line); padding:.75rem 0 .25rem; }
.nc-src:first-child { border-top:none; padding-top:.35rem; }
.nc-src-head { display:flex; align-items:center; gap:.5rem; flex-wrap:wrap; margin-bottom:.3rem; }
.nc-badge {
  font-size:.65rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase;
  padding:.16rem .45rem; border-radius:5px;
}
.nc-badge--faq { background:var(--nc-faq-bg); color:var(--nc-faq); }
.nc-badge--ticket { background:var(--nc-ticket-bg); color:var(--nc-ticket); }
.nc-badge--guide { background:var(--nc-guide-bg); color:var(--nc-guide); }
.nc-src-id { font-size:.75rem; font-weight:650; color:var(--nc-ink); font-variant-numeric:tabular-nums; }
.nc-src-title { font-size:.78rem; color:var(--nc-ink-3); }
/* Qualified with .nc-src so this outranks Streamlit's markdown paragraph rule. */
.nc-src p.nc-src-body { font-size:.8rem; line-height:1.55; color:var(--nc-ink-2); margin:0; }

/* --- feedback ---------------------------------------------------------- */
[data-testid="stFeedback"] { margin-top:.15rem; }
[data-testid="stFeedbackButton"] { color:var(--nc-ink-3); }

/* --- buttons ----------------------------------------------------------- */
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
  width:100%; text-align:left; justify-content:flex-start;
  background:#fff; border:1px solid var(--nc-line); border-radius:10px;
  padding:.5rem .7rem; font-size:.83rem; font-weight:450; color:var(--nc-ink-2);
  line-height:1.35; min-height:0; transition:all .12s ease;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
  border-color:var(--nc-brand); color:var(--nc-brand); background:var(--nc-brand-soft);
}
/* A Streamlit button label sits in two nested flex wrappers, both set to
   justify-content:center. These buttons are list rows and cards, so the label
   has to start at the left edge - override the wrappers, not just the text. */
[data-testid="stBaseButton-secondary"] > div,
[data-testid="stBaseButton-primary"] > div,
[data-testid="stBaseButton-secondary"] > div > span,
[data-testid="stBaseButton-primary"] > div > span {
  justify-content:flex-start; width:100%;
}
[data-testid="stBaseButton-secondary"] p,
[data-testid="stBaseButton-primary"] p,
[data-testid="stBaseButton-secondary"] [data-testid="stMarkdownContainer"],
[data-testid="stBaseButton-primary"] [data-testid="stMarkdownContainer"] {
  text-align:left; width:100%;
}

[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
  width:100%; background:#fff; color:var(--nc-ink-2);
  border:1px solid var(--nc-line); border-radius:10px; font-size:.82rem; font-weight:500;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover {
  border-color:#dc2626; color:#dc2626; background:#fef2f2;
}

/* Streamlit sizes each element wrapper to its content, so a button's own
   width:100% has nothing to fill. Stretch the wrappers that hold buttons. */
[data-testid="stSidebar"] [data-testid="stElementContainer"],
[data-testid="stMainBlockContainer"] [data-testid="stElementContainer"]:has([data-testid="stButton"]) {
  width:100%; align-self:stretch;
}
[data-testid="stSidebar"] [data-testid="stButton"],
[data-testid="stMainBlockContainer"] [data-testid="stButton"] { width:100%; }

/* suggestion chips in the empty state */
[data-testid="stMainBlockContainer"] [data-testid="stBaseButton-secondary"] {
  width:100%; text-align:left; justify-content:flex-start;
  background:var(--nc-surface); border:1px solid var(--nc-line); border-radius:12px;
  padding:.7rem .9rem; font-size:.85rem; font-weight:450; color:var(--nc-ink-2);
  line-height:1.4; min-height:0; transition:all .12s ease;
}
[data-testid="stMainBlockContainer"] [data-testid="stBaseButton-secondary"]:hover {
  border-color:var(--nc-brand); color:var(--nc-brand); background:var(--nc-brand-soft);
}

/* --- sidebar ----------------------------------------------------------- */
[data-testid="stSidebar"] { background:var(--nc-surface); border-right:1px solid var(--nc-line); }
[data-testid="stSidebarUserContent"] { padding-top:1.4rem; }
.nc-side-brand { display:flex; align-items:center; gap:.6rem; margin-bottom:.4rem; }
.nc-side-mark { width:34px; height:34px; border-radius:11px;
  background:linear-gradient(140deg, var(--nc-brand), #8b5cf6);
  display:grid; place-items:center; font-size:1rem; }
.nc-side-name { font-weight:700; font-size:1rem; letter-spacing:-.01em; }
.nc-side-sub { color:var(--nc-ink-3); font-size:.78rem; line-height:1.45; margin:0 0 1.1rem; }
.nc-label {
  font-size:.68rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase;
  color:var(--nc-ink-3); margin:1.2rem 0 .55rem;
}
.nc-kb { display:flex; align-items:center; gap:.55rem; padding:.34rem 0; font-size:.8rem; }
.nc-kb-name { font-weight:600; color:var(--nc-ink); }
.nc-kb-desc { color:var(--nc-ink-3); margin-left:auto; font-variant-numeric:tabular-nums; }
.nc-meta { font-size:.7rem; color:var(--nc-ink-3); line-height:1.6; margin-top:1.1rem;
           padding-top:.85rem; border-top:1px solid var(--nc-line); }
.nc-meta code { background:transparent; color:var(--nc-ink-2); font-size:.7rem; padding:0; }

/* --- input ------------------------------------------------------------- */
[data-testid="stChatInput"] { border-radius:14px; border:1px solid var(--nc-line); background:#fff; }
[data-testid="stChatInput"]:focus-within {
  border-color:var(--nc-brand); box-shadow:0 0 0 3px var(--nc-brand-soft);
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(subtitle: str) -> None:
    st.markdown(
        f"""<div class="nc-hero">
              <div class="nc-hero-mark">&#128246;</div>
              <div>
                <div class="nc-hero-title">NovaCell Support</div>
                <p>{html.escape(subtitle)}</p>
              </div>
            </div>""",
        unsafe_allow_html=True,
    )


def compact_header(note: str) -> None:
    """Slim header shown once the conversation is under way."""
    st.markdown(
        f"""<div class="nc-bar">
              <div class="nc-bar-mark">&#128246;</div>
              <div class="nc-bar-name">NovaCell Support</div>
              <div class="nc-bar-note">{html.escape(note)}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def knowledge_pills(status) -> None:
    """status: iterable of (CollectionSpec, document count)."""
    pills = "".join(
        f'<span class="nc-pill">'
        f'<span class="nc-dot nc-dot--{SOURCE_TONES.get(spec.source_label, "faq")}"></span>'
        f'<b>{n}</b> {html.escape(spec.description.lower())}</span>'
        for spec, n in status
    )
    st.markdown(f'<div class="nc-pills">{pills}</div>', unsafe_allow_html=True)


def source_cards(sources: list[dict]) -> None:
    cards = []
    for source in sources:
        tone = SOURCE_TONES.get(source["label"], "faq")
        title = (
            f'<span class="nc-src-title">{html.escape(source["title"])}</span>'
            if source.get("title")
            else ""
        )
        cards.append(
            f'<div class="nc-src"><div class="nc-src-head">'
            f'<span class="nc-badge nc-badge--{tone}">{html.escape(source["label"])}</span>'
            f'<span class="nc-src-id">{html.escape(source["ident"])}</span>{title}</div>'
            f'<p class="nc-src-body">{html.escape(source["content"])}</p></div>'
        )
    st.markdown("".join(cards), unsafe_allow_html=True)


def sidebar_brand() -> None:
    st.markdown(
        """<div class="nc-side-brand">
             <div class="nc-side-mark">&#128246;</div>
             <div class="nc-side-name">NovaCell</div>
           </div>
           <p class="nc-side-sub">Self-serve support, grounded in NovaCell's own
           FAQ, resolved tickets, and user guides.</p>""",
        unsafe_allow_html=True,
    )


def label(text: str) -> None:
    st.markdown(f'<div class="nc-label">{html.escape(text)}</div>', unsafe_allow_html=True)


def knowledge_rows(status) -> None:
    rows = "".join(
        f'<div class="nc-kb">'
        f'<span class="nc-dot nc-dot--{SOURCE_TONES.get(spec.source_label, "faq")}"></span>'
        f'<span class="nc-kb-name">{html.escape(spec.source_label)}</span>'
        f'<span class="nc-kb-desc">{n} docs</span></div>'
        for spec, n in status
    )
    st.markdown(rows, unsafe_allow_html=True)


def meta(model: str, store: str) -> None:
    st.markdown(
        f'<div class="nc-meta">Model <code>{html.escape(model)}</code><br>'
        f'Store <code>{html.escape(store)}</code></div>',
        unsafe_allow_html=True,
    )
