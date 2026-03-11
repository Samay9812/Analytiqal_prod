"""
ANALYTIQAL — AI Chat Assistant
Persistent expander at the bottom of every page.

Usage — one call at the end of each page block in app.py:
    from assistant_chat import render_chat
    render_chat()
"""

import streamlit as st
from typing import Dict, Any, List
from datetime import datetime

# ============================================================================
# CSS
# ============================================================================

CHAT_CSS = """
<style>

/* ── Chat message bubbles ────────────────────────────────────────── */
.msg-wrap-user {
    display: flex;
    justify-content: flex-end;
    margin: 6px 0;
}
.msg-wrap-bot {
    display: flex;
    justify-content: flex-start;
    margin: 6px 0;
}
.msg-user {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 9px 14px;
    border-radius: 16px 16px 4px 16px;
    font-size: 0.84rem;
    max-width: 78%;
    line-height: 1.5;
    word-wrap: break-word;
    box-shadow: 0 2px 6px rgba(102,126,234,0.25);
}
.msg-bot {
    background: #f4f5fb;
    color: #2d3148;
    padding: 9px 14px;
    border-radius: 16px 16px 16px 4px;
    font-size: 0.84rem;
    max-width: 88%;
    line-height: 1.55;
    border: 1px solid #e2e5f0;
    word-wrap: break-word;
}
.msg-ts {
    font-size: 0.65rem;
    color: #b0b8d0;
    margin-top: 2px;
    padding: 0 4px;
}

/* ── Workflow step card ───────────────────────────────────────────── */
.cs {
    background: white;
    border: 1px solid #e2e5f0;
    border-left: 3px solid #667eea;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 5px 0;
    font-size: 0.81rem;
}
.cs-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 19px; height: 19px;
    background: #667eea;
    color: white;
    border-radius: 50%;
    font-size: 0.64rem;
    font-weight: 700;
    margin-right: 5px;
}
.cs-page  { font-weight: 600; color: #3d4890; }
.cs-action{ color: #555e7a; margin-top: 3px; padding-left: 24px; }
.cs-detail{ color: #9098b8; font-size: 0.75rem; padding-left: 24px; margin-top: 2px; }

/* ── Recommendation card ──────────────────────────────────────────── */
.cr { border-radius: 7px; padding: 7px 11px; margin: 4px 0; font-size: 0.79rem; }
.cr-critical { background:#fff5f5; border-left:3px solid #e74c3c; }
.cr-high     { background:#fff8f0; border-left:3px solid #e67e22; }
.cr-medium   { background:#fffbf0; border-left:3px solid #f1c40f; }
.cr-low      { background:#f0fff4; border-left:3px solid #27ae60; }
.cr-title    { font-weight:600; color:#2d3148; margin-bottom:2px; }
.cr-detail   { color:#555e7a; }
.cr-nav      { color:#667eea; font-size:0.74rem; margin-top:3px; }

/* ── Quality badge ────────────────────────────────────────────────── */
.cq {
    display:inline-flex; align-items:center; gap:6px;
    padding:5px 13px; border-radius:20px;
    font-size:0.83rem; font-weight:700; margin:5px 0 10px 0;
}
.cq-A,.cq-B { background:#eafaf1; color:#1e8449; border:1px solid #a9dfbf; }
.cq-C       { background:#fef9e7; color:#b7860b; border:1px solid #f9e79f; }
.cq-D,.cq-F { background:#fdecea; color:#c0392b; border:1px solid #f5b7b1; }

/* ── Suggestion strip ─────────────────────────────────────────────── */
.csugg {
    background:#f0f4ff; border-left:3px solid #667eea;
    padding:6px 11px; border-radius:6px;
    font-size:0.79rem; color:#3d4890; margin:3px 0;
}

/* ── Warning strip ────────────────────────────────────────────────── */
.cwarn {
    background:#fff3cd; border-left:3px solid #f39c12;
    padding:6px 11px; border-radius:6px;
    font-size:0.79rem; margin:3px 0;
}

/* ── Example chip ─────────────────────────────────────────────────── */
.chip {
    display:inline-block;
    background:#f0f4ff; border:1px solid #d0d8f8;
    border-radius:14px; padding:3px 11px;
    font-size:0.75rem; color:#3d4890;
    margin:3px 2px;
}

/* ── Divider label ────────────────────────────────────────────────── */
.chat-divider {
    text-align:center; color:#b0b8d0;
    font-size:0.7rem; letter-spacing:0.08em;
    margin:8px 0 4px 0;
}

</style>
"""

# ============================================================================
# SESSION STATE
# ============================================================================

def _init():
    for k, v in {
        "chat_history":      [],   # [{role, content, ts, result}]
        "assistant":         None,
        "chat_input_pending": "",  # buffer for shortcut buttons
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _get_assistant():
    """Return (and lazily build/update) the AnalyticsAssistant."""
    from assistant_engine import AnalyticsAssistant
    df = st.session_state.get("df")
    if df is None:
        st.session_state.assistant = None
        return None
    try:
        from utils_robust import get_column_types
        col_types = get_column_types(df)
    except Exception:
        col_types = {"numeric": [], "categorical": [], "datetime": []}

    if st.session_state.assistant is None:
        st.session_state.assistant = AnalyticsAssistant(df, col_types)
    else:
        try:
            st.session_state.assistant.update_dataset(df, col_types)
        except Exception:
            st.session_state.assistant = AnalyticsAssistant(df, col_types)
    return st.session_state.assistant


# ============================================================================
# RENDER HELPERS
# ============================================================================

def _result_html(result: Dict) -> str:
    """Turn an assistant result dict into HTML for a bot bubble."""
    parts = []

    # ── Data-aware path ────────────────────────────────────────────────────
    if result.get("is_data_aware") and result.get("data_summary"):
        summary = result["data_summary"]
        score   = summary.get("quality_score", 0)
        grade   = summary.get("quality_grade", "?")

        parts.append(
            f"<div class='cq cq-{grade}'>"
            f"📊 Quality score: <strong>{score:.0f} / 100</strong>"
            f"&nbsp; Grade {grade}"
            f"</div>"
        )

        by_p  = summary.get("by_priority", {})
        crit  = by_p.get("critical", 0)
        high  = by_p.get("high", 0)
        med   = by_p.get("medium", 0)
        low   = by_p.get("low", 0)
        total = crit + high + med + low

        if total == 0:
            parts.append(
                "<p style='color:#27ae60;font-size:0.82rem;'>"
                "✅ No issues found — data looks clean.</p>"
            )
        else:
            parts.append(
                f"<p style='font-size:0.8rem;color:#555e7a;margin:0 0 8px 0;'>"
                f"🔴 {crit} critical &nbsp;·&nbsp; "
                f"🟠 {high} high &nbsp;·&nbsp; "
                f"🟡 {med} medium &nbsp;·&nbsp; "
                f"🟢 {low} low</p>"
            )
            recs = result.get("recommendations", [])
            shown = recs[:7]
            for rec in shown:
                p    = rec.get("priority", "low")
                icon = rec.get("icon", "•")
                col  = rec.get("column", "")
                parts.append(
                    f"<div class='cr cr-{p}'>"
                    f"<div class='cr-title'>{icon} {rec.get('issue','')}"
                    + (f" <span style='font-weight:400;color:#888;'>({col})</span>" if col else "")
                    + f"</div>"
                    f"<div class='cr-detail'>{rec.get('details','')}</div>"
                    f"<div class='cr-nav'>→ {rec.get('action','')} &nbsp;·&nbsp; {rec.get('page','')}</div>"
                    f"</div>"
                )
            if len(recs) > 7:
                parts.append(
                    f"<p style='font-size:0.74rem;color:#b0b8d0;margin:4px 0 0 0;'>"
                    f"…and {len(recs)-7} more — run again to see all.</p>"
                )

    # ── Regular workflow path ──────────────────────────────────────────────
    else:
        workflow = result.get("workflow", [])
        if not workflow:
            parts.append(
                "<p style='color:#9098b8;font-size:0.82rem;'>"
                "I couldn't identify specific steps — try rephrasing.</p>"
            )
        else:
            conf = result.get("confidence", 0)
            conf_str = (
                "🟢 High confidence" if conf >= 0.8 else
                "🟡 Medium confidence" if conf >= 0.5 else
                "⚠️ Low confidence — review carefully"
            )
            parts.append(
                f"<p style='font-size:0.74rem;color:#b0b8d0;margin:0 0 6px 0;'>"
                f"{conf_str}</p>"
            )
            for step in workflow:
                parts.append(
                    f"<div class='cs'>"
                    f"<div style='display:flex;align-items:center;'>"
                    f"<span class='cs-num'>{step['step']}</span>"
                    f"<span class='cs-page'>{step['page']}</span></div>"
                    f"<div class='cs-action'>→ {step['action']}</div>"
                    + (f"<div class='cs-detail'>ℹ️ {step['details']}</div>" if step.get("details") else "")
                    + (f"<div class='cs-detail'>💡 {step['guidance']}</div>" if step.get("guidance") else "")
                    + "</div>"
                )

    # ── Shared: suggestions + warnings ────────────────────────────────────
    for s in result.get("suggestions", []):
        parts.append(f"<div class='csugg'>{s}</div>")
    for w in result.get("warnings", []):
        parts.append(f"<div class='cwarn'>⚠️ {w}</div>")

    return "".join(parts) or "<p style='color:#9098b8;'>Done.</p>"


def _render_message(msg: Dict):
    role   = msg["role"]
    ts     = msg.get("ts", "")
    result = msg.get("result")

    if role == "user":
        st.markdown(
            f"<div class='msg-wrap-user'>"
            f"<div>"
            f"<div class='msg-user'>{msg['content']}</div>"
            f"<div class='msg-ts' style='text-align:right;'>{ts}</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
    else:
        if result and result.get("success"):
            body = _result_html(result)
        elif result and not result.get("success"):
            err  = result.get("error", "Something went wrong.")
            sugg = result.get("suggestions", [])[:4]
            body = (
                f"<p style='color:#e74c3c;font-size:0.83rem;margin:0 0 6px 0;'>{err}</p>"
                + "".join(f"<span class='chip'>{s}</span>" for s in sugg)
            )
        else:
            body = f"<p style='font-size:0.84rem;margin:0;'>{msg['content']}</p>"

        st.markdown(
            f"<div class='msg-wrap-bot'>"
            f"<div>"
            f"<div class='msg-bot'>{body}</div>"
            f"<div class='msg-ts'>{ts}</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )


def _empty_state():
    df = st.session_state.get("df")
    if df is None:
        st.info("📂 Load a dataset to start using the assistant.")
        return

    st.markdown(
        "<p style='font-size:0.82rem;color:#9098b8;margin:4px 0 8px 0;'>"
        "Ask me anything about your data:</p>",
        unsafe_allow_html=True,
    )
    examples = [
        "What should I do to clean my data?",
        "Check my data quality",
        "What issues does my data have?",
        "Remove missing values and duplicates",
        "Suggest visualizations",
        "Prepare for analysis",
    ]
    st.markdown(
        "".join(f"<span class='chip'>{e}</span>" for e in examples),
        unsafe_allow_html=True,
    )


# ============================================================================
# MAIN COMPONENT
# ============================================================================

def render_chat(page_name: str = ""):
    """
    Call at the bottom of every page block in app.py.

    Parameters
    ----------
    page_name : str
        Current page label — injected into assistant context.
    """
    _init()
    st.markdown(CHAT_CSS, unsafe_allow_html=True)

    history = st.session_state.chat_history

    # Count unread (messages since last open — simple proxy: total bot messages)
    n_bot = sum(1 for m in history if m["role"] == "assistant")
    label = f"🤖 Analytics Assistant" + (f"  ·  {n_bot} response{'s' if n_bot != 1 else ''}" if n_bot else "")

    with st.expander(label, expanded=False):

        # ── Header strip ──────────────────────────────────────────────────
        df = st.session_state.get("df")
        if df is not None:
            name  = st.session_state.get("raw_name", "Dataset")
            n_r   = len(df)
            n_c   = len(df.columns)
            st.markdown(
                f"<div style='background:#f4f5fb;border:1px solid #e2e5f0;"
                f"border-radius:8px;padding:7px 12px;margin-bottom:10px;"
                f"font-size:0.78rem;color:#555e7a;display:flex;gap:16px;'>"
                f"<span>📁 <strong>{name}</strong></span>"
                f"<span>{n_r:,} rows · {n_c} cols</span>"
                + (f"<span style='color:#667eea;'>📍 {page_name}</span>" if page_name else "")
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='background:#f4f5fb;border:1px solid #e2e5f0;"
                f"border-radius:8px;padding:7px 12px;margin-bottom:10px;"
                f"font-size:0.78rem;color:#9098b8;'>"
                f"No dataset loaded"
                + (f" &nbsp;·&nbsp; <span style='color:#667eea;'>📍 {page_name}</span>" if page_name else "")
                + "</div>",
                unsafe_allow_html=True,
            )

        # ── Message history ────────────────────────────────────────────────
        if not history:
            _empty_state()
        else:
            for i, msg in enumerate(history):
                _render_message(msg)
                # Thin divider between exchanges
                if i < len(history) - 1 and msg["role"] == "assistant":
                    st.markdown(
                        "<div class='chat-divider'>· · ·</div>",
                        unsafe_allow_html=True,
                    )

        # ── Input row ──────────────────────────────────────────────────────
        st.markdown(
            "<div style='height:6px;'></div>",
            unsafe_allow_html=True,
        )
        # Consume any pending value set by shortcut buttons (must happen
        # before the widget is instantiated, not after)
        _pending = st.session_state.pop("chat_input_pending", "")

        ic1, ic2, ic3 = st.columns([6, 1, 1])
        with ic1:
            user_input = st.text_input(
                "Message",
                value=_pending,
                placeholder="e.g. What should I do to clean my data?",
                key="chat_input",
                label_visibility="collapsed",
            )
        with ic2:
            send = st.button("Send", key="chat_send",
                             use_container_width=True, type="primary")
        with ic3:
            if st.button("Clear", key="chat_clear", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        # ── Process ────────────────────────────────────────────────────────
        if send and user_input.strip():
            ts        = datetime.now().strftime("%H:%M")
            assistant = _get_assistant()

            st.session_state.chat_history.append({
                "role": "user", "content": user_input.strip(), "ts": ts,
            })

            if assistant is None:
                result = {
                    "success": False,
                    "error":   "⚠️ Please upload a dataset first.",
                    "workflow": [], "warnings": [], "suggestions": [],
                    "recommendations": [],
                }
                reply = "Please upload a dataset first."
            else:
                ctx     = f"[Page: {page_name}] " if page_name else ""
                result  = assistant.process_query(ctx + user_input.strip())

                if result["success"]:
                    if result.get("is_data_aware") and result.get("data_summary"):
                        score = result["data_summary"].get("quality_score", 0)
                        reply = f"Here's your data quality analysis (score: {score:.0f}/100):"
                    else:
                        n = len(result.get("workflow", []))
                        reply = f"Here's a {n}-step workflow:" if n else "Here's what I found:"
                else:
                    reply = "I couldn't fully understand that — see below."

            st.session_state.chat_history.append({
                "role": "assistant", "content": reply,
                "ts": ts, "result": result,
            })

            # Cap history at 30 messages (15 exchanges)
            if len(st.session_state.chat_history) > 30:
                st.session_state.chat_history = st.session_state.chat_history[-30:]

            st.rerun()

        # ── Example query shortcuts ────────────────────────────────────────
        if not history and df is not None:
            st.markdown(
                "<p style='font-size:0.74rem;color:#b0b8d0;"
                "margin:10px 0 4px 0;'>Quick starters:</p>",
                unsafe_allow_html=True,
            )
            shortcuts = [
                "What should I do to clean my data?",
                "Check my data quality",
                "Suggest next steps",
            ]
            sc = st.columns(len(shortcuts))
            for col, q in zip(sc, shortcuts):
                with col:
                    if st.button(q, key=f"qs_{q[:20]}", use_container_width=True):
                        # Use a pending buffer — can't write to widget key
                        # after it's been instantiated in the same run
                        st.session_state["chat_input_pending"] = q
                        st.rerun()