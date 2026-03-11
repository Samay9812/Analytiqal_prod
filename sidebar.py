"""
ANALYTIQAL — Modern Sidebar
Button-based navigation with clean visual hierarchy
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime


# ============================================================================
# CSS
# ============================================================================

SIDEBAR_CSS = """
<style>

/* ── Sidebar background — light, works with Streamlit defaults ───── */
section[data-testid="stSidebar"] {
    background: #f4f5fb !important;
}
section[data-testid="stSidebar"] > div {
    padding: 0 !important;
}

/* ── Section label ──────────────────────────────────────────────── */
.sb-label {
    font-size: 0.63rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #a0a8c0;
    padding: 14px 16px 4px 16px;
    margin: 0;
    display: block;
}

/* ── Divider ────────────────────────────────────────────────────── */
.sb-hr {
    border: none;
    border-top: 1px solid #e2e5f0;
    margin: 6px 14px;
}

/* ── Nav buttons ─────────────────────────────────────────────────── */
section[data-testid="stSidebar"] .stButton > button {
    border-radius: 9px !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    text-align: left !important;
    padding: 9px 13px !important;
    transition: all 0.16s ease !important;
    border: 1px solid transparent !important;
    width: 100% !important;
}

/* inactive nav */
section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
    background: transparent !important;
    color: #555e7a !important;
    border-color: transparent !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
    background: rgba(102, 126, 234, 0.08) !important;
    color: #3d4890 !important;
    border-color: rgba(102, 126, 234, 0.15) !important;
    transform: translateX(2px) !important;
    box-shadow: none !important;
}

/* active nav */
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg,
        rgba(102,126,234,0.14) 0%,
        rgba(118,75,162,0.14) 100%) !important;
    color: #3d4890 !important;
    border-color: rgba(102,126,234,0.35) !important;
    border-left: 3px solid #667eea !important;
    padding-left: 10px !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 6px rgba(102,126,234,0.12) !important;
}

/* utility buttons hover */
section[data-testid="stSidebar"] .stButton > button:not([kind="primary"]):hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 3px 8px rgba(0,0,0,0.1) !important;
}

/* ── Dataset chip ────────────────────────────────────────────────── */
.ds-chip {
    margin: 6px 12px 8px 12px;
    padding: 10px 13px;
    background: #fff;
    border: 1px solid #e2e5f0;
    border-radius: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.ds-name {
    font-size: 0.8rem;
    font-weight: 600;
    color: #2d3148;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 5px;
}
.ds-meta {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}
.ds-stat { font-size: 0.72rem; color: #9098b8; }
.ds-stat strong { color: #555e7a; }
.ds-badge {
    font-size: 0.68rem;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 20px;
}
.badge-ok   { background: #eafaf1; color: #27ae60; border: 1px solid #a9dfbf; }
.badge-warn { background: #fef9e7; color: #d4a017; border: 1px solid #f9e79f; }
.badge-bad  { background: #fdecea; color: #c0392b; border: 1px solid #f5b7b1; }

/* ── Project chip ────────────────────────────────────────────────── */
.proj-chip {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 2px 12px 8px 12px;
    padding: 8px 11px;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 500;
    overflow: hidden;
}
.proj-chip-loaded {
    background: #eafaf1;
    border: 1px solid #a9dfbf;
    color: #1e8449;
}
.proj-chip-empty {
    background: #f8f9fc;
    border: 1px dashed #d0d5e8;
    color: #9098b8;
}

/* ── Expander ────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] .streamlit-expanderHeader {
    background: #fff !important;
    border: 1px solid #e2e5f0 !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    color: #555e7a !important;
    padding: 8px 12px !important;
}

/* ── Inputs ──────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] div[data-baseweb="input"] > div {
    background: #fff !important;
    border-color: #d8dce8 !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
}
section[data-testid="stSidebar"] textarea {
    background: #fff !important;
    border-color: #d8dce8 !important;
    font-size: 0.82rem !important;
}

/* ── Caption ─────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] .stCaption {
    color: #9098b8 !important;
    font-size: 0.72rem !important;
    padding: 0 14px !important;
}

</style>
"""

# ============================================================================
# PAGE REGISTRY
# ============================================================================

PAGES = [
    ("🏠", "Dataset"),
    ("🔍", "Data Profiling"),
    ("🧹", "Data Cleaning"),
    ("🔄", "Transformations"),
    ("📈", "Feature Engineering"),
    ("📊", "Visualizations"),
    ("💾", "Export & Reports"),
]

def _full(icon, label): return f"{icon} {label}"


# ============================================================================
# MAIN SIDEBAR
# ============================================================================

def render_modern_sidebar():
    st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)

    with st.sidebar:

        # ── Brand ─────────────────────────────────────────────────────────
        st.markdown("""
            <div style='
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 18px 16px 16px 16px;
            '>
                <div style='display:flex; align-items:center; gap:10px;'>
                    <span style='font-size:1.5rem; line-height:1;'>📊</span>
                    <div>
                        <div style='color:#fff; font-size:0.95rem; font-weight:700;
                                    letter-spacing:0.04em; line-height:1.2;'>
                            ANALYTIQAL
                        </div>
                        <div style='color:rgba(255,255,255,0.6); font-size:0.7rem;
                                    margin-top:2px;'>
                            Professional Analytics
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # ── Undo / Redo — top priority, always accessible ─────────────────
        st.markdown('<p class="sb-label">History</p>', unsafe_allow_html=True)

        h_count = len(st.session_state.get('history', []))
        r_count = len(st.session_state.get('redo_stack', []))

        uc1, uc2 = st.columns(2)
        with uc1:
            if st.button(
                f"↩ Undo{' (' + str(h_count) + ')' if h_count else ''}",
                disabled=h_count == 0, use_container_width=True,
                key="undo_btn", help=f"{h_count} action(s) to undo",
            ):
                _undo_action()
        with uc2:
            if st.button(
                f"↪ Redo{' (' + str(r_count) + ')' if r_count else ''}",
                disabled=r_count == 0, use_container_width=True,
                key="redo_btn", help=f"{r_count} action(s) to redo",
            ):
                _redo_action()

        # ── Dataset preview — collapsible, always near top ─────────────────
        st.markdown('<hr class="sb-hr"><p class="sb-label">Dataset</p>',
                    unsafe_allow_html=True)
        _render_dataset_section(st.session_state.get('df'))

        # ── Navigation ────────────────────────────────────────────────────
        st.markdown('<hr class="sb-hr"><p class="sb-label">Pipeline</p>',
                    unsafe_allow_html=True)

        current_page  = st.session_state.get('current_page', _full(*PAGES[0]))
        selected_page = current_page

        for icon, label in PAGES:
            full = _full(icon, label)
            is_active = (current_page == full)
            if st.button(
                f"{icon}  {label}",
                key=f"nav_{label}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.current_page = full
                selected_page = full
                st.rerun()

        # ── Project Manager ───────────────────────────────────────────────
        st.markdown('<hr class="sb-hr"><p class="sb-label">Project</p>',
                    unsafe_allow_html=True)
        _render_project_manager()

        # ── Code Editors ──────────────────────────────────────────────────
        st.markdown('<hr class="sb-hr"><p class="sb-label">Code</p>',
                    unsafe_allow_html=True)
        ec1, ec2 = st.columns(2)
        py_open  = st.session_state.get('show_python_editor', False)
        sql_open = st.session_state.get('show_sql_editor',   False)
        with ec1:
            if st.button(
                f"{'🟢' if py_open else '🐍'} Python",
                use_container_width=True, key="open_py",
            ):
                st.session_state.show_python_editor = not py_open
                st.session_state.show_sql_editor    = False
                st.rerun()
        with ec2:
            if st.button(
                f"{'🟢' if sql_open else '🗄️'} SQL",
                use_container_width=True, key="open_sql",
            ):
                st.session_state.show_sql_editor    = not sql_open
                st.session_state.show_python_editor = False
                st.rerun()

        # ── Footer ────────────────────────────────────────────────────────
        st.markdown('<hr class="sb-hr">', unsafe_allow_html=True)
        st.caption("🔒 Data stays on your device · 💬 AI assistant bottom of each page")

    return selected_page


# ============================================================================
# DATASET SECTION — chip summary + collapsible preview
# ============================================================================

def _render_dataset_section(df):
    """Always-visible stats chip, with an expander for the full preview."""
    if df is None:
        st.markdown("""
            <div class="ds-chip">
                <div class="ds-name" style="color:#9098b8; font-weight:400;">
                    No dataset loaded
                </div>
            </div>
        """, unsafe_allow_html=True)
        return

    name     = st.session_state.get('raw_name', 'Untitled')
    n_rows   = len(df)
    n_cols   = len(df.columns)
    total    = max(n_rows * n_cols, 1)
    n_miss   = int(df.isnull().sum().sum())
    miss_pct = n_miss / total * 100

    if miss_pct < 5:    bcls, blbl = "badge-ok",   f"{miss_pct:.0f}% missing"
    elif miss_pct < 20: bcls, blbl = "badge-warn",  f"{miss_pct:.0f}% missing"
    else:               bcls, blbl = "badge-bad",   f"{miss_pct:.0f}% missing"

    st.markdown(f"""
        <div class="ds-chip">
            <div class="ds-name">📁 {name}</div>
            <div class="ds-meta">
                <span class="ds-stat"><strong>{n_rows:,}</strong> rows</span>
                <span class="ds-stat"><strong>{n_cols}</strong> cols</span>
                <span class="ds-badge {bcls}">{blbl}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Collapsible preview
    with st.expander("👁 Preview data", expanded=False):
        _render_dataset_preview(df)


# ============================================================================
# DATASET PREVIEW (inside expander)
# ============================================================================

def _render_project_manager():
    from utils_robust import save_project, load_project, delete_project, list_projects

    current = st.session_state.get('current_project_name', None)
    if current:
        st.markdown(f"""
            <div class="proj-chip proj-chip-loaded">
                📂 <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
                                flex:1;">{current}</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="proj-chip proj-chip-empty">📂 No project loaded</div>
        """, unsafe_allow_html=True)

    pc1, pc2 = st.columns(2)
    with pc1:
        if st.button("💾 Save", use_container_width=True, key="save_btn_sidebar"):
            st.session_state.show_save_dialog = True
            st.session_state.show_load_dialog = False
    with pc2:
        if st.button("📂 Load", use_container_width=True, key="load_btn_sidebar"):
            st.session_state.show_load_dialog = True
            st.session_state.show_save_dialog = False

    # Save panel
    if st.session_state.get('show_save_dialog', False):
        with st.expander("💾 Save Project", expanded=True):
            project_name = st.text_input(
                "Project name", value=current or "",
                placeholder="My Analysis Project",
                key="save_project_name",
                label_visibility="collapsed",
            )
            st.caption("Name this project to save your dataset and history.")
            s1, s2 = st.columns(2)
            with s1:
                if st.button("Save", use_container_width=True,
                             type="primary", key="confirm_save"):
                    if project_name.strip():
                        if save_project(project_name):
                            st.session_state.current_project_name = project_name
                            st.session_state.show_save_dialog = False
                            st.toast(f"✅ Saved '{project_name}'")
                            st.rerun()
                    else:
                        st.error("Enter a name")
            with s2:
                if st.button("Cancel", use_container_width=True, key="cancel_save"):
                    st.session_state.show_save_dialog = False
                    st.rerun()

    # Load panel
    if st.session_state.get('show_load_dialog', False):
        with st.expander("📂 Projects", expanded=True):
            projects = list_projects()
            if projects:
                for i, proj in enumerate(projects):
                    saved = datetime.fromisoformat(proj['saved_date'])
                    lc1, lc2, lc3 = st.columns([3, 1, 1])
                    with lc1:
                        st.markdown(f"**{proj['project_name']}**")
                        st.caption(
                            f"{proj['row_count']:,}×{proj['column_count']} · "
                            f"{saved.strftime('%d %b %H:%M')}"
                        )
                    with lc2:
                        if st.button("Load", key=f"load_{i}", use_container_width=True):
                            if load_project(proj['project_name']):
                                st.session_state.current_project_name = proj['project_name']
                                st.session_state.show_load_dialog = False
                                st.toast(f"✅ Loaded '{proj['project_name']}'")
                                st.rerun()
                    with lc3:
                        if st.button("🗑", key=f"del_{i}", use_container_width=True):
                            if delete_project(proj['project_name']):
                                st.toast("Deleted")
                                st.rerun()
                    st.divider()
            else:
                st.caption("No saved projects yet.")
            if st.button("✕ Close", use_container_width=True, key="close_load"):
                st.session_state.show_load_dialog = False
                st.rerun()


# ============================================================================
# DATASET PREVIEW (inside sidebar)
# ============================================================================

def _render_dataset_preview(df: pd.DataFrame):
    sort_col = st.selectbox(
        "Sort by", df.columns,
        key="sidebar_sort_col",
        label_visibility="collapsed",
    )
    pc1, pc2 = st.columns([3, 2])
    with pc1:
        ascending = st.toggle("Ascending", value=True, key="sidebar_sort_asc")
    with pc2:
        n_rows = st.slider("", 5, 20, 8, key="sidebar_n_rows")

    try:
        display_df = df.copy()
        for col in display_df.columns:
            if display_df[col].dtype.name == 'category':
                display_df[col] = display_df[col].astype(str)
        sorted_df = display_df.sort_values(
            by=sort_col, ascending=ascending, na_position='last'
        )
        st.dataframe(sorted_df.head(n_rows),
                     use_container_width=True, hide_index=True, height=230)
    except Exception:
        st.dataframe(df.head(n_rows), use_container_width=True, hide_index=True)


# ============================================================================
# UNDO / REDO
# ============================================================================

def _undo_action():
    try:
        if not st.session_state.get('history'):
            st.toast("Nothing to undo"); return
        if 'redo_stack' not in st.session_state:
            st.session_state.redo_stack = []
        if st.session_state.df is not None:
            st.session_state.redo_stack.append(st.session_state.df.copy())
        st.session_state.df = st.session_state.history.pop()
        st.toast("↩ Undone"); st.rerun()
    except Exception as e:
        st.error(f"Undo failed: {e}")


def _redo_action():
    try:
        if not st.session_state.get('redo_stack'):
            st.toast("Nothing to redo"); return
        if 'history' not in st.session_state:
            st.session_state.history = []
        if st.session_state.df is not None:
            st.session_state.history.append(st.session_state.df.copy())
        st.session_state.df = st.session_state.redo_stack.pop()
        st.toast("↪ Redone"); st.rerun()
    except Exception as e:
        st.error(f"Redo failed: {e}")


# ============================================================================
# CODE EDITORS  (rendered in main content area)
# ============================================================================

def render_python_editor():
    if not st.session_state.get('show_python_editor', False):
        return
    st.markdown("---")
    with st.expander("🐍 Python Code Editor", expanded=True):
        st.markdown("""
            <div style='background:#f8f9fa; padding:1rem; border-radius:8px;
                        border-left:4px solid #667eea; margin-bottom:1rem;'>
                <strong style='color:#667eea;'>💡 Quick Guide</strong><br/>
                Use <code>df</code> for your dataset · <code>pd</code> · <code>np</code><br/>
                e.g. <code>df['profit'] = df['revenue'] - df['cost']</code>
            </div>
        """, unsafe_allow_html=True)
        st.warning("⚠️ Only run trusted code. Changes are permanent after applying.")

        default = "# Create a new column\ndf['new_col'] = df['existing_col'] * 2\n\ndf.head()"
        python_code = st.text_area(
            "Python code", height=250,
            value=st.session_state.get('python_code', default),
            key="python_code_input",
        )
        st.session_state.python_code = python_code

        r1, r2, r3 = st.columns(3)
        with r1:
            if st.button("▶ Run Preview", use_container_width=True, key="run_python"):
                _run_python_code(python_code)
        with r2:
            if st.button("✅ Apply", use_container_width=True,
                         type="primary", key="apply_python"):
                _apply_python_changes()
        with r3:
            if st.button("✕ Close", use_container_width=True, key="cancel_python"):
                st.session_state.show_python_editor = False
                st.session_state.python_preview = None
                st.rerun()

        if st.session_state.get('python_preview') is not None:
            st.markdown("---")
            prev = st.session_state.python_preview
            c1, c2 = st.columns(2)
            c1.metric("Rows", len(prev))
            c2.metric("Columns", len(prev.columns))
            st.dataframe(prev.head(20), use_container_width=True)


def _run_python_code(code: str):
    df = st.session_state.get('df', None)
    if df is None:
        st.error("No dataset loaded"); return
    try:
        lv = {"df": df.copy(), "pd": pd, "np": np}
        exec(code, {}, lv)
        st.session_state.python_preview = lv["df"]
        st.success("✓ Code executed — preview below.")
    except Exception as e:
        st.error(f"❌ {e}")
        st.session_state.python_preview = None


def _apply_python_changes():
    if st.session_state.get('python_preview') is None:
        st.warning("Run preview first"); return
    try:
        from utils_robust import update_df
        update_df(st.session_state.python_preview, "Custom Python code")
        st.session_state.show_python_editor = False
        st.session_state.python_preview = None
        st.session_state.python_code = ""
        st.toast("✅ Applied"); st.rerun()
    except Exception as e:
        st.error(f"Failed: {e}")


def render_sql_editor():
    if not st.session_state.get('show_sql_editor', False):
        return
    st.markdown("---")
    with st.expander("🗄️ SQL Query Editor", expanded=True):
        st.markdown("""
            <div style='background:#f8f9fa; padding:1rem; border-radius:8px;
                        border-left:4px solid #17a2b8; margin-bottom:1rem;'>
                <strong style='color:#17a2b8;'>💡 SQL Guide</strong><br/>
                Table: <code>data</code> · In-memory SQLite<br/>
                e.g. <code>SELECT * FROM data WHERE age > 25</code>
            </div>
        """, unsafe_allow_html=True)

        sql_code = st.text_area(
            "SQL query", height=200,
            value=st.session_state.get('sql_query', "SELECT * FROM data LIMIT 10;"),
            key="sql_query_input",
        )
        st.session_state.sql_query = sql_code

        r1, r2, r3 = st.columns(3)
        with r1:
            if st.button("▶ Run Query", use_container_width=True, key="run_sql_btn"):
                _run_sql_query(sql_code)
        with r2:
            if st.button("✅ Apply", use_container_width=True,
                         type="primary", key="apply_sql_btn"):
                _apply_sql_changes()
        with r3:
            if st.button("✕ Close", use_container_width=True, key="cancel_sql_btn"):
                st.session_state.show_sql_editor = False
                st.session_state.sql_preview = None
                st.rerun()

        if st.session_state.get('sql_preview') is not None:
            st.markdown("---")
            res = st.session_state.sql_preview
            c1, c2 = st.columns(2)
            c1.metric("Rows", len(res))
            c2.metric("Columns", len(res.columns))
            st.dataframe(res, use_container_width=True)


def _run_sql_query(query: str):
    df = st.session_state.get('df', None)
    if df is None:
        st.error("No dataset loaded"); return
    try:
        conn = sqlite3.connect(":memory:")
        df.to_sql("data", conn, index=False, if_exists="replace")
        res = pd.read_sql(query, conn)
        conn.close()
        st.session_state.sql_preview = res
        st.success(f"✓ {len(res):,} rows returned.")
    except Exception as e:
        st.error(f"❌ SQL Error: {e}")
        st.session_state.sql_preview = None


def _apply_sql_changes():
    if st.session_state.get('sql_preview') is None:
        st.warning("Run query first"); return
    try:
        from utils_robust import update_df
        update_df(st.session_state.sql_preview, "Custom SQL query")
        st.session_state.show_sql_editor = False
        st.session_state.sql_preview = None
        st.session_state.sql_query = ""
        st.toast("✅ Applied"); st.rerun()
    except Exception as e:
        st.error(f"Failed: {e}")


# ============================================================================
# SESSION STATE INIT
# ============================================================================

def initialize_sidebar_state():
    defaults = {
        'show_python_editor':   False,
        'show_sql_editor':      False,
        'python_preview':       None,
        'sql_preview':          None,
        'python_code':          "",
        'sql_query':            "",
        'current_page':         _full(*PAGES[0]),
        'history':              [],
        'redo_stack':           [],
        'show_save_dialog':     False,
        'show_load_dialog':     False,
        'current_project_name': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    st.set_page_config(page_title="ANALYTIQAL", page_icon="📊", layout="wide")
    initialize_sidebar_state()
    selected_page = render_modern_sidebar()
    render_python_editor()
    render_sql_editor()
    st.title(selected_page)