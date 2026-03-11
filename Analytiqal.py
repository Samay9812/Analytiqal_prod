"""
ANALYTIQAL — Main Router
Controls top-level navigation: upload → home → descriptive
"""

import streamlit as st

st.set_page_config(
    page_title="Analytiqal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ═══════════════════════════════════════════════════════════════════════════

BRAND_GRADIENT = "135deg, #667eea 0%, #764ba2 100%"
BRAND_PRIMARY  = "#667eea"
D = {
    "c_border":       "#e8eaf0",
    "c_surface":      "#f8f9ff",
    "c_text_primary": "#1a1a2e",
    "c_text_muted":   "#6c757d",
    "t_sm":           "0.83rem",
}

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════════════════

def _init():
    defaults = {
        "view":         "upload",  # "upload" | "home" | "descriptive"
        "df":           None,
        "dataset_meta": {},
        "_go_home":     False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════════════

_sidebar_hidden = st.session_state.view != "descriptive"

st.markdown(
    f"""
    <style>
    {'[data-testid="stSidebar"] { display: none; }' if _sidebar_hidden else ''}
    [data-testid="stSidebarNav"] {{ display: none; }}
    .block-container {{ padding-top: 1.5rem; }}
    .main {{ padding: 2rem; }}
    h1 {{ color: #1f77b4; padding-bottom: 1rem; border-bottom: 2px solid #e0e0e0; }}
    h2 {{ color: #2c3e50; margin-top: 1.5rem; }}
    h3 {{ color: #34495e; }}
    [data-testid="stMetricValue"] {{ font-size: 2rem; font-weight: bold; }}
    .stButton > button {{ border-radius: 8px; transition: all 0.3s; }}
    .stButton > button:hover {{ transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
    .stButton > button[kind="primary"] {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
    .streamlit-expanderHeader {{ background-color: #f8f9fa; border-radius: 8px; border-left: 4px solid #667eea; }}
    .stSuccess {{ background-color: #d4edda; border-left: 4px solid #28a745; border-radius: 4px; }}
    .stWarning {{ background-color: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px; }}
    .stError   {{ background-color: #f8d7da; border-left: 4px solid #dc3545; border-radius: 4px; }}
    .stInfo    {{ background-color: #d1ecf1; border-left: 4px solid #17a2b8; border-radius: 4px; }}
    .dataframe {{ border-radius: 8px; overflow: hidden; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{ border-radius: 8px 8px 0 0; padding: 12px 24px; background-color: #f8f9fa; }}
    .stTabs [aria-selected="true"] {{ background-color: #667eea; color: white; }}
    [data-testid="stPopover"] {{ background-color: #f8f9fa; border-radius: 8px; padding: 1rem; }}
    [data-baseweb="tag"] {{ background-color: lightgreen; color: #155724 !important; border-radius: 6px; border: 1px solid #28a745; }}
    [data-baseweb="tag"] svg {{ fill: #155724 !important; }}
    [data-baseweb="tag"]:hover {{ background-color: #c3e6cb !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# HOME SCREEN
# ═══════════════════════════════════════════════════════════════════════════

def _render_home():
    meta = st.session_state.get("dataset_meta", {})
    name = meta.get("name", "dataset")
    rows = meta.get("rows", 0)
    cols = meta.get("cols", 0)

    st.markdown(
        f"""
        <div style='background:linear-gradient({BRAND_GRADIENT});
             padding:1.4rem 2rem;border-radius:12px;margin-bottom:2rem;'>
          <h1 style='color:white;margin:0;font-size:1.7rem;letter-spacing:-0.02em;
                     border:none;padding:0;'>ANALYTIQAL</h1>
          <p style='color:rgba(255,255,255,0.75);margin:4px 0 0;font-size:0.85rem;'>
            Choose your analysis type
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style='background:{D["c_surface"]};border:1px solid {D["c_border"]};
             border-radius:8px;padding:8px 16px;margin-bottom:2rem;'>
          <span style='font-size:{D["t_sm"]};color:{D["c_text_muted"]};'>
            📁 <b>{name}</b> &nbsp;·&nbsp;
            <b>{rows:,}</b> rows &nbsp;·&nbsp;
            <b>{cols}</b> columns
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown(
            f"""
            <div style='background:white;border:2px solid {BRAND_PRIMARY};
                 border-radius:14px;padding:2rem;height:380px;
                 box-shadow:0 4px 20px rgba(102,126,234,0.12);'>
              <div style='font-size:2.4rem;margin-bottom:0.6rem;'>📊</div>
              <h2 style='margin:0 0 0.4rem;font-size:1.35rem;color:{D["c_text_primary"]};
                         border:none;margin-top:0;'>Descriptive Analysis</h2>
              <p style='color:{D["c_text_muted"]};font-size:{D["t_sm"]};
                        margin:0 0 1.2rem;line-height:1.6;'>
                Explore, clean, and understand your data before modelling.
              </p>
              <ul style='color:{D["c_text_muted"]};font-size:{D["t_sm"]};
                         padding-left:1.2rem;margin:0;line-height:2.2;'>
                <li>Data Profiling</li>
                <li>Data Cleaning</li>
                <li>Transformations</li>
                <li>Feature Engineering</li>
                <li>Visualizations</li>
                <li>Export &amp; Reports</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        if st.button(
            "Enter Descriptive Analysis →",
            type="primary",
            use_container_width=True,
            key="enter_descriptive",
        ):
            st.session_state.view = "descriptive"
            st.rerun()

    with c2:
        st.markdown(
            f"""
            <div style='background:{D["c_surface"]};border:2px solid {D["c_border"]};
                 border-radius:14px;padding:2rem;height:380px;
                 position:relative;overflow:hidden;opacity:0.72;'>
              <div style='position:absolute;top:14px;right:16px;
                   background:#f39c12;color:white;font-size:0.7rem;
                   font-weight:700;padding:3px 10px;border-radius:20px;
                   letter-spacing:0.06em;'>COMING SOON</div>
              <div style='font-size:2.4rem;margin-bottom:0.6rem;filter:grayscale(1);'>🤖</div>
              <h2 style='margin:0 0 0.4rem;font-size:1.35rem;color:{D["c_text_muted"]};
                         border:none;margin-top:0;'>Predictive Analysis</h2>
              <p style='color:{D["c_text_muted"]};font-size:{D["t_sm"]};
                        margin:0 0 1.2rem;line-height:1.6;'>
                Build and evaluate machine learning models on your prepared data.
              </p>
              <ul style='color:#bbb;font-size:{D["t_sm"]};
                         padding-left:1.2rem;margin:0;line-height:2.2;'>
                <li>AutoML</li>
                <li>Model Comparison</li>
                <li>Hyperparameter Tuning</li>
                <li>Model Evaluation</li>
                <li>Predictions &amp; Export</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.button("🔒 Coming Soon", use_container_width=True, disabled=True, key="enter_predictive")

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    _, mid, _ = st.columns([2, 1, 2])
    with mid:
        if st.button("↩ Upload different file", use_container_width=True, key="back_to_upload"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# DESCRIPTIVE — uses your real sidebar.py
# ═══════════════════════════════════════════════════════════════════════════

def _render_descriptive():
    from sidebar import (
        initialize_sidebar_state,
        render_modern_sidebar,
        render_python_editor,
        render_sql_editor,
    )

    initialize_sidebar_state()
    selected_page = render_modern_sidebar()
    render_python_editor()
    render_sql_editor()

    # ← Home button injected at bottom of your real sidebar
    with st.sidebar:
        st.markdown("---")
        if st.button("← Home", use_container_width=True, key="nav_home"):
            st.session_state.view = "home"
            st.rerun()

    # Page routing — exact labels from your sidebar.py
    if selected_page == "🏠 Dataset":
        from dataset_page import render_dataset_manager
        render_dataset_manager()

    elif selected_page == "🔍 Data Profiling":
        from Data_profiling import render_data_profiling_page
        render_data_profiling_page()

    elif selected_page == "🧹 Data Cleaning":
        from Data_cleaning import render_data_cleaning_page
        render_data_cleaning_page()

    elif selected_page == "🔄 Transformations":
        from Data_Transformation import render_transformations_page
        render_transformations_page()

    elif selected_page == "📈 Feature Engineering":
        from Feature_engineering import render_feature_engineering_page
        render_feature_engineering_page()

    elif selected_page == "📊 Visualizations":
        from Data_visualisation import render_visualizations_page
        render_visualizations_page()

    elif selected_page == "💾 Export & Reports":
        from Export_page import render_export_reports_page
        render_export_reports_page()


# ═══════════════════════════════════════════════════════════════════════════
# TOP-LEVEL ROUTER
# ═══════════════════════════════════════════════════════════════════════════

view = st.session_state.view

if view == "upload":
    # Flag check FIRST — before dataset manager gets a chance to rerun
    if st.session_state.get("_go_home"):
        st.session_state._go_home = False
        st.session_state.view = "home"
        st.rerun()

    from dataset_page import render_dataset_manager
    render_dataset_manager()

    if st.session_state.get("df") is not None:
        st.markdown("---")
        if st.button(
            "✅ Done — Go to Analysis →",
            type="primary",
            use_container_width=True,
            key="go_to_analysis",
        ):
            st.session_state._go_home = True
            st.session_state.dataset_meta = {
                "name": st.session_state.get("raw_name", "dataset"),
                "rows": len(st.session_state.df),
                "cols": len(st.session_state.df.columns),
            }
            st.rerun()

elif view == "home":
    _render_home()

elif view == "descriptive":
    _render_descriptive()