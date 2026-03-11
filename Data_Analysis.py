import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from io import BytesIO
import streamlit.components.v1 as components
import warnings
from datetime import datetime
from scipy import stats
from scipy.stats import mstats
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

from dataset_page import render_dataset_manager
from utils_robust import load_data, update_df, get_column_types
from Data_Transformation import render_transformations_page
from Data_cleaning import render_data_cleaning_page
from Data_profiling import render_data_profiling_page
from Feature_engineering import render_feature_engineering_page
from Data_visualisation import render_visualizations_page
from Export_page import render_export_reports_page
from sidebar import initialize_sidebar_state, render_modern_sidebar, render_python_editor, render_sql_editor
from assistant_chat import render_chat

GA_MEASUREMENT_ID = "G-0H7ZRMDBNW"  # replace with your ID

ga_code = f"""
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id={GA_MEASUREMENT_ID}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', '{GA_MEASUREMENT_ID}');
</script>
"""

components.html(ga_code)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ANALYTIQAL",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ─────────────────────────────────────────────────
_defaults = {
    "df":             None,
    "history":        [],
    "redo_stack":     [],
    "processing_log": [],
    "chat_history":   [],
    "assistant":      None,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

warnings.filterwarnings("ignore")
plt.style.use("default")
sns.set_palette("husl")

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main { padding: 2rem; }

h1 { color: #1f77b4; padding-bottom: 1rem; border-bottom: 2px solid #e0e0e0; }
h2 { color: #2c3e50; margin-top: 1.5rem; }
h3 { color: #34495e; }

[data-testid="stMetricValue"] { font-size: 2rem; font-weight: bold; }

.stButton > button {
    border-radius: 8px;
    transition: all 0.3s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.streamlit-expanderHeader {
    background-color: #f8f9fa;
    border-radius: 8px;
    border-left: 4px solid #667eea;
}

.stSuccess { background-color:#d4edda; border-left:4px solid #28a745; border-radius:4px; }
.stWarning { background-color:#fff3cd; border-left:4px solid #ffc107; border-radius:4px; }
.stError   { background-color:#f8d7da; border-left:4px solid #dc3545; border-radius:4px; }
.stInfo    { background-color:#d1ecf1; border-left:4px solid #17a2b8; border-radius:4px; }

.dataframe { border-radius: 8px; overflow: hidden; }

.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 12px 24px;
    background-color: #f8f9fa;
}
.stTabs [aria-selected="true"] { background-color: #667eea; color: white; }

[data-baseweb="tag"] {
    background-color: lightgreen;
    color: #155724 !important;
    border-radius: 6px;
    border: 1px solid #28a745;
}
[data-baseweb="tag"] svg  { fill: #155724 !important; }
[data-baseweb="tag"]:hover { background-color: #c3e6cb !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────
initialize_sidebar_state()
selected_page = render_modern_sidebar()
render_python_editor()
render_sql_editor()

# ── Page routing ───────────────────────────────────────────────────────────

if selected_page == "🏠 Dataset":
    render_dataset_manager()
    render_chat("🏠 Dataset")

elif selected_page == "🔍 Data Profiling":
    render_data_profiling_page()
    render_chat("🔍 Data Profiling")

elif selected_page == "🧹 Data Cleaning":
    render_data_cleaning_page()
    render_chat("🧹 Data Cleaning")

elif selected_page == "🔄 Transformations":
    render_transformations_page()
    render_chat("🔄 Transformations")

elif selected_page == "📈 Feature Engineering":
    render_feature_engineering_page()
    render_chat("📈 Feature Engineering")

elif selected_page == "📊 Visualizations":
    render_visualizations_page()
    render_chat("📊 Visualizations")

elif selected_page == "💾 Export & Reports":
    render_export_reports_page()

    render_chat("💾 Export & Reports")


