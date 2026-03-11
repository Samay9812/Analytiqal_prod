"""
Data Cleaning Page - Professional & Robust
Advanced data cleaning operations with validation, preview, and smart recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mstats
from typing import Optional, List, Tuple, Dict
import plotly.express as px
import plotly.graph_objects as go
from difflib import SequenceMatcher
from itertools import combinations
import re


def render_data_cleaning_page():
    """
    Main data cleaning page with 5 comprehensive cleaning categories.
    All navigation uses st.tabs — no floating radio bars anywhere.
    """

    # ── Page header ────────────────────────────────────────────────────────
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 2rem; border-radius: 12px; margin-bottom: 2rem;'>
            <h1 style='color: white; margin: 0; font-size: 2.5rem;'>🧹 Data Cleaning</h1>
            <p style='color: rgba(255,255,255,0.9); margin-top: 0.5rem; font-size: 1.1rem;'>
                Clean, validate, and prepare your data for analysis
            </p>
        </div>
    """, unsafe_allow_html=True)

    # ── Guard: no data ─────────────────────────────────────────────────────
    df = st.session_state.get('df', None)
    if df is None or df.empty:
        st.info("""
            📂 **No dataset loaded**

            Please load a dataset from the **🏠 Dataset** page first.

            Once loaded, you'll be able to:
            - Handle missing values intelligently
            - Remove duplicate records
            - Detect and treat outliers
            - Apply general cleaning operations
        """)
        return

    # ── Quality dashboard ──────────────────────────────────────────────────
    st.markdown("### 📊 Data Quality Overview")
    quality_metrics = _calculate_quality_metrics(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        missing_pct = quality_metrics['missing_percentage']
        color = "🟢" if missing_pct < 5 else "🟡" if missing_pct < 20 else "🔴"
        st.metric("Missing Values", f"{missing_pct:.1f}%",
                  help=f"{quality_metrics['missing_cells']:,} missing cells")
        st.caption(f"{color} Quality Score")
    with col2:
        dup_pct = quality_metrics['duplicate_percentage']
        color = "🟢" if dup_pct < 1 else "🟡" if dup_pct < 5 else "🔴"
        st.metric("Duplicates", f"{dup_pct:.1f}%",
                  help=f"{quality_metrics['duplicate_count']:,} duplicate rows")
        st.caption(f"{color} Quality Score")
    with col3:
        color = "🟢" if quality_metrics['constant_count'] == 0 else "🟡" if quality_metrics['constant_percentage'] < 10 else "🔴"
        st.metric("Constant Columns", quality_metrics['constant_count'],
                  help="Columns with only one unique value")
        st.caption(f"{color} Quality Score")
    with col4:
        overall_score = quality_metrics['overall_score']
        color = "🟢" if overall_score >= 80 else "🟡" if overall_score >= 60 else "🔴"
        st.metric("Overall Quality", f"{overall_score}/100",
                  help="Composite quality score")
        st.caption(f"{color} Data Health")

    st.markdown("---")

    # ── Main tabs ──────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔧 Missing Values",
        "👥 Duplicates",
        "🔍 Smart Duplicates",
        "📊 Outliers",
        "🧹 General Cleaning",
    ])

    with tab1:
        render_missing_values_section(df)
    with tab2:
        render_duplicates_section(df)
    with tab3:
        render_smart_duplicates_section(df)
    with tab4:
        render_outliers_section(df)
    with tab5:
        render_general_cleaning_section(df)


# ============================================================================
# HELPERS
# ============================================================================

def _calculate_quality_metrics(df: pd.DataFrame) -> Dict:
    total_cells   = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    missing_pct   = (missing_cells / total_cells) * 100

    duplicate_count = df.duplicated().sum()
    duplicate_pct   = (duplicate_count / len(df)) * 100

    constant_cols   = [col for col in df.columns if df[col].nunique() <= 1]
    constant_count  = len(constant_cols)
    constant_pct    = (constant_count / len(df.columns)) * 100

    overall_score = int((
        max(0, 100 - missing_pct * 2) +
        max(0, 100 - duplicate_pct * 5) +
        max(0, 100 - constant_pct * 2)
    ) / 3)

    return {
        'missing_cells':        missing_cells,
        'missing_percentage':   missing_pct,
        'duplicate_count':      duplicate_count,
        'duplicate_percentage': duplicate_pct,
        'constant_count':       constant_count,
        'constant_percentage':  constant_pct,
        'overall_score':        overall_score,
    }


# ============================================================================
# MISSING VALUES
# ============================================================================

def render_missing_values_section(df: pd.DataFrame):
    st.markdown("### 🔧 Missing Value Treatment")
    st.caption("Intelligent handling of missing data with context-aware recommendations")

    with st.expander("ℹ️ Understanding Missing Values"):
        st.markdown("""
        **Treatment Strategies:**
        - **Remove Data:** Best when missing values are random and < 5% of data
        - **Simple Fill:** Quick fixes using statistics (mean, median, mode)
        - **Sequential:** Best for time-series or ordered data
        - **Advanced:** Context-aware filling using related columns
        """)

    missing_summary = df.isnull().sum()
    missing_cols    = missing_summary[missing_summary > 0].index.tolist()

    if not missing_cols:
        st.success("✅ **No missing values found!** Your dataset is complete.")
        return

    # ── Analysis table ─────────────────────────────────────────────────────
    st.markdown("### 📋 Missing Value Analysis")

    missing_df = pd.DataFrame({
        "Column":        missing_cols,
        "Missing Count": [missing_summary[col] for col in missing_cols],
        "Missing %":     [(missing_summary[col] / len(df) * 100) for col in missing_cols],
        "Data Type":     [str(df[col].dtype) for col in missing_cols],
    }).sort_values("Missing %", ascending=False)

    missing_df['Severity'] = missing_df['Missing %'].apply(
        lambda pct: "🟢 Low" if pct < 5 else "🟡 Medium" if pct < 20 else "🔴 High"
    )

    st.dataframe(
        missing_df, use_container_width=True, hide_index=True,
        column_config={
            "Missing %": st.column_config.ProgressColumn(
                "Missing %", format="%.1f%%", min_value=0, max_value=100
            )
        }
    )

    fig = px.bar(
        missing_df, x='Column', y='Missing %',
        title='Missing Values by Column',
        color='Missing %', color_continuous_scale='reds', text='Missing %'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔧 Apply Treatment")

    col1, col2 = st.columns([2, 1])
    with col1:
        target_col = st.selectbox(
            "Select column to treat", missing_cols,
            help="Choose the column you want to handle",
            key="mv_target_col",
        )
    is_numeric  = pd.api.types.is_numeric_dtype(df[target_col])
    missing_pct = (missing_summary[target_col] / len(df)) * 100

    with col2:
        st.markdown("**Smart Recommendation:**")
        recommendation = _get_missing_value_recommendation(df, target_col, is_numeric, missing_pct)
        st.info(f"💡 {recommendation}")

    with st.expander(f"📊 Column Statistics: {target_col}"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Rows", f"{len(df):,}")
            st.metric("Missing",    f"{missing_summary[target_col]:,}")
        with c2:
            st.metric("Missing %", f"{missing_pct:.2f}%")
            st.metric("Valid",     f"{len(df) - missing_summary[target_col]:,}")
        with c3:
            st.metric("Unique Values", f"{df[target_col].nunique():,}")
            if is_numeric:
                st.metric("Mean", f"{df[target_col].mean():.2f}")

    st.markdown("---")

    # ── st.radio → st.tabs for treatment category ──────────────────────────
    rm_tab, fill_tab, seq_tab, adv_tab = st.tabs([
        "🧹 Remove Data",
        "📊 Simple Fill",
        "🔁 Sequential",
        "🧠 Advanced",
    ])

    action     = None
    fill_value = None
    group_col  = None

    with rm_tab:
        st.caption("Best when missing values are random and < 5% of data.")
        action_rm = st.selectbox(
            "Method",
            ["Drop Rows with Missing Values", "Drop This Column Entirely"],
            key="mv_action_rm",
        )
        if st.button("🔍 Preview", key="mv_preview_rm", use_container_width=True):
            st.session_state["mv_pending_action"]     = action_rm
            st.session_state["mv_pending_fill"]       = None
            st.session_state["mv_pending_group"]      = None
            st.session_state["mv_pending_target"]     = target_col

    with fill_tab:
        st.caption("Quick statistical fills. Good for small gaps (< 20%).")
        fill_opts = (
            ["Fill with Median", "Fill with Mean", "Fill with Mode", "Fill with Constant"]
            if is_numeric else
            ["Fill with Mode", "Fill with Most Frequent", "Fill with Constant"]
        )
        action_fill = st.selectbox("Method", fill_opts, key="mv_action_fill")
        fill_value_in = None
        if "Constant" in action_fill:
            if is_numeric:
                fill_value_in = st.number_input("Fill value", value=0.0, key="mv_fill_num")
            else:
                fill_value_in = st.text_input("Fill value", value="Unknown", key="mv_fill_txt")
        if st.button("🔍 Preview", key="mv_preview_fill", use_container_width=True):
            st.session_state["mv_pending_action"] = action_fill
            st.session_state["mv_pending_fill"]   = fill_value_in
            st.session_state["mv_pending_group"]  = None
            st.session_state["mv_pending_target"] = target_col

    with seq_tab:
        st.caption("Best for time-series or ordered data where adjacent values are related.")
        seq_opts = (
            ["Forward Fill (Use Previous Value)", "Backward Fill (Use Next Value)", "Interpolate (Linear)"]
            if is_numeric else
            ["Forward Fill (Use Previous Value)", "Backward Fill (Use Next Value)"]
        )
        action_seq = st.selectbox("Method", seq_opts, key="mv_action_seq")
        if st.button("🔍 Preview", key="mv_preview_seq", use_container_width=True):
            st.session_state["mv_pending_action"] = action_seq
            st.session_state["mv_pending_fill"]   = None
            st.session_state["mv_pending_group"]  = None
            st.session_state["mv_pending_target"] = target_col

    with adv_tab:
        st.caption("Context-aware fills using groupings in your data.")
        adv_opts = (
            ["Fill with Grouped Mean", "Fill with Grouped Median", "Fill with Grouped Mode"]
            if is_numeric else
            ["Fill with Grouped Mode"]
        )
        action_adv = st.selectbox("Method", adv_opts, key="mv_action_adv")
        possible_group_cols = [c for c in df.columns if c != target_col and df[c].nunique() < 50]
        if not possible_group_cols:
            st.warning("⚠️ No suitable grouping columns found (need < 50 unique values)")
        else:
            group_col_in = st.selectbox(
                "Group by", possible_group_cols,
                help="Fill missing values based on the statistic within each group",
                key="mv_group_col",
            )
            with st.expander("👁️ Preview Groups"):
                grp = df.groupby(group_col_in)[target_col].agg(
                    ['count', 'mean' if is_numeric else 'first']
                )
                st.dataframe(grp, use_container_width=True)

            if st.button("🔍 Preview", key="mv_preview_adv", use_container_width=True):
                st.session_state["mv_pending_action"] = action_adv
                st.session_state["mv_pending_fill"]   = None
                st.session_state["mv_pending_group"]  = group_col_in
                st.session_state["mv_pending_target"] = target_col

    # ── Shared preview/apply block ─────────────────────────────────────────
    if "mv_pending_action" in st.session_state and \
       st.session_state.get("mv_pending_target") == target_col:

        pending_action = st.session_state["mv_pending_action"]
        pending_fill   = st.session_state["mv_pending_fill"]
        pending_group  = st.session_state["mv_pending_group"]

        preview_df = _preview_missing_value_treatment(
            df.copy(), target_col, pending_action, pending_fill, pending_group
        )

        if preview_df is not None:
            st.markdown("---")
            st.markdown(f"#### 👁️ Preview — `{pending_action}`")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Original Missing", f"{missing_summary[target_col]:,}")
            with c2:
                new_missing = preview_df[target_col].isnull().sum() if target_col in preview_df.columns else 0
                st.metric("After Treatment", f"{new_missing:,}",
                          delta=f"{new_missing - missing_summary[target_col]:,}",
                          delta_color="inverse")
            with c3:
                if "Drop" in pending_action:
                    st.metric("Rows Affected", f"{len(df) - len(preview_df):,}")
                else:
                    st.metric("Values Filled", f"{missing_summary[target_col] - new_missing:,}")

            if target_col in preview_df.columns:
                cmp = pd.DataFrame({
                    'Before': df[target_col].head(10),
                    'After':  preview_df[target_col].head(10),
                })
                st.dataframe(cmp, use_container_width=True)

            st.markdown("---")
            ap1, ap2 = st.columns([3, 1])
            with ap1:
                if st.button("✅ Apply Treatment", key="mv_apply",
                             type="primary", use_container_width=True):
                    from utils_robust import update_df
                    update_df(preview_df, f"Missing values: {pending_action} on {target_col}")
                    for k in ("mv_pending_action", "mv_pending_fill",
                              "mv_pending_group", "mv_pending_target"):
                        st.session_state.pop(k, None)
                    st.success("✅ Treatment applied successfully!")
                    st.rerun()
            with ap2:
                if st.button("❌ Cancel", key="mv_cancel", use_container_width=True):
                    for k in ("mv_pending_action", "mv_pending_fill",
                              "mv_pending_group", "mv_pending_target"):
                        st.session_state.pop(k, None)
                    st.rerun()


def _get_missing_value_recommendation(df, col, is_numeric, missing_pct):
    if missing_pct > 50:
        return "🔴 Consider dropping this column (>50% missing)"
    elif missing_pct > 20:
        return "🟡 Use grouped imputation or drop rows"
    elif missing_pct < 5:
        return "🟢 Safe to drop rows or use simple fill"
    else:
        if is_numeric:
            return "📊 Use Median (skewed)" if abs(df[col].skew()) > 1 else "📊 Use Mean or Median"
        return "📊 Use Mode or Most Frequent"


def _preview_missing_value_treatment(df, target_col, action, fill_value=None, group_col=None):
    try:
        preview_df = df.copy()

        if action == "Drop Rows with Missing Values":
            preview_df = preview_df.dropna(subset=[target_col])
        elif action == "Drop This Column Entirely":
            preview_df = preview_df.drop(columns=[target_col])
        elif "Mean" in action:
            preview_df[target_col] = preview_df[target_col].fillna(preview_df[target_col].mean())
        elif "Median" in action:
            preview_df[target_col] = preview_df[target_col].fillna(preview_df[target_col].median())
        elif "Mode" in action or "Most Frequent" in action:
            mode_val = preview_df[target_col].mode()
            if len(mode_val) > 0:
                preview_df[target_col] = preview_df[target_col].fillna(mode_val[0])
        elif "Forward Fill" in action:
            preview_df[target_col] = preview_df[target_col].ffill()
        elif "Backward Fill" in action:
            preview_df[target_col] = preview_df[target_col].bfill()
        elif "Interpolate" in action:
            preview_df[target_col] = preview_df[target_col].interpolate()
        elif "Constant" in action:
            preview_df[target_col] = preview_df[target_col].fillna(fill_value)
        elif action == "Fill with Grouped Mean":
            preview_df[target_col] = preview_df[target_col].fillna(
                preview_df.groupby(group_col)[target_col].transform("mean")
            )
        elif action == "Fill with Grouped Median":
            preview_df[target_col] = preview_df[target_col].fillna(
                preview_df.groupby(group_col)[target_col].transform("median")
            )
        elif action == "Fill with Grouped Mode":
            def mode_func(x):
                m = x.mode()
                return m[0] if not m.empty else np.nan
            preview_df[target_col] = preview_df[target_col].fillna(
                preview_df.groupby(group_col)[target_col].transform(mode_func)
            )
        elif "KNN" in action:
            st.info("🚧 KNN Imputation coming soon!")
            return None

        return preview_df

    except Exception as e:
        st.error(f"Preview error: {str(e)}")
        return None


# ============================================================================
# DUPLICATES
# ============================================================================

def render_duplicates_section(df: pd.DataFrame):
    st.markdown("### 👥 Duplicate Record Treatment")
    st.caption("Identify and remove duplicate rows to ensure data quality")

    with st.expander("ℹ️ Understanding Duplicates"):
        st.markdown("""
        **Detection Methods:**
        - **All Columns:** Rows must match in every column
        - **Subset of Columns:** Rows match only in selected columns

        **Treatment Methods:**
        - **Keep First / Keep Last:** Retains one copy, removes the rest
        - **Remove All:** Removes every copy, keeps only truly unique rows
        - **Mark Only:** Flags duplicates without removing them
        """)

    # ── st.radio → st.tabs for detection method ────────────────────────────
    all_tab, subset_tab = st.tabs(["📋 All Columns", "🎯 Subset of Columns"])

    subset_cols = None

    with all_tab:
        st.caption("A row is a duplicate only if every column matches another row.")

    with subset_tab:
        st.caption("A row is a duplicate if the selected columns match — other columns may differ.")
        subset_cols_input = st.multiselect(
            "Select columns to check",
            df.columns.tolist(),
            help="Rows are considered duplicates if they match in these columns",
            key="dup_subset_cols",
        )
        if subset_cols_input:
            subset_cols = subset_cols_input

    # Resolve active detection scope
    _use_subset = bool(subset_cols)
    if _use_subset and not subset_cols:
        st.warning("⚠️ Please select at least one column in the Subset tab")
        return

    duplicates = df.duplicated(subset=subset_cols, keep=False) if _use_subset else df.duplicated(keep=False)
    dup_count  = duplicates.sum()
    dup_pct    = (dup_count / len(df)) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows",     f"{len(df):,}")
    c2.metric("Duplicate Rows", f"{dup_count:,}")
    c3.metric("Duplicate %",    f"{dup_pct:.2f}%")
    c4.metric("Unique Rows",    f"{len(df) - dup_count:,}")

    if dup_count == 0:
        st.success("✅ **No duplicates found!** Your dataset has unique records.")
        return

    severity = "🔴 High" if dup_pct > 10 else "🟡 Medium" if dup_pct > 5 else "🟢 Low"
    st.warning(f"⚠️ Found {dup_count:,} duplicate rows ({dup_pct:.2f}%) — Severity: {severity}")

    fig = go.Figure(data=[go.Pie(
        labels=['Unique', 'Duplicates'],
        values=[len(df) - dup_count, dup_count],
        hole=0.4, marker_colors=['#4CAF50', '#FF5252']
    )])
    fig.update_layout(title="Duplicate Distribution", height=280, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    if st.button("🔍 Show Duplicate Records", key="dup_show_records", use_container_width=True):
        if subset_cols:
            dup_records = df[df.duplicated(subset=subset_cols, keep=False)].sort_values(by=subset_cols)
        else:
            dup_records = df[df.duplicated(keep=False)].sort_values(by=df.columns.tolist())
        st.info(f"Showing {len(dup_records):,} duplicate records")
        st.dataframe(dup_records, use_container_width=True, height=350)

    st.markdown("---")
    st.markdown("### 🔧 Apply Treatment")

    dup_method = st.selectbox(
        "Treatment method",
        ["Keep First Occurrence", "Keep Last Occurrence",
         "Remove All Duplicates", "Mark as Duplicate (Add Column)"],
        key="dup_method",
    )

    # Impact preview inline
    if "Keep First" in dup_method:
        ud = df[df.duplicated(subset=subset_cols, keep=False)].drop_duplicates(subset=subset_cols) if subset_cols \
             else df[df.duplicated(keep=False)].drop_duplicates()
        st.caption(f"→ Removes {dup_count - len(ud):,} rows, keeps {len(ud):,} unique records")
    elif "Keep Last" in dup_method:
        ud = df[df.duplicated(subset=subset_cols, keep=False)].drop_duplicates(subset=subset_cols, keep='last') if subset_cols \
             else df[df.duplicated(keep=False)].drop_duplicates(keep='last')
        st.caption(f"→ Removes {dup_count - len(ud):,} rows, keeps {len(ud):,} unique records")
    elif "Remove All" in dup_method:
        st.caption(f"→ Removes all {dup_count:,} duplicate rows")
    else:
        st.caption("→ Adds an `is_duplicate` boolean column, no rows removed")

    if st.button("👁️ Generate Preview", key="dup_gen_preview", use_container_width=True):
        try:
            preview_df = df.copy()
            if dup_method == "Keep First Occurrence":
                preview_df = preview_df.drop_duplicates(subset=subset_cols, keep='first') if subset_cols \
                             else preview_df.drop_duplicates(keep='first')
            elif dup_method == "Keep Last Occurrence":
                preview_df = preview_df.drop_duplicates(subset=subset_cols, keep='last') if subset_cols \
                             else preview_df.drop_duplicates(keep='last')
            elif dup_method == "Remove All Duplicates":
                preview_df = preview_df.drop_duplicates(subset=subset_cols, keep=False) if subset_cols \
                             else preview_df.drop_duplicates(keep=False)
            else:
                preview_df['is_duplicate'] = preview_df.duplicated(subset=subset_cols, keep=False) if subset_cols \
                                             else preview_df.duplicated(keep=False)
            st.session_state['dup_preview'] = preview_df
            st.session_state['dup_method']  = dup_method
        except Exception as e:
            st.error(f"Preview error: {str(e)}")

    if 'dup_preview' in st.session_state:
        preview_df = st.session_state['dup_preview']
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Original Rows",   f"{len(df):,}")
        c2.metric("After Treatment", f"{len(preview_df):,}")
        removed = len(df) - len(preview_df)
        c3.metric("Rows Removed", f"{removed:,}",
                  delta=f"-{removed:,}", delta_color="inverse")
        st.dataframe(preview_df.head(20), use_container_width=True)

        ap1, ap2 = st.columns([3, 1])
        with ap1:
            if st.button("✅ Apply Treatment", key="dup_apply",
                         type="primary", use_container_width=True):
                from utils_robust import update_df
                update_df(preview_df, f"Duplicates: {st.session_state['dup_method']}")
                st.session_state.pop('dup_preview', None)
                st.session_state.pop('dup_method', None)
                st.success("✅ Duplicate treatment applied!")
                st.rerun()
        with ap2:
            if st.button("❌ Cancel", key="dup_cancel", use_container_width=True):
                st.session_state.pop('dup_preview', None)
                st.session_state.pop('dup_method', None)
                st.rerun()


# ============================================================================
# OUTLIERS
# ============================================================================

def render_outliers_section(df: pd.DataFrame):
    st.markdown("### 📊 Outlier Detection & Treatment")
    st.caption("Identify and handle extreme values that may affect analysis")

    with st.expander("ℹ️ Understanding Outliers"):
        st.markdown("""
        **Detection Methods:**
        - **IQR (Recommended):** Uses quartiles; robust to extreme values
        - **Z-Score:** Based on standard deviations; sensitive to outliers
        - **Modified Z-Score:** Uses median; more robust than Z-score
        - **Custom Bounds:** Manual threshold setting

        **Treatment Options:**
        - **Remove:** Delete outlier rows
        - **Cap:** Limit values to boundaries
        - **Winsorize:** Replace extreme values with percentile values
        """)

    from utils_robust import get_column_types
    numeric_cols = get_column_types(df)['numeric']

    if not numeric_cols:
        st.warning("⚠️ No numeric columns available for outlier detection")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        target_col = st.selectbox("Select numeric column", numeric_cols, key="outlier_col")
    with col2:
        st.markdown("**Column Stats:**")
        st.metric("Min", f"{df[target_col].min():.2f}")
        st.metric("Max", f"{df[target_col].max():.2f}")

    # ── Distribution chart ─────────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df[target_col], name='Distribution',
        nbinsx=50, marker_color='lightblue', opacity=0.7
    ))
    fig.update_layout(title=f'Distribution of {target_col}', height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean",     f"{df[target_col].mean():.2f}")
    c2.metric("Median",   f"{df[target_col].median():.2f}")
    c3.metric("Std Dev",  f"{df[target_col].std():.2f}")
    c4.metric("Skewness", f"{df[target_col].skew():.2f}")

    st.markdown("---")
    st.markdown("### 🔍 Detection Method")

    # ── st.radio → st.tabs for detection method ────────────────────────────
    iqr_tab, zscore_tab, modz_tab, custom_tab = st.tabs([
        "📐 IQR  (Recommended)",
        "📊 Z-Score",
        "🔬 Modified Z-Score",
        "✏️ Custom Bounds",
    ])

    outlier_mask = None
    lower_bound  = None
    upper_bound  = None

    with iqr_tab:
        st.caption("Uses quartile distances. Robust — not skewed by the outliers themselves.")
        iqr_multiplier = st.slider(
            "IQR Multiplier", 1.0, 3.0, 1.5, 0.1,
            help="1.5 = standard. Increase to flag fewer points.",
            key="outlier_iqr_mult",
        )
        Q1 = df[target_col].quantile(0.25)
        Q3 = df[target_col].quantile(0.75)
        IQR = Q3 - Q1
        _lb_iqr = Q1 - iqr_multiplier * IQR
        _ub_iqr = Q3 + iqr_multiplier * IQR
        _mask_iqr = (df[target_col] < _lb_iqr) | (df[target_col] > _ub_iqr)
        st.info(f"Q1: {Q1:.2f} · Q3: {Q3:.2f} · IQR: {IQR:.2f} · "
                f"Lower: **{_lb_iqr:.2f}** · Upper: **{_ub_iqr:.2f}** · "
                f"Flagged: **{int(_mask_iqr.sum()):,}** rows")
        if st.button("Use IQR for detection →", key="outlier_use_iqr",
                     type="primary", use_container_width=True):
            st.session_state["outlier_active_method"] = "iqr"
            st.rerun()

    with zscore_tab:
        st.caption("Flags values beyond N standard deviations from the mean.")
        z_threshold = st.slider(
            "Z-Score Threshold", 2.0, 4.0, 3.0, 0.1,
            help="3.0 = standard. Decrease to be stricter.",
            key="outlier_z_thresh",
        )
        mean     = df[target_col].mean()
        std      = df[target_col].std()
        _lb_z    = mean - z_threshold * std
        _ub_z    = mean + z_threshold * std
        _mask_z  = np.abs((df[target_col] - mean) / std) > z_threshold
        st.info(f"Mean: {mean:.2f} · Std: {std:.2f} · "
                f"Lower: **{_lb_z:.2f}** · Upper: **{_ub_z:.2f}** · "
                f"Flagged: **{int(_mask_z.sum()):,}** rows")
        if st.button("Use Z-Score for detection →", key="outlier_use_z",
                     type="primary", use_container_width=True):
            st.session_state["outlier_active_method"] = "zscore"
            st.rerun()

    with modz_tab:
        st.caption("Uses median absolute deviation (MAD). Better than Z-score for skewed data.")
        mod_threshold = st.slider(
            "Modified Z-Score Threshold", 2.0, 4.0, 3.5, 0.1,
            key="outlier_modz_thresh",
        )
        median     = df[target_col].median()
        mad        = np.median(np.abs(df[target_col] - median))
        _lb_mz     = median - mod_threshold * mad / 0.6745
        _ub_mz     = median + mod_threshold * mad / 0.6745
        mod_z      = 0.6745 * (df[target_col] - median) / mad if mad != 0 else pd.Series(0, index=df.index)
        _mask_mz   = np.abs(mod_z) > mod_threshold
        st.info(f"Median: {median:.2f} · MAD: {mad:.2f} · "
                f"Lower: **{_lb_mz:.2f}** · Upper: **{_ub_mz:.2f}** · "
                f"Flagged: **{int(_mask_mz.sum()):,}** rows")
        if st.button("Use Modified Z-Score →", key="outlier_use_mz",
                     type="primary", use_container_width=True):
            st.session_state["outlier_active_method"] = "modz"
            st.rerun()

    with custom_tab:
        st.caption("Manually set the bounds. Useful when you know the valid data range.")
        cc1, cc2 = st.columns(2)
        with cc1:
            _lb_custom = st.number_input("Lower Bound",
                                         value=float(df[target_col].min()),
                                         key="outlier_lower")
        with cc2:
            _ub_custom = st.number_input("Upper Bound",
                                         value=float(df[target_col].max()),
                                         key="outlier_upper")
        _mask_custom = (df[target_col] < _lb_custom) | (df[target_col] > _ub_custom)
        st.caption(f"→ Flagged: **{int(_mask_custom.sum()):,}** rows")
        if st.button("Use Custom Bounds →", key="outlier_use_custom",
                     type="primary", use_container_width=True):
            st.session_state["outlier_active_method"] = "custom"
            st.rerun()

    # Resolve active detection method
    active_method = st.session_state.get("outlier_active_method", "iqr")
    method_map = {
        "iqr":    (_mask_iqr,    _lb_iqr,    _ub_iqr,    "IQR"),
        "zscore": (_mask_z,      _lb_z,      _ub_z,      "Z-Score"),
        "modz":   (_mask_mz,     _lb_mz,     _ub_mz,     "Modified Z-Score"),
        "custom": (_mask_custom, _lb_custom, _ub_custom, "Custom Bounds"),
    }
    outlier_mask, lower_bound, upper_bound, active_label = method_map[active_method]

    st.markdown("---")
    outlier_count = int(outlier_mask.sum())
    outlier_pct   = (outlier_count / len(df)) * 100

    st.markdown(f"### 📊 Detection Results — using **{active_label}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows",        f"{len(df):,}")
    c2.metric("Outliers Detected", f"{outlier_count:,}")
    c3.metric("Outlier %",         f"{outlier_pct:.2f}%")
    c4.metric("Normal Values",     f"{len(df) - outlier_count:,}")

    if outlier_count > 0:
        fig2 = px.box(df, y=target_col, title=f'Outlier Visualization: {target_col}',
                      color_discrete_sequence=['coral'])
        outlier_vals = df[outlier_mask][target_col]
        fig2.add_trace(go.Scatter(
            y=outlier_vals, mode='markers', name='Outliers',
            marker=dict(color='red', size=8)
        ))
        fig2.add_hline(y=lower_bound, line_dash="dash", line_color="green",
                       annotation_text="Lower Bound")
        fig2.add_hline(y=upper_bound, line_dash="dash", line_color="green",
                       annotation_text="Upper Bound")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔧 Treatment Method")

    # ── st.radio → st.tabs for treatment ──────────────────────────────────
    remove_tab, cap_tab, winsor_tab = st.tabs([
        "🗑️ Remove Outliers",
        "📌 Cap to Bounds",
        "🔄 Winsorize",
    ])

    treatment      = None
    lower_limit    = 0.05
    upper_limit    = 0.05

    with remove_tab:
        st.caption("Delete rows that contain outlier values. Best when outliers are data errors.")
        st.caption(f"→ Will remove **{outlier_count:,}** rows ({outlier_pct:.1f}% of data)")
        if st.button("Use: Remove Outliers", key="outlier_treat_rm",
                     type="primary", use_container_width=True):
            st.session_state["outlier_active_treatment"] = "Remove Outliers"
            st.rerun()

    with cap_tab:
        st.caption("Replace outlier values with the boundary values. Rows are kept.")
        st.caption(f"→ Caps values to [{lower_bound:.2f}, {upper_bound:.2f}]")
        if st.button("Use: Cap to Bounds", key="outlier_treat_cap",
                     type="primary", use_container_width=True):
            st.session_state["outlier_active_treatment"] = "Cap to Bounds"
            st.rerun()

    with winsor_tab:
        st.caption("Replace extreme tails with percentile values. A softer version of capping.")
        wc1, wc2 = st.columns(2)
        with wc1:
            lower_limit = st.slider("Lower limit", 0.01, 0.1, 0.05, 0.01,
                                    key="outlier_wins_lower")
        with wc2:
            upper_limit = st.slider("Upper limit", 0.01, 0.1, 0.05, 0.01,
                                    key="outlier_wins_upper")
        if st.button("Use: Winsorize", key="outlier_treat_wins",
                     type="primary", use_container_width=True):
            st.session_state["outlier_active_treatment"] = "Winsorize"
            st.rerun()

    treatment = st.session_state.get("outlier_active_treatment", "Remove Outliers")
    st.caption(f"Active treatment: **{treatment}**")

    if st.button("👁️ Generate Preview", key="outlier_gen_preview", use_container_width=True):
        try:
            preview_df = df.copy()
            if treatment == "Remove Outliers":
                preview_df = preview_df[~outlier_mask]
            elif treatment == "Cap to Bounds":
                preview_df[target_col] = np.clip(preview_df[target_col], lower_bound, upper_bound)
            elif treatment == "Winsorize":
                preview_df[target_col] = mstats.winsorize(
                    preview_df[target_col], limits=[lower_limit, upper_limit]
                )
            st.session_state['outlier_preview']   = preview_df
            st.session_state['outlier_treatment']  = treatment
            st.session_state['outlier_target']     = target_col
        except Exception as e:
            st.error(f"Treatment error: {str(e)}")

    if 'outlier_preview' in st.session_state:
        preview_df = st.session_state['outlier_preview']
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Original Mean", f"{df[target_col].mean():.2f}")
        c2.metric("After Mean",    f"{preview_df[target_col].mean():.2f}")
        c3.metric("Original Std",  f"{df[target_col].std():.2f}")
        c4.metric("After Std",     f"{preview_df[target_col].std():.2f}")

        fig3 = go.Figure()
        fig3.add_trace(go.Box(y=df[target_col],         name='Before', marker_color='lightblue'))
        fig3.add_trace(go.Box(y=preview_df[target_col], name='After',  marker_color='lightgreen'))
        fig3.update_layout(title="Before vs After Treatment", height=380)
        st.plotly_chart(fig3, use_container_width=True)

        ap1, ap2 = st.columns([3, 1])
        with ap1:
            if st.button("✅ Apply Treatment", key="outlier_apply",
                         type="primary", use_container_width=True):
                from utils_robust import update_df
                update_df(preview_df, f"Outliers: {treatment} on {target_col}")
                for k in ('outlier_preview', 'outlier_treatment', 'outlier_target'):
                    st.session_state.pop(k, None)
                st.success("✅ Outlier treatment applied!")
                st.rerun()
        with ap2:
            if st.button("❌ Cancel", key="outlier_cancel", use_container_width=True):
                for k in ('outlier_preview', 'outlier_treatment', 'outlier_target'):
                    st.session_state.pop(k, None)
                st.rerun()


# ============================================================================
# GENERAL CLEANING
# ============================================================================

def render_general_cleaning_section(df: pd.DataFrame):
    st.markdown("### 🧹 General Cleaning Operations")
    st.caption("Additional cleaning operations to improve data quality")

    with st.expander("ℹ️ What's here"):
        st.markdown("""
        - **Constant Columns:** Remove columns that carry no information
        - **High Missing:** Remove columns with excessive missing values
        - **Standardize Text:** Consistent casing and whitespace
        - **Special Characters:** Strip unwanted characters from text
        - **Trim Whitespace:** Strip leading/trailing spaces from all text columns

        ℹ️ Log Transform → Transformation stage · High Correlation → Feature Engineering · Type Optimization → Dataset Prep
        """)

    # ── st.selectbox → st.tabs ─────────────────────────────────────────────
    const_tab, miss_tab, std_tab, spec_tab, trim_tab = st.tabs([
        "🔇 Constant Columns",
        "🕳️ High Missing %",
        "🔤 Standardize Text",
        "🚿 Special Characters",
        "✂️ Trim Whitespace",
    ])

    # ── Constant Columns ───────────────────────────────────────────────────
    with const_tab:
        st.markdown("**Remove Constant Columns**")
        st.caption("Columns where all values are the same provide no information for analysis.")

        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]

        if not constant_cols:
            st.success("✅ No constant columns found!")
        else:
            st.warning(f"⚠️ Found {len(constant_cols)} constant columns")
            st.dataframe(pd.DataFrame({
                'Column':       constant_cols,
                'Unique Value': [df[col].unique()[0] if len(df[col].unique()) > 0 else None
                                 for col in constant_cols],
                'Data Type':    [str(df[col].dtype) for col in constant_cols],
            }), use_container_width=True, hide_index=True)
            st.markdown("---")
            if st.button("🧹 Remove Constant Columns", key="gc_remove_constant",
                         type="primary", use_container_width=True):
                from utils_robust import update_df
                update_df(df.drop(columns=constant_cols),
                          f"Removed {len(constant_cols)} constant columns")
                st.success(f"✅ Removed {len(constant_cols)} constant columns")
                st.rerun()

    # ── High Missing % ─────────────────────────────────────────────────────
    with miss_tab:
        st.markdown("**Remove Columns by Missing Percentage**")
        st.caption("Remove columns with excessive missing values.")

        threshold = st.slider(
            "Missing % Threshold", 10, 90, 50, 5,
            help="Columns with missing % above this will be removed",
            key="gc_missing_thresh",
        )

        missing_pct  = (df.isnull().sum() / len(df)) * 100
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()

        fig = px.bar(
            x=missing_pct.index, y=missing_pct.values,
            title='Missing Percentage by Column',
            labels={'x': 'Column', 'y': 'Missing %'},
            color=missing_pct.values, color_continuous_scale='Reds',
        )
        fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                      annotation_text=f"Threshold: {threshold}%")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

        if not cols_to_drop:
            st.success(f"✅ No columns with > {threshold}% missing values")
        else:
            st.warning(f"⚠️ {len(cols_to_drop)} columns above threshold")
            st.dataframe(pd.DataFrame({
                'Column':        cols_to_drop,
                'Missing %':     [missing_pct[c] for c in cols_to_drop],
                'Missing Count': [df[c].isnull().sum() for c in cols_to_drop],
            }).sort_values('Missing %', ascending=False),
            use_container_width=True, hide_index=True)
            st.markdown("---")
            if st.button("🧹 Remove Columns", key="gc_remove_missing_cols",
                         type="primary", use_container_width=True):
                from utils_robust import update_df
                update_df(df.drop(columns=cols_to_drop),
                          f"Removed {len(cols_to_drop)} columns with >{threshold}% missing")
                st.success(f"✅ Removed {len(cols_to_drop)} columns")
                st.rerun()

    # ── Standardize Text ───────────────────────────────────────────────────
    with std_tab:
        st.markdown("**Standardize Text Columns**")
        st.caption("Make text formatting consistent across the column.")

        from utils_robust import get_column_types
        cat_cols = get_column_types(df)['categorical']

        if not cat_cols:
            st.warning("⚠️ No text columns available")
        else:
            c1, c2 = st.columns([2, 1])
            with c1:
                target_col = st.selectbox("Select column", cat_cols, key="gc_std_col")
            with c2:
                operation = st.selectbox("Operation",
                    ["Lowercase", "Uppercase", "Title Case", "Strip Whitespace"],
                    key="gc_std_op")

            preview_series = df[target_col].copy()
            if operation == "Lowercase":         preview_series = preview_series.str.lower()
            elif operation == "Uppercase":        preview_series = preview_series.str.upper()
            elif operation == "Title Case":       preview_series = preview_series.str.title()
            elif operation == "Strip Whitespace": preview_series = preview_series.str.strip()

            st.dataframe(pd.DataFrame({
                'Before': df[target_col].head(10),
                'After':  preview_series.head(10),
            }), use_container_width=True)
            st.markdown("---")
            if st.button("🧹 Apply Standardization", key="gc_apply_std",
                         type="primary", use_container_width=True):
                from utils_robust import update_df
                cleaned_df = df.copy()
                cleaned_df[target_col] = preview_series
                update_df(cleaned_df, f"Standardized {target_col}: {operation}")
                st.success(f"✅ Applied {operation}")
                st.rerun()

    # ── Special Characters ─────────────────────────────────────────────────
    with spec_tab:
        st.markdown("**Remove Special Characters**")
        st.caption("Clean text by removing unwanted characters.")

        from utils_robust import get_column_types
        cat_cols = get_column_types(df)['categorical']

        if not cat_cols:
            st.warning("⚠️ No text columns available")
        else:
            target_col = st.selectbox("Select column", cat_cols, key="gc_spec_col")

            # ── st.radio → st.tabs for char type ──────────────────────────
            punct_tab, digits_tab, specch_tab, custom_tab2 = st.tabs([
                "Punctuation", "Digits", "Special Characters", "Custom Pattern"
            ])

            pattern    = None
            char_type  = st.session_state.get("gc_spec_type_val", "Punctuation")

            with punct_tab:
                st.caption("Removes all punctuation marks (. , ! ? etc)")
                if st.button("Use: Punctuation", key="gc_spec_punct",
                             type="primary", use_container_width=True):
                    st.session_state["gc_spec_type_val"] = "Punctuation"
                    st.rerun()
            with digits_tab:
                st.caption("Removes all numeric digits (0–9)")
                if st.button("Use: Digits", key="gc_spec_digits",
                             type="primary", use_container_width=True):
                    st.session_state["gc_spec_type_val"] = "Digits"
                    st.rerun()
            with specch_tab:
                st.caption("Removes everything except letters, numbers, and spaces")
                if st.button("Use: Special Characters", key="gc_spec_special",
                             type="primary", use_container_width=True):
                    st.session_state["gc_spec_type_val"] = "Special Characters"
                    st.rerun()
            with custom_tab2:
                st.caption("Define your own regex pattern to remove.")
                pattern = st.text_input("Regex pattern", value=r"[^a-zA-Z0-9\s]",
                                        key="gc_spec_pattern")
                if st.button("Use: Custom Pattern", key="gc_spec_custom",
                             type="primary", use_container_width=True):
                    st.session_state["gc_spec_type_val"] = "Custom Pattern"
                    st.rerun()

            char_type = st.session_state.get("gc_spec_type_val", "Punctuation")
            st.caption(f"Active: **{char_type}**")

            preview_series = df[target_col].copy().astype(str)
            try:
                if char_type == "Punctuation":
                    preview_series = preview_series.str.replace(r'[^\w\s]', '', regex=True)
                elif char_type == "Digits":
                    preview_series = preview_series.str.replace(r'\d+', '', regex=True)
                elif char_type == "Special Characters":
                    preview_series = preview_series.str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
                elif char_type == "Custom Pattern" and pattern:
                    preview_series = preview_series.str.replace(pattern, '', regex=True)
            except Exception:
                st.error("Invalid regex pattern")
            else:
                st.dataframe(pd.DataFrame({
                    'Before': df[target_col].head(10).astype(str),
                    'After':  preview_series.head(10),
                }), use_container_width=True)
                st.markdown("---")
                if st.button("🧹 Apply Cleaning", key="gc_apply_special",
                             type="primary", use_container_width=True):
                    from utils_robust import update_df
                    cleaned_df = df.copy()
                    cleaned_df[target_col] = preview_series
                    update_df(cleaned_df, f"Removed {char_type} from {target_col}")
                    st.success("✅ Cleaning applied")
                    st.rerun()

    # ── Trim Whitespace ────────────────────────────────────────────────────
    with trim_tab:
        st.markdown("**Trim Whitespace**")
        st.caption("Remove leading and trailing spaces from all text columns.")

        from utils_robust import get_column_types
        cat_cols = get_column_types(df)['categorical']

        if not cat_cols:
            st.warning("⚠️ No text columns available")
        else:
            st.info(f"📋 Will trim whitespace from **{len(cat_cols)}** text columns")

            sample_col = cat_cols[0]
            st.markdown(f"**Sample preview (column: `{sample_col}`):**")
            st.dataframe(pd.DataFrame({
                'Before': [f"'{v}'" for v in df[sample_col].head(5).astype(str)],
                'After':  [f"'{v.strip()}'" for v in df[sample_col].head(5).astype(str)],
            }), use_container_width=True, hide_index=True)
            st.markdown("---")
            if st.button("🧹 Trim All Text Columns", key="gc_trim_all",
                         type="primary", use_container_width=True):
                from utils_robust import update_df
                cleaned_df = df.copy()
                for col in cat_cols:
                    cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
                update_df(cleaned_df, f"Trimmed whitespace from {len(cat_cols)} columns")
                st.success(f"✅ Trimmed {len(cat_cols)} columns")
                st.rerun()


# ============================================================================
# SMART DUPLICATES  (unchanged — no radios were in this section)
# ============================================================================

def render_smart_duplicates_section(df):
    """Multi-field fuzzy duplicate detection with confidence scoring."""

    st.markdown("### 🔍 Smart Duplicate Detection")
    st.caption("Intelligent entity resolution using multiple fields and confidence scoring")

    row_count = len(df)
    st.info(f"📊 Dataset: {row_count:,} rows × {len(df.columns)} columns")

    if row_count > 10000:
        st.warning(f"""
        ⚠️ **Large Dataset Detected ({row_count:,} rows)**

        Smart duplicate detection may take several minutes on large datasets.

        **Recommendations:**
        - Use sampling (below) for faster initial analysis
        - Increase similarity threshold (85% → 90%)
        - Limit supporting fields (3–4 is optimal)
        """)

    st.markdown("---")
    st.markdown("### ⚙️ Configuration")

    use_sampling = False
    sample_size  = row_count

    if row_count > 5000:
        st.markdown("**🎯 Performance Optimization**")
        use_sampling = st.checkbox(
            f"Use sampling (recommended for {row_count:,} rows)",
            value=row_count > 10000,
            key="smart_dup_use_sampling",
        )
        if use_sampling:
            sample_size = st.slider(
                "Sample size",
                min_value=1000,
                max_value=min(50000, row_count),
                value=min(5000, row_count),
                step=1000,
                key="smart_dup_sample_size_slider",
            )
            st.info(f"📊 Will analyse {sample_size:,} random rows (out of {row_count:,})")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Primary Identifier**")
        primary_field = st.selectbox("Primary field", df.columns.tolist(),
                                     key="smart_dup_primary_field")
        fuzzy_threshold = st.slider(
            "Similarity threshold", 0.60, 1.0, 0.85, 0.05,
            key="smart_dup_fuzzy_threshold",
        )
    with col2:
        st.markdown("**Supporting Fields**")
        available_fields = [col for col in df.columns if col != primary_field]
        default_fields = [
            f for f in available_fields[:10]
            if any(kw in f.lower() for kw in ['email','phone','age','id','name','address'])
        ]
        supporting_fields = st.multiselect(
            "Supporting fields", available_fields,
            default=default_fields[:3] if default_fields else available_fields[:3],
            key="smart_dup_supporting_fields",
        )
        confidence_threshold = st.slider(
            "Minimum confidence", 0.30, 1.0, 0.65, 0.05,
            key="smart_dup_confidence_threshold",
        )

    if not supporting_fields:
        st.warning("⚠️ Please select at least one supporting field")
        return

    st.markdown("---")

    if st.button("🔍 Detect Smart Duplicates", type="primary",
                 use_container_width=True, key="smart_dup_detect_btn"):
        df_to_analyze = df.sample(n=sample_size, random_state=42) if use_sampling else df
        if use_sampling:
            st.info(f"🎯 Analysing random sample of {sample_size:,} rows...")

        progress_bar = st.progress(0)
        status_text  = st.empty()

        try:
            status_text.text("🔍 Step 1/3: Grouping similar records...")
            progress_bar.progress(10)

            results = detect_smart_duplicates_optimized(
                df=df_to_analyze,
                primary_field=primary_field,
                supporting_fields=supporting_fields,
                fuzzy_threshold=fuzzy_threshold,
                confidence_threshold=confidence_threshold,
                progress_callback=lambda pct, msg: (
                    progress_bar.progress(pct), status_text.text(msg)
                ),
            )

            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")

            st.session_state['smart_dup_results']      = results
            st.session_state['smart_dup_primary']       = primary_field
            st.session_state['smart_dup_supporting']    = supporting_fields
            st.session_state['smart_dup_used_sampling'] = use_sampling
            st.session_state['smart_dup_sample_size']   = sample_size if use_sampling else row_count
            st.session_state['smart_dup_sample_df']     = df_to_analyze

            import time; time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            import traceback; st.code(traceback.format_exc())
            progress_bar.empty()
            status_text.empty()
            return

    if 'smart_dup_results' in st.session_state and st.session_state.smart_dup_results:
        results       = st.session_state['smart_dup_results']
        primary       = st.session_state.get('smart_dup_primary',    primary_field)
        supporting    = st.session_state.get('smart_dup_supporting', supporting_fields)
        used_sampling = st.session_state.get('smart_dup_used_sampling', False)
        s_size        = st.session_state.get('smart_dup_sample_size', row_count)

        st.markdown("---")
        if used_sampling:
            st.success(f"✅ Found {len(results)} potential duplicate pairs in sample of {s_size:,} rows")
            st.info("💡 Uncheck 'Use sampling' and run again to scan the full dataset.")
        else:
            st.success(f"✅ Found {len(results)} potential duplicate pairs")

        high_conf   = [r for r in results if r['confidence'] >= 0.85]
        medium_conf = [r for r in results if 0.60 <= r['confidence'] < 0.85]
        low_conf    = [r for r in results if r['confidence'] < 0.60]

        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 High Confidence (≥85%)",    len(high_conf))
        c2.metric("🟡 Medium Confidence (60–85%)", len(medium_conf))
        c3.metric("⚪ Low Confidence (<60%)",      len(low_conf))

        dt1, dt2, dt3 = st.tabs([
            f"🟢 High ({len(high_conf)})",
            f"🟡 Medium ({len(medium_conf)})",
            f"⚪ Low ({len(low_conf)})",
        ])
        with dt1:
            render_duplicate_pairs(high_conf, primary, supporting) if high_conf \
            else st.info("No high confidence duplicates found")
        with dt2:
            render_duplicate_pairs(medium_conf, primary, supporting) if medium_conf \
            else st.info("No medium confidence duplicates found")
        with dt3:
            render_duplicate_pairs(low_conf, primary, supporting) if low_conf \
            else st.info("No low confidence duplicates found")

    elif 'smart_dup_results' in st.session_state:
        st.info("ℹ️ No duplicates found. Try lowering confidence or similarity threshold.")


# ============================================================================
# SMART DUPLICATE ENGINE  (unchanged)
# ============================================================================

def detect_smart_duplicates_optimized(df, primary_field, supporting_fields,
                                      fuzzy_threshold, confidence_threshold,
                                      progress_callback=None):
    df_work = df.reset_index(drop=True).copy()
    if progress_callback:
        progress_callback(15, "🔍 Step 1/3: Finding similar records...")

    primary_groups = group_by_fuzzy_match_optimized(
        df_work, primary_field, fuzzy_threshold, progress_callback
    )
    if not primary_groups:
        if progress_callback:
            progress_callback(100, "✅ No potential duplicates found")
        return []

    if progress_callback:
        progress_callback(40, f"🔍 Step 2/3: Found {len(primary_groups)} groups. Scoring...")

    results          = []
    total_groups     = len(primary_groups)
    processed_groups = 0

    for group in primary_groups:
        if len(group) < 2:
            continue
        if len(group) > 50:
            group = group[:50]

        for idx1, idx2 in combinations(group, 2):
            try:
                record1    = df_work.iloc[idx1]
                record2    = df_work.iloc[idx2]
                match_info = calculate_match_confidence(
                    record1, record2, primary_field, supporting_fields
                )
                if match_info['confidence'] >= confidence_threshold:
                    match_info['index_1']  = df.index[idx1]
                    match_info['index_2']  = df.index[idx2]
                    match_info['record_1'] = record1.to_dict()
                    match_info['record_2'] = record2.to_dict()
                    results.append(match_info)
            except Exception:
                continue

        processed_groups += 1
        if progress_callback and processed_groups % max(1, total_groups // 10) == 0:
            pct = 40 + int((processed_groups / total_groups) * 50)
            progress_callback(pct, f"🔍 Analysing group {processed_groups}/{total_groups}...")

    if progress_callback:
        progress_callback(95, "🔍 Step 3/3: Sorting results...")

    results.sort(key=lambda x: x['confidence'], reverse=True)
    if progress_callback:
        progress_callback(100, "✅ Complete!")
    return results


def _safe_to_str_series(series: pd.Series) -> pd.Series:
    return series.astype(object).fillna('').astype(str)


def group_by_fuzzy_match_optimized(df, field, threshold, progress_callback=None):
    groups    = []
    processed = set()
    values    = _safe_to_str_series(df[field])

    skip_values   = {'unknown', 'n/a', 'na', 'null', 'none', 'test', '', 'nan'}
    valid_indices = [
        i for i in range(len(values))
        if values.iloc[i].strip() and len(values.iloc[i]) > 2
        and values.iloc[i].lower().strip() not in skip_values
    ]

    if progress_callback:
        progress_callback(20, f"🔍 Analysing {len(valid_indices):,} valid records...")

    total_valid     = len(valid_indices)
    processed_count = 0

    for idx1 in valid_indices:
        if idx1 in processed:
            continue
        try:
            val1 = values.iloc[idx1].lower().strip()
        except Exception:
            processed.add(idx1); continue

        if val1 in skip_values or not val1:
            processed.add(idx1); continue

        group    = [idx1]
        processed.add(idx1)
        val1_len = len(val1)

        for idx2 in valid_indices:
            if idx2 <= idx1 or idx2 in processed:
                continue
            try:
                val2 = values.iloc[idx2].lower().strip()
            except Exception:
                continue

            if val2 in skip_values or not val2:
                continue
            val2_len = len(val2)
            if val1_len == 0 or val2_len == 0:
                continue
            if min(val1_len, val2_len) / max(val1_len, val2_len) < 0.5:
                continue
            if string_similarity(val1, val2) >= threshold:
                group.append(idx2)
                processed.add(idx2)
                if len(group) >= 100:
                    break

        if len(group) > 1:
            groups.append(group)

        processed_count += 1
        if progress_callback and processed_count % max(1, total_valid // 20) == 0:
            pct = 20 + int((processed_count / total_valid) * 20)
            progress_callback(pct, f"🔍 Grouped {processed_count:,}/{total_valid:,} records...")

    return groups


def calculate_match_confidence(record1, record2, primary_field, supporting_fields):
    matches = {}
    weights = {}

    primary_sim            = string_similarity(str(record1[primary_field]), str(record2[primary_field]))
    matches[primary_field] = primary_sim
    weights[primary_field] = 0.3

    field_weight = 0.7 / len(supporting_fields) if supporting_fields else 0

    for field in supporting_fields:
        if field not in record1.index or field not in record2.index:
            continue
        val1, val2 = record1[field], record2[field]

        if pd.isna(val1) and pd.isna(val2):
            matches[field] = None; weights[field] = 0; continue
        if pd.isna(val1) or pd.isna(val2):
            matches[field] = 0.0; weights[field] = field_weight * 0.5; continue

        if is_numeric_field(val1, val2):
            match_score = 1.0 if numeric_match(val1, val2) else 0.0
        elif is_email_field(field):
            match_score = email_similarity(str(val1), str(val2))
        elif is_phone_field(field):
            match_score = phone_similarity(str(val1), str(val2))
        else:
            match_score = string_similarity(str(val1), str(val2))

        matches[field] = match_score
        weights[field] = field_weight

    total_weight = sum(weights.values())
    confidence   = (
        sum(matches[f] * weights[f] for f in matches if matches[f] is not None) / total_weight
        if total_weight else primary_sim
    )
    return {'confidence': confidence, 'matches': matches, 'weights': weights}


def string_similarity(str1, str2):
    str1 = str(str1).lower().strip()
    str2 = str(str2).lower().strip()
    if not str1 or not str2:
        return 0.0
    return SequenceMatcher(None, str1, str2).ratio()


def is_numeric_field(val1, val2):
    try:    float(val1); float(val2); return True
    except: return False


def numeric_match(val1, val2, tolerance=0.01):
    try:
        n1, n2 = float(val1), float(val2)
        return abs(n1 - n2) <= tolerance * max(abs(n1), abs(n2))
    except: return False


def is_email_field(field_name):
    return any(kw in field_name.lower() for kw in ['email', 'e-mail', 'mail', 'e_mail'])


def email_similarity(email1, email2):
    email1, email2 = email1.lower().strip(), email2.lower().strip()
    if email1 == email2: return 1.0
    parts1, parts2 = email1.split('@'), email2.split('@')
    if len(parts1) != 2 or len(parts2) != 2:
        return string_similarity(email1, email2)
    return string_similarity(parts1[0], parts2[0]) * 0.6 + \
           domain_similarity(parts1[1], parts2[1]) * 0.4


def domain_similarity(domain1, domain2):
    common_domains = {
        'gmail.com':   ['gamil.com', 'gmai.com', 'gmial.com'],
        'yahoo.com':   ['yaho.com', 'yahooo.com'],
        'hotmail.com': ['hotmial.com', 'hotmai.com'],
        'outlook.com': ['outlok.com'],
    }
    for correct, typos in common_domains.items():
        if (domain1 == correct and domain2 in typos) or \
           (domain2 == correct and domain1 in typos):
            return 0.95
    return string_similarity(domain1, domain2)


def is_phone_field(field_name):
    return any(kw in field_name.lower() for kw in ['phone','tel','mobile','cell','telephone'])


def phone_similarity(phone1, phone2):
    digits1 = re.sub(r'\D', '', str(phone1))
    digits2 = re.sub(r'\D', '', str(phone2))
    if not digits1 or not digits2: return 0.0
    if digits1 == digits2: return 1.0
    if len(digits1) >= 10 and len(digits2) >= 10:
        if digits1[-10:] == digits2[-10:]: return 0.95
    return string_similarity(digits1, digits2)


def render_duplicate_pairs(results, primary_field, supporting_fields):
    if 'selected_smart_merges' not in st.session_state:
        st.session_state.selected_smart_merges = set()

    for i, result in enumerate(results):
        confidence = result['confidence']
        record1    = result['record_1']
        record2    = result['record_2']
        idx1       = result['index_1']
        idx2       = result['index_2']
        pair_key   = f"{idx1}_{idx2}"

        with st.container():
            c1, c2, c3 = st.columns([3, 1, 1])
            with c1:
                conf_color = "🟢" if confidence >= 0.85 else "🟡" if confidence >= 0.60 else "⚪"
                st.markdown(f"**{conf_color} Match #{i+1}: {confidence*100:.1f}% Confidence**")
            with c2:
                merge_selected = st.checkbox(
                    "Select", key=f"smc_{pair_key}_{i}",
                    value=pair_key in st.session_state.selected_smart_merges,
                )
                if merge_selected: st.session_state.selected_smart_merges.add(pair_key)
                else:              st.session_state.selected_smart_merges.discard(pair_key)
            with c3:
                if st.button("👁️ Details", key=f"smd_{pair_key}_{i}", use_container_width=True):
                    tk = f"show_smd_{pair_key}"
                    st.session_state[tk] = not st.session_state.get(tk, False)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Record A** (Row {idx1})")
                d1 = {primary_field: record1.get(primary_field)}
                d1.update({f: record1[f] for f in supporting_fields if f in record1})
                st.json(d1, expanded=False)
            with c2:
                st.markdown(f"**Record B** (Row {idx2})")
                d2 = {primary_field: record2.get(primary_field)}
                d2.update({f: record2[f] for f in supporting_fields if f in record2})
                st.json(d2, expanded=False)

            if st.session_state.get(f"show_smd_{pair_key}", False):
                st.markdown("**Match Analysis:**")
                match_data = []
                for field, score in result['matches'].items():
                    if score is None:       status = "❓ Both missing";  score_display = "N/A"
                    elif score >= 0.95:     status = "✅ Perfect match";  score_display = f"{score*100:.0f}%"
                    elif score >= 0.85:     status = "✅ Strong match";   score_display = f"{score*100:.0f}%"
                    elif score >= 0.60:     status = "⚠️ Partial match";  score_display = f"{score*100:.0f}%"
                    else:                   status = "❌ Weak match";     score_display = f"{score*100:.0f}%"
                    match_data.append({'Field': field, 'Score': score_display, 'Status': status})
                st.dataframe(pd.DataFrame(match_data), use_container_width=True, hide_index=True)

            st.divider()

    if st.session_state.selected_smart_merges:
        st.markdown(f"**✓ {len(st.session_state.selected_smart_merges)} pairs selected**")
        m1, m2 = st.columns([1, 3])
        with m1:
            if st.button("🔀 Merge Selected", type="primary",
                         use_container_width=True, key="smart_merge_final_btn"):
                merge_smart_duplicate_pairs(results)
        with m2:
            if st.button("Clear Selection", use_container_width=True,
                         key="smart_clear_selection_btn"):
                st.session_state.selected_smart_merges = set()
                st.rerun()


def merge_smart_duplicate_pairs(results):
    if 'df' not in st.session_state or not st.session_state.selected_smart_merges:
        return

    df           = st.session_state.df.copy()
    rows_to_drop = set()

    for pair_key in st.session_state.selected_smart_merges:
        try:
            idx1, idx2 = map(int, pair_key.split('_'))
            result = next(
                (r for r in results if r['index_1'] == idx1 and r['index_2'] == idx2),
                None,
            )
            if result:
                rows_to_drop.add(idx2)
        except Exception as e:
            st.warning(f"Skipping pair {pair_key}: {str(e)}")
            continue

    if not rows_to_drop:
        st.warning("No valid rows to drop")
        return

    valid_indices = [idx for idx in rows_to_drop if idx in df.index]
    if not valid_indices:
        st.error("Error: selected indices don't exist in the dataframe")
        return

    try:
        df_cleaned = df.drop(index=valid_indices).reset_index(drop=True)
        from utils_robust import update_df, log_action
        log_action("Smart Merge Duplicates",
                   f"Merged {len(valid_indices)} duplicate records using multi-field matching")
        update_df(df_cleaned, f"Smart merged {len(valid_indices)} duplicate records")
        st.session_state.selected_smart_merges = set()
        st.session_state.pop('smart_dup_results', None)
        st.session_state.pop('smart_dup_sample_df', None)
        st.success(f"✅ Merged {len(valid_indices)} duplicate records!")
        st.rerun()
    except Exception as e:
        st.error(f"Error during merge: {str(e)}")
        import traceback; st.code(traceback.format_exc())


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    render_data_cleaning_page()