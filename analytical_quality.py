"""
ANALYTIQAL — Data Quality & Analytical Readiness Assessment
============================================================
Stage 5 of the Dataset Manager workflow.

Sits between:   Row Integrity (Step 4)  →  THIS  →  Reshape (Step 6)

Design contract:
  • READ-ONLY. Never modifies st.session_state.df or any other data key.
  • Never calls TransformationEngine or HistoryManager.
  • Never calls WorkflowState.done() — only the Continue button does that.
  • All computation uses @st.cache_data to avoid rerunning on widget changes.
  • Every metric is accompanied by a plain-language explanation of WHY it matters.
  • Every risk flag surfaces a decision question, not an answer.

Wiring into dataset_manager.py (4 surgical patches):
  1. Add "quality_assessed" to WORKFLOW_STAGES (after "row_integrity")
  2. Add entry to STAGE_CONFIG (step=5, shifts others +1)
  3. Add entry to STAGE_LABELS
  4. Add routing block in render_dataset_manager()
  5. Import render_quality_assessment_stage at top of dataset_manager.py
     OR keep this file standalone and call directly.

Public entry point:
    from analytiqal_quality import render_quality_assessment_stage
    render_quality_assessment_stage()
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── Design tokens (mirrors dataset_manager.py D dict) ────────────────────
_D = {
    "c_brand":        "#667eea",
    "c_brand_dark":   "#5a6fd8",
    "c_success":      "#27ae60",
    "c_warning":      "#f39c12",
    "c_danger":       "#e74c3c",
    "c_neutral":      "#6c757d",
    "c_surface":      "#f8f9ff",
    "c_border":       "#e8eaf0",
    "c_text_primary": "#1a1a2e",
    "c_text_muted":   "#6c757d",
    "t_xs":  "0.72rem",
    "t_sm":  "0.83rem",
    "t_base":"0.92rem",
    "t_lg":  "1.1rem",
    "t_xl":  "1.4rem",
}

_BRAND_GRADIENT = "135deg, #667eea 0%, #764ba2 100%"

MISSING_HIGH  = 0.20   # > 20% → critical
MISSING_MED   = 0.05   # > 5%  → warning
SKEW_HIGH     = 2.0    # |skew| > 2 → flag
CARD_HIGH     = 0.90   # > 90% unique in text col → flag
OUTLIER_PCT   = 0.05   # > 5% rows are outliers → flag
CORR_HIGH     = 0.85   # |r| > 0.85 → multicollinearity flag


# ═══════════════════════════════════════════════════════════════════════════
# SHARED UI PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════

def _insight_box(title: str, body: str, level: str = "info"):
    """Render a styled insight card. level: info | warn | danger | success"""
    cfg = {
        "info":    (_D["c_brand"],   "#f0f4ff"),
        "warn":    (_D["c_warning"], "#fff8e1"),
        "danger":  (_D["c_danger"],  "#fef0f0"),
        "success": (_D["c_success"], "#f0faf4"),
    }.get(level, (_D["c_neutral"], _D["c_surface"]))
    border, bg = cfg
    st.markdown(
        f"<div style='background:{bg};border-left:4px solid {border};"
        f"border-radius:6px;padding:12px 16px;margin:6px 0;'>"
        f"<div style='font-size:{_D['t_sm']};font-weight:700;color:{border};"
        f"margin-bottom:4px;'>{title}</div>"
        f"<div style='font-size:{_D['t_sm']};color:{_D['c_text_primary']};'>{body}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _section(title: str):
    st.markdown(
        f"<div style='margin:28px 0 10px;'>"
        f"<span style='font-size:{_D['t_lg']};font-weight:700;"
        f"color:{_D['c_text_primary']};'>{title}</span>"
        f"<div style='height:2px;background:linear-gradient({_BRAND_GRADIENT});"
        f"border-radius:2px;width:40px;margin-top:4px;'></div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _why_box(text: str):
    """Small 'Why this matters' explanation card."""
    st.markdown(
        f"<div style='background:{_D['c_surface']};border:1px solid {_D['c_border']};"
        f"border-radius:6px;padding:8px 12px;margin:4px 0 12px;'>"
        f"<span style='font-size:{_D['t_xs']};color:{_D['c_text_muted']};font-weight:600;"
        f"text-transform:uppercase;letter-spacing:0.06em;'>Why this matters</span>"
        f"<div style='font-size:{_D['t_sm']};color:{_D['c_text_primary']};margin-top:3px;'>"
        f"{text}</div></div>",
        unsafe_allow_html=True,
    )


def _decision_box(questions: List[str]):
    """Render 'Decisions to consider' list."""
    qs = "".join(f"<li style='margin:3px 0;'>{q}</li>" for q in questions)
    st.markdown(
        f"<div style='background:#fffde7;border:1px solid #f9a825;"
        f"border-radius:6px;padding:10px 14px;margin:6px 0 14px;'>"
        f"<span style='font-size:{_D['t_xs']};color:#e65100;font-weight:700;"
        f"text-transform:uppercase;letter-spacing:0.06em;'>🤔 Decisions to consider</span>"
        f"<ul style='margin:6px 0 0;padding-left:18px;font-size:{_D['t_sm']};"
        f"color:{_D['c_text_primary']};'>{qs}</ul></div>",
        unsafe_allow_html=True,
    )


def _risk_badge(label: str, level: str):
    """Inline coloured badge for risk level."""
    colors = {
        "critical": ("#e74c3c", "#fef0f0"),
        "high":     ("#e74c3c", "#fef0f0"),
        "medium":   ("#f39c12", "#fff8e1"),
        "low":      ("#27ae60", "#f0faf4"),
        "none":     ("#6c757d", "#f8f9ff"),
    }
    fg, bg = colors.get(level, ("#6c757d", "#f8f9ff"))
    return (
        f"<span style='background:{bg};color:{fg};border:1px solid {fg}44;"
        f"border-radius:4px;padding:2px 8px;font-size:{_D['t_xs']};"
        f"font-weight:700;'>{label}</span>"
    )


# ═══════════════════════════════════════════════════════════════════════════
# CACHED COMPUTATION LAYER — all heavy lifting is memoised
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _compute_missingness(df: pd.DataFrame) -> Dict:
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols

    # Column-level
    col_null    = df.isnull().sum()
    col_pct     = col_null / n_rows
    col_df = pd.DataFrame({
        "Column":    col_null.index,
        "Missing":   col_null.values,
        "Missing %": (col_pct * 100).round(2),
        "Present":   n_rows - col_null.values,
        "Risk":      col_pct.apply(
            lambda p: "critical" if p >= MISSING_HIGH else
                      "medium"   if p >= MISSING_MED  else "none"
        ).values,
    })

    # Row-level
    row_null  = df.isnull().sum(axis=1)
    row_pct   = row_null / n_cols
    row_dist  = pd.cut(
        row_pct,
        bins=[-0.001, 0, 0.25, 0.50, 0.75, 1.001],
        labels=["Complete", "1–25% missing", "26–50%", "51–75%", ">75%"],
    ).value_counts().reindex(
        ["Complete", "1–25% missing", "26–50%", "51–75%", ">75%"], fill_value=0
    )

    # Missingness pattern (which columns tend to be missing together)
    miss_bool   = df.isnull()
    pattern_corr = miss_bool.corr() if miss_bool.any().any() else pd.DataFrame()

    return {
        "n_rows":         n_rows,
        "n_cols":         n_cols,
        "total_cells":    total_cells,
        "total_missing":  int(col_null.sum()),
        "overall_pct":    col_null.sum() / total_cells * 100,
        "col_df":         col_df,
        "row_null_series":row_null,
        "row_pct_series": row_pct,
        "row_dist":       row_dist,
        "rows_complete":  int((row_null == 0).sum()),
        "rows_partial":   int((row_null > 0).sum()),
        "pattern_corr":   pattern_corr,
        "cols_critical":  col_df[col_df["Risk"] == "critical"]["Column"].tolist(),
        "cols_warn":      col_df[col_df["Risk"] == "medium"]["Column"].tolist(),
    }


@st.cache_data(show_spinner=False)
def _compute_duplicates(df: pd.DataFrame) -> Dict:
    n_full   = int(df.duplicated().sum())
    pct_full = n_full / len(df) * 100

    # Column-subset duplicates (detect near-duplicates)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols     = df.select_dtypes(include="object").columns.tolist()
    n_num_dup    = int(df.duplicated(subset=numeric_cols).sum()) if numeric_cols else 0
    n_cat_dup    = int(df.duplicated(subset=cat_cols).sum())     if cat_cols    else 0

    risk = ("critical" if pct_full > 10 else
            "medium"   if pct_full > 2  else
            "low"      if pct_full > 0  else "none")

    return {
        "n_full":    n_full,
        "pct_full":  pct_full,
        "n_num_dup": n_num_dup,
        "n_cat_dup": n_cat_dup,
        "risk":      risk,
    }


@st.cache_data(show_spinner=False)
def _compute_numeric_health(df: pd.DataFrame) -> Dict:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        return {"cols": [], "stats_df": pd.DataFrame(), "outlier_df": pd.DataFrame()}

    rows = []
    outlier_rows = []

    for col in num_cols:
        s = df[col].dropna()
        if len(s) < 4:
            continue

        mean   = float(s.mean())
        median = float(s.median())
        std    = float(s.std())
        skew   = float(s.skew())
        kurt   = float(s.kurtosis())
        cv     = abs(std / mean * 100) if mean != 0 else 0.0

        # IQR outliers
        q1, q3   = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr      = q3 - q1
        lb       = q1 - 1.5 * iqr
        ub       = q3 + 1.5 * iqr
        n_out    = int(((s < lb) | (s > ub)).sum())
        out_pct  = n_out / len(s) * 100

        # Z-score outliers (|z| > 3)
        z_out    = int((np.abs((s - mean) / std) > 3).sum()) if std > 0 else 0

        # Risk classification
        skew_risk  = "high"   if abs(skew)  >= SKEW_HIGH else "low"
        out_risk   = "high"   if out_pct    >= OUTLIER_PCT * 100 else "low"
        miss_pct   = df[col].isna().mean() * 100

        rows.append({
            "Column":     col,
            "Mean":       round(mean, 4),
            "Median":     round(median, 4),
            "Std Dev":    round(std, 4),
            "Skewness":   round(skew, 3),
            "Kurtosis":   round(kurt, 3),
            "CV %":       round(cv, 1),
            "Missing %":  round(miss_pct, 1),
            "Skew Risk":  skew_risk,
        })
        outlier_rows.append({
            "Column":       col,
            "IQR Outliers": n_out,
            "IQR Out %":    round(out_pct, 1),
            "Z>3 Outliers": z_out,
            "Lower Fence":  round(lb, 4),
            "Upper Fence":  round(ub, 4),
            "Out Risk":     out_risk,
        })

    return {
        "cols":       num_cols,
        "stats_df":   pd.DataFrame(rows),
        "outlier_df": pd.DataFrame(outlier_rows),
    }


@st.cache_data(show_spinner=False)
def _compute_categorical_health(df: pd.DataFrame) -> Dict:
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if not cat_cols:
        return {"cols": [], "stats_df": pd.DataFrame()}

    rows = []
    for col in cat_cols:
        s           = df[col].dropna()
        n_unique    = int(df[col].nunique())
        card_ratio  = n_unique / max(len(df), 1)
        mode_vals   = s.mode()
        mode_val    = str(mode_vals.iloc[0]) if not mode_vals.empty else "—"
        mode_freq   = int(s.value_counts().iloc[0]) if len(s) > 0 else 0
        mode_pct    = mode_freq / max(len(df), 1) * 100

        card_risk = (
            "high"   if card_ratio >= CARD_HIGH else
            "medium" if card_ratio >= 0.50      else
            "low"    if n_unique <= 2            else "none"
        )

        rows.append({
            "Column":         col,
            "Unique Values":  n_unique,
            "Cardinality %":  round(card_ratio * 100, 1),
            "Mode":           mode_val[:30] + ("…" if len(mode_val) > 30 else ""),
            "Mode Freq %":    round(mode_pct, 1),
            "Missing %":      round(df[col].isna().mean() * 100, 1),
            "Card Risk":      card_risk,
        })

    return {"cols": cat_cols, "stats_df": pd.DataFrame(rows)}


@st.cache_data(show_spinner=False)
def _compute_correlation_health(df: pd.DataFrame) -> Dict:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) < 2:
        return {"matrix": pd.DataFrame(), "high_pairs": [], "cols": num_cols}

    corr = df[num_cols].corr()
    high_pairs = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            val = corr.iloc[i, j]
            if abs(val) >= CORR_HIGH:
                high_pairs.append({
                    "Column A":    num_cols[i],
                    "Column B":    num_cols[j],
                    "Pearson r":   round(val, 3),
                    "|r|":         round(abs(val), 3),
                    "Direction":   "Positive" if val > 0 else "Negative",
                })
    high_pairs.sort(key=lambda x: -x["|r|"])
    return {"matrix": corr, "high_pairs": high_pairs, "cols": num_cols}


@st.cache_data(show_spinner=False)
def _compute_analytical_readiness(df: pd.DataFrame) -> Dict:
    """
    Compute an Analytical Readiness Score (0–100) across five dimensions.
    Read-only. No data is modified.
    """
    n_rows, n_cols = df.shape

    # 1. Completeness (40 pts)
    miss_rate   = df.isnull().mean().mean()
    completeness = max(0.0, 1.0 - miss_rate * 2) * 40

    # 2. Uniqueness (20 pts) — penalise duplicates
    dup_rate    = df.duplicated().sum() / max(n_rows, 1)
    uniqueness  = max(0.0, 1.0 - dup_rate * 5) * 20

    # 3. Numeric health (20 pts) — penalise extreme skew and outliers
    num_cols    = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        avg_skew   = float(df[num_cols].skew().abs().mean())
        skew_pen   = min(1.0, avg_skew / (SKEW_HIGH * 2))
        num_health = max(0.0, 1.0 - skew_pen) * 20
    else:
        num_health = 20.0

    # 4. Categorical health (10 pts) — penalise very high cardinality
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        avg_card   = float(pd.Series([df[c].nunique() / n_rows for c in cat_cols]).mean())
        cat_health = max(0.0, 1.0 - avg_card) * 10
    else:
        cat_health = 10.0

    # 5. Structural soundness (10 pts)
    constant_cols = sum(1 for c in df.columns if df[c].nunique(dropna=True) <= 1)
    struct_health = max(0.0, (1.0 - constant_cols / max(n_cols, 1))) * 10

    total   = completeness + uniqueness + num_health + cat_health + struct_health
    total   = round(min(100.0, total), 1)

    grade   = ("A" if total >= 85 else
               "B" if total >= 70 else
               "C" if total >= 55 else
               "D" if total >= 40 else "F")

    verdict = {
        "A": ("Ready for analysis", "success"),
        "B": ("Minor issues — proceed with caution", "info"),
        "C": ("Moderate issues — address before modeling", "warn"),
        "D": ("Significant issues — clean before analysis", "danger"),
        "F": ("Critical issues — dataset not analytically trustworthy", "danger"),
    }[grade]

    return {
        "total":           total,
        "grade":           grade,
        "verdict":         verdict[0],
        "verdict_level":   verdict[1],
        "completeness":    round(completeness / 40 * 100, 1),
        "uniqueness":      round(uniqueness   / 20 * 100, 1),
        "num_health":      round(num_health   / 20 * 100, 1),
        "cat_health":      round(cat_health   / 10 * 100, 1),
        "struct_health":   round(struct_health/ 10 * 100, 1),
        "dim_scores":      {
            "Completeness (40pts)":      round(completeness, 1),
            "Uniqueness (20pts)":        round(uniqueness, 1),
            "Numeric Health (20pts)":    round(num_health, 1),
            "Categorical Health (10pts)":round(cat_health, 1),
            "Structural Soundness (10pts)":round(struct_health, 1),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# SUB-STEP RENDERERS
# ═══════════════════════════════════════════════════════════════════════════

def _render_readiness_score(df: pd.DataFrame):
    """
    SUB-STEP 1 — Analytical Readiness Score
    Purpose   : Give users an at-a-glance quality verdict before they invest
                time exploring or modelling.
    Metrics   : Weighted composite across 5 dimensions (0–100).
    Risk flags: Grade C/D/F → escalate to user.
    """
    _section("① Analytical Readiness Score")
    _why_box(
        "A dataset can be structurally correct but analytically untrustworthy. "
        "This score tells you how confidently you can draw conclusions from the "
        "data as it stands — before cleaning or modelling. Think of it as a "
        "<b>trust rating</b> for your dataset."
    )

    r = _compute_analytical_readiness(df)
    score, grade, verdict, level = r["total"], r["grade"], r["verdict"], r["verdict_level"]

    g1, g2, g3 = st.columns([1, 2, 3])
    with g1:
        grade_colors = {"A": "#27ae60", "B": "#667eea", "C": "#f39c12", "D": "#e74c3c", "F": "#c0392b"}
        st.markdown(
            f"<div style='background:{grade_colors.get(grade, '#6c757d')}18;"
            f"border:2px solid {grade_colors.get(grade, '#6c757d')};"
            f"border-radius:12px;padding:20px;text-align:center;'>"
            f"<div style='font-size:3rem;font-weight:900;color:{grade_colors.get(grade, '#6c757d')};'>{grade}</div>"
            f"<div style='font-size:0.75rem;color:#6c757d;font-weight:600;text-transform:uppercase;'>Grade</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with g2:
        st.markdown(
            f"<div style='padding:10px 0;'>"
            f"<div style='font-size:2.4rem;font-weight:800;color:{_D['c_text_primary']};'>{score}/100</div>"
            f"<div style='font-size:{_D['t_base']};color:{_D['c_text_muted']};margin:4px 0;'>{verdict}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.progress(score / 100)
    with g3:
        # Dimension breakdown
        for dim, pts in r["dim_scores"].items():
            max_pts = int(dim.split("(")[1].replace("pts)", "").strip())
            bar_pct = pts / max_pts
            bar_color = "#27ae60" if bar_pct >= 0.85 else "#f39c12" if bar_pct >= 0.60 else "#e74c3c"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;margin:3px 0;'>"
                f"<div style='font-size:{_D['t_xs']};width:190px;color:{_D['c_text_muted']};'>{dim}</div>"
                f"<div style='flex:1;background:#e8eaf0;border-radius:4px;height:8px;'>"
                f"<div style='width:{bar_pct*100:.0f}%;background:{bar_color};"
                f"border-radius:4px;height:8px;'></div></div>"
                f"<div style='font-size:{_D['t_xs']};color:{_D['c_text_muted']};width:36px;text-align:right;'>"
                f"{pts:.0f}</div></div>",
                unsafe_allow_html=True,
            )

    _insight_box(f"Verdict: {verdict}", (
        "This score is computed from missingness, duplicates, distribution health, "
        "categorical cardinality, and structural soundness. "
        "Work through the sub-steps below to understand exactly what is driving it."
    ), level=level)

    _decision_box([
        "Is this score acceptable for your intended analysis?",
        "Do you need to address all issues, or only the critical ones?",
        "Will low scores in specific dimensions affect your particular analysis goal?",
    ])


def _render_missingness(df: pd.DataFrame):
    """
    SUB-STEP 2 — Missingness Intelligence
    Purpose   : Understand WHERE data is absent and WHETHER the absence is random.
    Metrics   : Column-wise %, row-wise %, distribution, co-missingness pattern.
    Risk flags: > 20% col missingness, > 30% rows partially missing, clustered patterns.
    """
    _section("② Missingness Intelligence")
    _why_box(
        "Missing values are not all equal. A column missing 5% of values is very different "
        "from one missing 60%. More importantly: <b>why</b> values are missing matters more than "
        "how many. Random missingness (MCAR) can be handled by dropping rows. Systematic "
        "missingness (MAR or MNAR) can introduce bias that silently corrupts your analysis."
    )

    m = _compute_missingness(df)

    # ── Overall snapshot ─────────────────────────────────────────────────
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Overall Missing %", f"{m['overall_pct']:.1f}%")
    mc2.metric("Complete Rows",     f"{m['rows_complete']:,}", delta=f"of {m['n_rows']:,}")
    mc3.metric("Affected Columns",  f"{len(m['cols_critical']) + len(m['cols_warn'])}")
    mc4.metric("Critical Columns",  f"{len(m['cols_critical'])}", delta="(>20% missing)")

    if m["total_missing"] == 0:
        _insight_box("✅ No missing values", "This dataset is fully complete.", "success")
        return

    tab_col, tab_row, tab_pattern = st.tabs([
        "Column-wise Missing", "Row-wise Missing", "Co-missingness Pattern"
    ])

    # ── A. Column-wise ────────────────────────────────────────────────────
    with tab_col:
        st.caption(
            "**Purpose:** Identify which columns have unacceptable data loss and "
            "whether entire features need to be dropped or imputed."
        )

        col_df = m["col_df"].sort_values("Missing %", ascending=False)
        col_df_display = col_df.copy()
        col_df_display["Risk Signal"] = col_df_display["Risk"].apply(
            lambda r: "🔴 Critical (>20%)" if r == "critical" else
                      "🟡 Warning (5–20%)" if r == "medium"   else "🟢 Acceptable"
        )

        # Progress bar chart
        fig = go.Figure()
        colors = col_df["Risk"].apply(
            lambda r: "#e74c3c" if r == "critical" else
                      "#f39c12" if r == "medium"   else "#27ae60"
        ).tolist()

        fig.add_trace(go.Bar(
            x=col_df["Column"],
            y=col_df["Missing %"],
            marker_color=colors,
            text=col_df["Missing %"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
        ))
        fig.add_hline(y=20, line_dash="dash", line_color="#e74c3c",
                      annotation_text="20% threshold", annotation_position="right")
        fig.add_hline(y=5,  line_dash="dash", line_color="#f39c12",
                      annotation_text="5% threshold",  annotation_position="right")
        fig.update_layout(
            title="Column Missing %",
            height=380,
            xaxis_tickangle=-30,
            showlegend=False,
            plot_bgcolor="white",
            yaxis=dict(title="Missing %", range=[0, 105]),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            col_df_display[["Column", "Missing", "Missing %", "Present", "Risk Signal"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Missing %": st.column_config.ProgressColumn(
                    "Missing %", format="%.1f%%", min_value=0, max_value=100
                ),
            },
        )

        if m["cols_critical"]:
            _insight_box(
                f"🔴 {len(m['cols_critical'])} critical column(s): {', '.join(m['cols_critical'][:5])}",
                "These columns are missing more than 20% of their values. Imputing from so few "
                "observed values risks introducing statistical bias. Dropping rows would "
                "discard substantial portions of the dataset.",
                "danger",
            )

        _decision_box([
            "For columns with >50% missing: can they be dropped entirely, or are they analytically essential?",
            "For columns with 5–20% missing: is the missingness random, or related to another variable?",
            "Would imputing with mean/median distort the true distribution of this variable?",
            "Do missing values in key columns indicate a data collection problem worth investigating upstream?",
        ])

    # ── B. Row-wise ───────────────────────────────────────────────────────
    with tab_row:
        st.caption(
            "**Purpose:** Understand whether missingness is concentrated in a "
            "few problematic rows (which might be safely removed) or spread "
            "uniformly (which suggests a systemic collection issue)."
        )

        # Distribution of row completeness
        dist_df = m["row_dist"].reset_index()
        dist_df.columns = ["Completeness Band", "Row Count"]
        dist_df["Row %"] = (dist_df["Row Count"] / m["n_rows"] * 100).round(1)

        fig = go.Figure(go.Bar(
            x=dist_df["Completeness Band"],
            y=dist_df["Row Count"],
            text=dist_df["Row %"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
            marker_color=["#27ae60", "#a8d5a2", "#f39c12", "#e87040", "#e74c3c"],
        ))
        fig.update_layout(
            title="Row Missingness Distribution",
            height=360,
            yaxis_title="Number of Rows",
            plot_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            dist_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Row %": st.column_config.ProgressColumn(
                    "Row %", format="%.1f%%", min_value=0, max_value=100
                ),
            },
        )

        # Row-level histogram
        row_pct_series = m["row_pct_series"]
        heavy_rows = int((row_pct_series > 0.5).sum())
        if heavy_rows > 0:
            _insight_box(
                f"🟡 {heavy_rows:,} rows are >50% empty",
                f"That's {heavy_rows/m['n_rows']*100:.1f}% of your dataset. "
                "Rows that are mostly empty often provide little analytical value and "
                "can introduce distortion in aggregate calculations.",
                "warn",
            )

        _decision_box([
            "Should rows that are >50% empty be dropped or flagged?",
            "Are the heavily-missing rows clustered around a specific time period, category, or region?",
            "Does the row missingness pattern suggest a data export or join problem?",
            "If you drop high-missingness rows, will you lose important subgroups from your analysis?",
        ])

    # ── C. Co-missingness Pattern ─────────────────────────────────────────
    with tab_pattern:
        st.caption(
            "**Purpose:** Detect whether columns are missing *together* — "
            "a sign that the data is Missing Not At Random (MNAR), which means "
            "the missingness itself carries information and simple imputation will introduce bias."
        )

        miss_cols = m["col_df"][m["col_df"]["Missing"] > 0]["Column"].tolist()
        if len(miss_cols) < 2:
            st.info("Need at least 2 columns with missing values to detect co-missingness patterns.")
        else:
            miss_bool  = df[miss_cols].isnull().astype(int)
            co_corr    = miss_bool.corr()

            fig = go.Figure(go.Heatmap(
                z=co_corr.values,
                x=co_corr.columns.tolist(),
                y=co_corr.columns.tolist(),
                colorscale="RdBu_r",
                zmid=0,
                text=co_corr.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Correlation of missingness"),
            ))
            fig.update_layout(
                title="Co-missingness Correlation Heatmap",
                height=420,
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Flag high co-missingness pairs
            strong_pairs = []
            for i in range(len(miss_cols)):
                for j in range(i + 1, len(miss_cols)):
                    val = co_corr.iloc[i, j]
                    if abs(val) >= 0.7:
                        strong_pairs.append(
                            (miss_cols[i], miss_cols[j], round(val, 2))
                        )
            if strong_pairs:
                _insight_box(
                    f"🔴 {len(strong_pairs)} column pair(s) have correlated missingness",
                    "When columns are missing together, imputing them independently "
                    "will produce inconsistent data. Consider joint imputation strategies "
                    "or investigate whether a shared upstream source caused the gap.",
                    "danger",
                )
                for a, b, r in strong_pairs[:6]:
                    st.caption(f"• `{a}` and `{b}` — missingness correlation = **{r}**")

        _decision_box([
            "Are any pairs of columns missing for the same rows? If so, why?",
            "Does the co-missingness align with a known data collection event?",
            "Would treating these as Missing At Random (and filling independently) be accurate?",
            "Should you create an explicit 'was_missing' indicator column before imputing?",
        ])


def _render_duplicates(df: pd.DataFrame):
    """
    SUB-STEP 3 — Duplicate & Near-Duplicate Detection
    Purpose   : Identify rows that repeat information and assess the risk.
    Metrics   : Exact duplicates, numeric-only dupes, categorical-only dupes.
    Risk flags: > 2% duplicates is analytically significant.
    """
    _section("③ Duplicate & Near-Duplicate Detection")
    _why_box(
        "Duplicate rows silently distort every aggregate statistic you compute — "
        "means, counts, proportions, and correlations are all inflated. In a dataset "
        "used for modelling, duplicates cause <b>data leakage</b> between training and "
        "test sets, giving you false confidence in model performance."
    )

    d = _compute_duplicates(df)

    dc1, dc2, dc3 = st.columns(3)
    risk_html = _risk_badge(d["risk"].upper(), d["risk"])
    dc1.metric("Exact Duplicates", f"{d['n_full']:,}",
               delta=f"{d['pct_full']:.1f}% of rows")
    dc2.metric("Numeric-only Dupes", f"{d['n_num_dup']:,}")
    dc3.metric("Categorical-only Dupes", f"{d['n_cat_dup']:,}")

    st.markdown(f"Risk level: {risk_html}", unsafe_allow_html=True)

    if d["n_full"] == 0:
        _insight_box("✅ No exact duplicates", "Dataset rows are all unique.", "success")
    else:
        # Show sample
        dupes_df = df[df.duplicated(keep=False)].head(20)
        with st.expander(f"Preview duplicate rows (showing up to 20 of {d['n_full']:,})", expanded=False):
            st.dataframe(dupes_df, use_container_width=True, height=240, hide_index=False)

        if d["pct_full"] > 10:
            _insight_box(
                f"🔴 High duplicate rate: {d['pct_full']:.1f}%",
                "More than 1 in 10 rows is a repeat. This strongly suggests a data "
                "extraction or join problem — e.g. a many-to-many join without deduplication.",
                "danger",
            )
        elif d["pct_full"] > 2:
            _insight_box(
                f"🟡 Moderate duplicates: {d['pct_full']:.1f}%",
                "This level of duplication will noticeably bias aggregation results. "
                "Verify whether duplicates are intentional (e.g. repeated measurements) "
                "or accidental (data extraction bug).",
                "warn",
            )

    _decision_box([
        "Are duplicates intentional (e.g. multiple observations per subject) or errors?",
        "If intentional, do you need to aggregate them before analysis?",
        "Will you remove all duplicates, or keep the first/last occurrence?",
        "Do numeric-only duplicates suggest different categories with identical measurements?",
    ])


def _render_numeric_health(df: pd.DataFrame):
    """
    SUB-STEP 4 — Numeric Variable Health
    Purpose   : Flag distributions that will break statistical assumptions.
    Metrics   : Skewness, kurtosis, CV, IQR and Z-score outlier counts.
    Risk flags: |skew| > 2, outlier % > 5%, CV > 100%.
    """
    _section("④ Numeric Variable Health")
    _why_box(
        "Most statistical methods (regression, ANOVA, correlation) assume that numeric "
        "variables follow approximately normal distributions. Severely skewed data or "
        "extreme outliers can make your coefficients unreliable, your p-values "
        "meaningless, and your predictions biased toward extreme observations. "
        "This sub-step tells you which variables need transformation before modelling."
    )

    h = _compute_numeric_health(df)

    if not h["cols"]:
        st.info("No numeric columns in the dataset.")
        return

    stats_df  = h["stats_df"]
    out_df    = h["outlier_df"]

    tab_dist, tab_outlier, tab_viz = st.tabs([
        "Distribution Flags", "Outlier Census", "Visual Inspection"
    ])

    with tab_dist:
        st.caption(
            "**Purpose:** Identify which variables violate normality assumptions "
            "and need log, sqrt, or Box-Cox transformation before analysis."
        )

        display_df = stats_df.copy()
        display_df["Skew Signal"] = display_df["Skewness"].apply(
            lambda x: "🔴 High" if abs(x) >= SKEW_HIGH else "🟢 OK"
        )
        display_df["CV Signal"] = display_df["CV %"].apply(
            lambda x: "🔴 Unstable" if x > 100 else "🟢 Stable"
        )

        st.dataframe(
            display_df[["Column", "Mean", "Median", "Std Dev", "Skewness",
                         "Kurtosis", "CV %", "Missing %", "Skew Signal", "CV Signal"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Missing %": st.column_config.ProgressColumn(
                    "Missing %", format="%.1f%%", min_value=0, max_value=100
                ),
            },
        )

        skewed = stats_df[stats_df["Skew Risk"] == "high"]["Column"].tolist()
        if skewed:
            _insight_box(
                f"🔴 {len(skewed)} highly skewed column(s): {', '.join(skewed[:6])}",
                "Skewness above 2 means the distribution is far from symmetric. "
                "Pearson correlation, linear regression, and t-tests all assume "
                "approximate normality — results on these columns may be unreliable.",
                "danger",
            )

        _decision_box([
            "Which skewed columns are critical enough to transform (log, sqrt, Box-Cox)?",
            "Is high skewness expected due to the nature of the variable (e.g. income, sales)?",
            "Should you analyse these using non-parametric methods instead of transforming?",
            "Does a large mean-vs-median gap suggest outliers are pulling the mean?",
        ])

    with tab_outlier:
        st.caption(
            "**Purpose:** Count extreme values by two methods. "
            "IQR is distribution-free and robust. Z-score assumes normality. "
            "Disagreement between them is itself a signal."
        )

        display_out = out_df.copy()
        display_out["Risk"] = out_df["Out Risk"].apply(
            lambda r: "🔴 High (>5%)" if r == "high" else "🟢 Acceptable"
        )
        st.dataframe(
            display_out[["Column", "IQR Outliers", "IQR Out %",
                         "Z>3 Outliers", "Lower Fence", "Upper Fence", "Risk"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "IQR Out %": st.column_config.ProgressColumn(
                    "IQR Out %", format="%.1f%%", min_value=0, max_value=100
                ),
            },
        )

        high_out_cols = out_df[out_df["Out Risk"] == "high"]["Column"].tolist()
        if high_out_cols:
            _insight_box(
                f"🟡 {len(high_out_cols)} column(s) with >5% outliers: {', '.join(high_out_cols[:5])}",
                "High outlier rates skew means, inflate standard deviations, and "
                "can dominate regression coefficients. They may represent data entry "
                "errors — or they may be the most analytically important observations.",
                "warn",
            )

        _decision_box([
            "Are outliers genuine observations or data entry errors?",
            "Do outliers cluster in specific categories or time periods?",
            "Should outliers be winsorised, removed, or analysed as a separate segment?",
            "Would a robust statistical method (median-based) be more appropriate here?",
        ])

    with tab_viz:
        if len(h["cols"]) > 0:
            sel_col = st.selectbox("Select column to inspect", h["cols"], key="qa_num_col")
            s = df[sel_col].dropna()

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Distribution", "Box Plot"),
            )

            # Histogram
            fig.add_trace(
                go.Histogram(x=s, nbinsx=40, marker_color="#667eea",
                             opacity=0.75, name="Histogram"),
                row=1, col=1,
            )
            fig.add_vline(x=float(s.mean()),   line_dash="dash",
                          line_color="#e74c3c", annotation_text="Mean")
            fig.add_vline(x=float(s.median()), line_dash="dash",
                          line_color="#27ae60", annotation_text="Median",
                          row=1, col=1)

            # Box
            fig.add_trace(
                go.Box(y=s, marker_color="#764ba2", boxmean="sd", name="Box"),
                row=1, col=2,
            )

            fig.update_layout(height=380, showlegend=False, plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

            sk = float(s.skew())
            ku = float(s.kurtosis())
            sk_dir = "right (positive) skew — tail extends to the right" if sk > 0 \
                     else "left (negative) skew — tail extends to the left"
            st.caption(
                f"**{sel_col}** — skewness: `{sk:.3f}` ({sk_dir}) · "
                f"kurtosis: `{ku:.3f}` "
                f"({'heavy tails / outlier-prone' if ku > 3 else 'light tails'})"
            )


def _render_categorical_health(df: pd.DataFrame):
    """
    SUB-STEP 5 — Categorical Variable Health
    Purpose   : Identify encoding problems, near-ID columns, and class imbalance.
    Metrics   : Cardinality ratio, mode dominance, rare category counts.
    Risk flags: Cardinality > 90%, mode dominance > 90%, rare values < 1%.
    """
    _section("⑤ Categorical Variable Health")
    _why_box(
        "Categorical variables that are nearly unique (high cardinality) cannot be "
        "one-hot encoded and will cause memory explosions. Variables with one dominant "
        "category (mode dominance > 90%) provide almost no discriminating power. "
        "Rare categories will appear in only a handful of rows, making any frequency-based "
        "analysis meaningless and causing test-set generalisation failures."
    )

    c = _compute_categorical_health(df)
    if not c["cols"]:
        st.info("No categorical (text) columns in the dataset.")
        return

    cat_df = c["stats_df"]

    display = cat_df.copy()
    display["Cardinality Signal"] = cat_df["Card Risk"].apply(
        lambda r: "🔴 Near-ID (>90%)"   if r == "high"   else
                  "🟡 High (>50%)"       if r == "medium" else
                  "⚠️ Constant"          if r == "low"    else "🟢 OK"
    )
    display["Mode Dominance"] = cat_df["Mode Freq %"].apply(
        lambda x: "🔴 Dominant (>90%)" if x >= 90 else
                  "🟡 Skewed (>70%)"    if x >= 70 else "🟢 Balanced"
    )

    st.dataframe(
        display[["Column", "Unique Values", "Cardinality %",
                 "Mode", "Mode Freq %", "Missing %",
                 "Cardinality Signal", "Mode Dominance"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Cardinality %": st.column_config.ProgressColumn(
                "Cardinality %", format="%.1f%%", min_value=0, max_value=100
            ),
            "Mode Freq %": st.column_config.ProgressColumn(
                "Mode Freq %", format="%.1f%%", min_value=0, max_value=100
            ),
            "Missing %": st.column_config.ProgressColumn(
                "Missing %", format="%.1f%%", min_value=0, max_value=100
            ),
        },
    )

    # Rare value drill-down
    sel_cat = st.selectbox("Inspect value distribution for:", c["cols"], key="qa_cat_col")
    vc = df[sel_cat].value_counts()

    col_a, col_b = st.columns(2)
    with col_a:
        top_n = min(15, len(vc))
        fig = go.Figure(go.Bar(
            x=vc.head(top_n).index.astype(str),
            y=vc.head(top_n).values,
            marker_color="#667eea",
            text=vc.head(top_n).values,
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Top {top_n} values — {sel_cat}",
            height=340, xaxis_tickangle=-30, plot_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        rare_threshold = max(1, int(len(df) * 0.01))
        rare_vals = vc[vc <= rare_threshold]
        st.metric("Rare values (≤1% frequency)", len(rare_vals))
        if len(rare_vals) > 0:
            _insight_box(
                f"🟡 {len(rare_vals)} rare categories in '{sel_cat}'",
                f"{len(rare_vals)} values each appear in ≤{rare_threshold} rows. "
                "These may be typos, data inconsistencies, or genuinely rare segments. "
                "They will have unreliable statistics when analysed independently.",
                "warn",
            )
            with st.expander("Show rare values"):
                st.dataframe(
                    pd.DataFrame({"Value": rare_vals.index.astype(str),
                                  "Count": rare_vals.values}),
                    use_container_width=True, hide_index=True,
                )

    _decision_box([
        "Should near-ID columns (>90% unique) be dropped or used only for row identification?",
        "Should rare categories be grouped into an 'Other' bucket?",
        "Do dominant categories (>90% frequency) add any discriminating power to your analysis?",
        "Are cardinality levels compatible with your intended encoding strategy (one-hot, target encoding)?",
    ])


def _render_correlation_health(df: pd.DataFrame):
    """
    SUB-STEP 6 — Multicollinearity & Correlation Structure
    Purpose   : Detect variable pairs that carry redundant information.
    Metrics   : Pearson correlation matrix, high-|r| pair list.
    Risk flags: |r| > 0.85 between any two predictors.
    """
    _section("⑥ Multicollinearity & Correlation Structure")
    _why_box(
        "When two numeric variables are highly correlated, they contain nearly the same "
        "information. In regression models this causes <b>multicollinearity</b> — "
        "coefficient estimates become unstable, standard errors inflate, and feature "
        "importance scores become meaningless. Even in non-model analysis, high correlation "
        "means you may be overstating the evidence by treating two redundant signals as independent."
    )

    corr = _compute_correlation_health(df)

    if len(corr["cols"]) < 2:
        st.info("Need at least 2 numeric columns for correlation analysis.")
        return

    tab_heat, tab_pairs = st.tabs(["Correlation Heatmap", "High-Correlation Pairs"])

    with tab_heat:
        mat = corr["matrix"]
        fig = go.Figure(go.Heatmap(
            z=mat.values,
            x=mat.columns.tolist(),
            y=mat.columns.tolist(),
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=mat.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 9},
            colorbar=dict(title="r"),
        ))
        fig.update_layout(
            title="Pearson Correlation Matrix",
            height=max(400, len(corr["cols"]) * 40),
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_pairs:
        pairs = corr["high_pairs"]
        if not pairs:
            _insight_box(
                "✅ No high correlation pairs",
                f"No numeric variable pairs exceed |r| = {CORR_HIGH}.",
                "success",
            )
        else:
            _insight_box(
                f"🟡 {len(pairs)} high-correlation pair(s) detected",
                f"These pairs have |r| ≥ {CORR_HIGH} and may carry redundant information.",
                "warn",
            )
            pairs_df = pd.DataFrame(pairs)
            st.dataframe(
                pairs_df[["Column A", "Column B", "Pearson r", "Direction"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Pearson r": st.column_config.NumberColumn(format="%.3f"),
                },
            )

            # Interactive scatter for selected pair
            if len(pairs) > 0:
                pair_labels = [f"{p['Column A']} vs {p['Column B']} (r={p['Pearson r']:.3f})"
                               for p in pairs]
                sel_pair = st.selectbox("Inspect pair", range(len(pairs)),
                                        format_func=lambda i: pair_labels[i],
                                        key="qa_corr_pair")
                ca, cb = pairs[sel_pair]["Column A"], pairs[sel_pair]["Column B"]

                fig2 = px.scatter(
                    df, x=ca, y=cb, opacity=0.5,
                    trendline="ols",
                    title=f"{ca} vs {cb}",
                    color_discrete_sequence=["#667eea"],
                )
                fig2.update_layout(height=360, plot_bgcolor="white")
                st.plotly_chart(fig2, use_container_width=True)

    _decision_box([
        "Should one variable in each high-correlation pair be removed to reduce redundancy?",
        "Is the correlation causal, coincidental, or structural (e.g. a variable derived from another)?",
        "If you intend to run regression, will you use VIF (Variance Inflation Factor) to confirm multicollinearity?",
        "Could you combine correlated columns into a composite index instead of dropping one?",
    ])


def _render_structural_flags(df: pd.DataFrame):
    """
    SUB-STEP 7 — Structural Integrity Flags
    Purpose   : Surface remaining dataset-level concerns that don't fit other categories.
    Metrics   : Constant columns, near-constant, zero-variance, sample size adequacy.
    Risk flags: < 30 rows per variable (rule of thumb for regression), constant cols.
    """
    _section("⑦ Structural Integrity & Sample Size")
    _why_box(
        "Even a clean dataset can be analytically fragile if its structure violates "
        "basic requirements: too few rows relative to the number of variables, columns "
        "that never vary (adding no information), or columns that are almost constant "
        "(adding almost no information but inflating model complexity)."
    )

    n_rows, n_cols = df.shape
    num_cols       = df.select_dtypes(include=np.number).columns.tolist()
    constant_cols  = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    near_const     = [c for c in num_cols
                      if c not in constant_cols and
                      df[c].dropna().std() / (abs(df[c].dropna().mean()) + 1e-9) < 0.01]
    # Sample adequacy: rule of thumb is ≥30 rows per feature for regression
    rows_per_col   = n_rows / max(n_cols, 1)
    sample_ok      = rows_per_col >= 30

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Rows",              f"{n_rows:,}")
    sc2.metric("Columns",           f"{n_cols}")
    sc3.metric("Rows per Column",   f"{rows_per_col:.0f}",
               delta="OK" if sample_ok else "Low — see below",
               delta_color="normal" if sample_ok else "inverse")
    sc4.metric("Constant Columns",  f"{len(constant_cols)}")

    if constant_cols:
        _insight_box(
            f"🔴 {len(constant_cols)} constant column(s): {', '.join(constant_cols)}",
            "These columns have exactly one unique value. They carry zero information "
            "and will cause errors in many algorithms (e.g. division by zero in normalisation).",
            "danger",
        )

    if near_const:
        _insight_box(
            f"🟡 {len(near_const)} near-constant column(s): {', '.join(near_const[:5])}",
            "These numeric columns have a coefficient of variation below 1% — "
            "virtually no variation across rows. They contribute almost no predictive "
            "or discriminating power.",
            "warn",
        )

    if not sample_ok:
        _insight_box(
            f"🟡 Low sample density: {rows_per_col:.0f} rows per column",
            f"With {n_rows:,} rows and {n_cols} columns, you have less than the commonly "
            f"recommended 30 observations per variable. Statistical estimates will have "
            f"wide confidence intervals, and overfitting risk in ML models will be high.",
            "warn",
        )
    else:
        _insight_box(
            f"✅ Sample density: {rows_per_col:.0f} rows per column",
            "Sample size is adequate relative to the number of variables.",
            "success",
        )

    _decision_box([
        "Should constant columns be dropped before proceeding?",
        "Is the sample size sufficient for the specific analysis you intend (regression, clustering, ML)?",
        "If rows-per-column is low, can you collect more data or reduce the number of variables?",
        "Are near-constant columns genuinely uninformative, or do they capture a rare-but-important signal?",
    ])


# ═══════════════════════════════════════════════════════════════════════════
# MAIN STAGE RENDERER — public entry point
# ═══════════════════════════════════════════════════════════════════════════

def render_quality_assessment_stage():
    """
    Stage 5 — Data Quality & Analytical Readiness Assessment.

    Wiring:
        In dataset_manager.py render_dataset_manager(), after the
        row_integrity routing block, add:

            if not WorkflowState.is_done("quality_assessed"):
                from analytiqal_quality import render_quality_assessment_stage
                render_quality_assessment_stage()
                return

    This function:
        • Never modifies st.session_state.df
        • Never calls TransformationEngine
        • Never calls HistoryManager.push()
        • Only reads st.session_state.df (read-only snapshot)
        • All computation is @st.cache_data-memoised
    """
    df = st.session_state.get("df")
    if df is None:
        st.warning("No dataset loaded — return to the Import stage.")
        return

    # ── Stage header ─────────────────────────────────────────────────────
    st.markdown(
        f"<div style='background:linear-gradient({_BRAND_GRADIENT});"
        f"padding:1.1rem 1.6rem;border-radius:10px;margin-bottom:18px;'>"
        f"<h2 style='color:white;margin:0;font-size:1.3rem;font-weight:700;'>"
        f"🔬 Step 5 — Data Quality & Analytical Readiness Assessment</h2>"
        f"<p style='color:rgba(255,255,255,0.80);margin:4px 0 0;font-size:0.83rem;'>"
        f"Read-only assessment · No data is modified · You decide what to do next"
        f"</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='background:{_D['c_surface']};border:1px solid {_D['c_border']};"
        f"border-radius:8px;padding:10px 16px;margin-bottom:18px;'>"
        f"<span style='font-size:{_D['t_sm']};color:{_D['c_text_muted']};'>"
        f"This stage evaluates whether your dataset is <b>analytically trustworthy</b> "
        f"before you run statistics or build models. It flags risks, explains why they "
        f"matter, and asks questions — the decisions are yours.</span></div>",
        unsafe_allow_html=True,
    )

    # ── Dataset snapshot ─────────────────────────────────────────────────
    n_rows, n_cols = df.shape
    num_c   = len(df.select_dtypes(include=np.number).columns)
    cat_c   = len(df.select_dtypes(include="object").columns)
    miss_pct = df.isnull().mean().mean() * 100

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows",     f"{n_rows:,}")
    m2.metric("Columns",  f"{n_cols}")
    m3.metric("Numeric",  f"{num_c}")
    m4.metric("Text",     f"{cat_c}")
    m5.metric("Missing",  f"{miss_pct:.1f}%")

    st.markdown(
        f"<div style='height:1px;background:{_D['c_border']};margin:14px 0;'></div>",
        unsafe_allow_html=True,
    )

    # ── Seven sub-steps in one scrollable page ───────────────────────────
    with st.spinner("Computing quality metrics…"):
        _render_readiness_score(df)
        _render_missingness(df)
        _render_duplicates(df)
        _render_numeric_health(df)
        _render_categorical_health(df)
        _render_correlation_health(df)
        _render_structural_flags(df)

    # ── Summary flag panel ───────────────────────────────────────────────
    st.markdown(
        f"<div style='height:1px;background:{_D['c_border']};margin:28px 0 16px;'></div>",
        unsafe_allow_html=True,
    )
    _section("Assessment Summary")

    r    = _compute_analytical_readiness(df)
    m    = _compute_missingness(df)
    d    = _compute_duplicates(df)
    nh   = _compute_numeric_health(df)
    corr = _compute_correlation_health(df)

    flags = []
    if m["overall_pct"] >= 20:
        flags.append(("🔴", "Critical missingness", f"{m['overall_pct']:.1f}% of all cells are empty"))
    if m["cols_critical"]:
        flags.append(("🔴", "Critically missing columns", f"{', '.join(m['cols_critical'][:4])}"))
    if d["risk"] in ("critical", "medium"):
        flags.append(("🟡", "Significant duplicates", f"{d['n_full']:,} rows ({d['pct_full']:.1f}%)"))
    if not nh["stats_df"].empty:
        skewed = nh["stats_df"][nh["stats_df"]["Skew Risk"] == "high"]["Column"].tolist()
        if skewed:
            flags.append(("🟡", "Skewed distributions", f"{len(skewed)} column(s) need transformation"))
    if corr["high_pairs"]:
        flags.append(("🟡", "Multicollinearity", f"{len(corr['high_pairs'])} high-correlation pair(s)"))

    if not flags:
        _insight_box(
            "✅ No critical flags — dataset is analytically ready",
            f"Readiness score: {r['total']}/100 (Grade {r['grade']}). "
            "Proceed with confidence.",
            "success",
        )
    else:
        for icon, label, detail in flags:
            st.markdown(
                f"<div style='display:flex;gap:10px;align-items:flex-start;"
                f"padding:8px 12px;border-left:3px solid "
                f"{'#e74c3c' if icon == '🔴' else '#f39c12'};"
                f"margin:4px 0;background:{_D['c_surface']};border-radius:0 6px 6px 0;'>"
                f"<span style='font-size:1rem;'>{icon}</span>"
                f"<div><span style='font-weight:700;font-size:{_D['t_sm']};'>{label}</span> "
                f"<span style='color:{_D['c_text_muted']};font-size:{_D['t_sm']};'>— {detail}</span>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    # ── Continue / Go Back ───────────────────────────────────────────────
    st.markdown(
        f"<div style='height:1px;background:{_D['c_border']};margin:20px 0;'></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='background:{_D['c_surface']};border:1px solid {_D['c_border']};"
        f"border-radius:8px;padding:10px 16px;margin-bottom:12px;'>"
        f"<span style='font-size:{_D['t_sm']};color:{_D['c_text_muted']};'>"
        f"ℹ You have reviewed the quality assessment. If you want to act on any of these "
        f"findings — remove duplicates, filter rows, fix types — go back to <b>Row Integrity</b> "
        f"or <b>Column Restructuring</b>. When you are satisfied, continue to <b>Reshape</b>."
        f"</span></div>",
        unsafe_allow_html=True,
    )

    ca, cb = st.columns(2)
    with ca:
        if st.button(
            "✅ I've reviewed the assessment — Continue to Reshape →",
            type="primary",
            use_container_width=True,
            key="qa_continue",
        ):
            # This is the ONLY place WorkflowState.done is called
            try:
                from dataset_manager import WorkflowState
                WorkflowState.done("quality_assessed")
                st.rerun()
            except ImportError:
                st.session_state["workflow"]["quality_assessed"] = True
                st.rerun()

    with cb:
        if st.button(
            "← Back to Row Integrity",
            use_container_width=True,
            key="qa_back",
        ):
            try:
                from dataset_manager import WorkflowState
                WorkflowState.go_back()
                st.rerun()
            except ImportError:
                st.session_state["workflow"]["row_integrity"] = False
                st.rerun()