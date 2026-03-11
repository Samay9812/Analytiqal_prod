"""
Visualizations Page — Fixed & Enhanced
ANALYTIQAL Dataset Manager

Fixes applied vs previous version:
  1. Large-dataset safety guard — auto-samples rows for heavy chart types,
     warns the user and shows exact sample size used.
  2. Scatter trendline graceful fallback — checks for statsmodels before
     requesting OLS; shows a clear install hint if missing.
  3. Time-series no longer mutates st.session_state['df'] — works on a
     local copy only.
  4. Density Plot fill colour fixed — hex colours are converted to rgba()
     correctly instead of the broken rgb-string replacement.
  5. Chart state persistence — last generated figure is stored in
     session_state so it survives slider/selectbox re-runs without
     needing another Generate click.
  6. y_col / x_col safe initialisation in bivariate — variables always
     assigned before use.

Additions vs previous version:
  A. Axis controls — log-scale toggles and manual min/max range inputs on
     numeric axes for all applicable charts.
  B. Inline reference line annotation — add a horizontal or vertical
     threshold line with a custom label to any chart.
  C. Pre-chart data filter — slice by a categorical column value or clip
     numeric outliers (IQR-based) before charting, without touching the
     working dataset.
  D. Summary statistics panel — mean / median / std / min / max shown
     beside every chart.
  E. Saved charts gallery — pin any chart to a session gallery; compare
     up to 6 saved charts side-by-side.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
from io import BytesIO
import base64


# ============================================================================
# COLOUR SCHEME DEFINITIONS
# ============================================================================

COLOR_SCHEMES = {
    "Plotly":  ["#636EFA","#EF553B","#00CC96","#AB63FA","#FFA15A","#19D3F3","#FF6692","#B6E880"],
    "Viridis": ["#440154","#31688e","#35b779","#fde725","#21918c","#5ec962","#a8db34","#fde725"],
    "Blues":   ["#084594","#2171b5","#4292c6","#6baed6","#9ecae1","#c6dbef","#deebf7","#f7fbff"],
    "Reds":    ["#67000d","#a50f15","#cb181d","#ef3b2c","#fb6a4a","#fc9272","#fcbba1","#fff5f0"],
    "Greens":  ["#00441b","#006d2c","#238b45","#41ab5d","#74c476","#a1d99b","#c7e9c0","#f7fcf5"],
    "Rainbow": ["#e41a1c","#ff7f00","#ffff33","#4daf4a","#377eb8","#984ea3","#a65628","#f781bf"],
    "Plasma":  ["#0d0887","#5302a3","#8b0aa5","#b83289","#db5c68","#f48849","#febc2a","#f0f921"],
}

CONTINUOUS_SCALES = {
    "Viridis": "Viridis", "Blues": "Blues", "Reds": "Reds",
    "Greens": "Greens", "Rainbow": "Rainbow", "Plasma": "Plasma",
    "Plotly": "Viridis",
}

# Chart types that render every row and need the large-dataset guard
HEAVY_CHART_TYPES = {
    "Scatter Plot", "Strip Plot", "Joint Plot",
    "Scatter Matrix", "3D Scatter Plot", "Parallel Coordinates",
}
LARGE_DATASET_THRESHOLD = 50_000
SAMPLE_SIZE = 20_000


def get_discrete_colors(scheme: str) -> list:
    return COLOR_SCHEMES.get(scheme, COLOR_SCHEMES["Plotly"])


def get_continuous_scale(scheme: str) -> str:
    return CONTINUOUS_SCALES.get(scheme, "Viridis")


def _hex_to_rgba(hex_color: str, alpha: float = 0.25) -> str:
    """Convert a hex color string like #636EFA to rgba(99,110,250,0.25)."""
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return f"rgba(100,100,100,{alpha})"


def _swatches_html(colors: list, n: int = 6) -> str:
    return "".join(
        f"<span style='display:inline-block;width:18px;height:18px;"
        f"background:{c};border-radius:3px;margin-right:2px;'></span>"
        for c in colors[:n]
    )


# ============================================================================
# LARGE-DATASET SAFETY GUARD
# ============================================================================

def _safe_sample(df: pd.DataFrame, chart_type: str, key_suffix: str) -> pd.DataFrame:
    """
    For chart types that render every row, sample the dataframe if it is
    large. Shows an info banner and lets the user override the sample size.
    Returns the (possibly sampled) dataframe.
    """
    if chart_type not in HEAVY_CHART_TYPES or len(df) <= LARGE_DATASET_THRESHOLD:
        return df

    with st.expander(
        f"⚠️ Large dataset detected ({len(df):,} rows) — sampling applied", expanded=True
    ):
        st.warning(
            f"**{chart_type}** renders every data point. "
            f"Plotting {len(df):,} rows will be very slow or crash the browser."
        )
        sample_n = st.slider(
            "Rows to plot",
            min_value=1_000,
            max_value=min(len(df), 100_000),
            value=SAMPLE_SIZE,
            step=1_000,
            key=f"sample_slider_{key_suffix}",
        )
        seed = st.number_input("Random seed", value=42, min_value=0, step=1,
                               key=f"sample_seed_{key_suffix}")
        st.info(
            f"Showing a random sample of **{sample_n:,}** rows "
            f"({sample_n / len(df) * 100:.1f}% of data). "
            f"Summary statistics below the chart are computed on the **full** dataset."
        )
        return df.sample(n=sample_n, random_state=int(seed))

    return df


# ============================================================================
# AXIS CONTROLS
# ============================================================================

def _axis_controls(key_prefix: str, axes: list = ("x", "y")) -> dict:
    """
    Render log-scale checkboxes and optional min/max range inputs.
    Returns a dict with keys like x_log, x_min, x_max, y_log, y_min, y_max.
    """
    opts = {}
    with st.expander("🔧 Axis Controls", expanded=False):
        cols = st.columns(len(axes) * 3)
        for i, ax in enumerate(axes):
            base = i * 3
            opts[f"{ax}_log"] = cols[base].checkbox(
                f"{ax.upper()} log scale", value=False, key=f"{key_prefix}_{ax}_log"
            )
            opts[f"{ax}_min"] = cols[base + 1].text_input(
                f"{ax.upper()} min", value="", placeholder="auto", key=f"{key_prefix}_{ax}_min"
            )
            opts[f"{ax}_max"] = cols[base + 2].text_input(
                f"{ax.upper()} max", value="", placeholder="auto", key=f"{key_prefix}_{ax}_max"
            )
    return opts


def _apply_axis_opts(fig: go.Figure, opts: dict) -> go.Figure:
    """Apply axis options dict to a plotly figure."""
    updates = {}
    for ax in ("x", "y", "z"):
        if opts.get(f"{ax}_log"):
            updates[f"{ax}axis_type"] = "log"
        mn = opts.get(f"{ax}_min", "")
        mx = opts.get(f"{ax}_max", "")
        rng = []
        try:
            if mn:
                rng.append(float(mn))
            if mx:
                rng.append(float(mx))
        except ValueError:
            pass
        if len(rng) == 2:
            if opts.get(f"{ax}_log"):
                rng = [np.log10(v) if v > 0 else 0 for v in rng]
            updates[f"{ax}axis_range"] = rng
    if updates:
        fig.update_layout(**updates)
    return fig


# ============================================================================
# REFERENCE LINE ANNOTATION
# ============================================================================

def _reference_line_ui(key_prefix: str) -> Optional[dict]:
    """UI for adding one reference line. Returns config dict or None."""
    with st.expander("📏 Add Reference Line (optional)", expanded=False):
        col1, col2, col3, col4 = st.columns([1, 1, 2, 2])
        enabled = col1.checkbox("Enable", value=False, key=f"{key_prefix}_refline_on")
        if not enabled:
            return None
        direction = col2.radio("Direction", ["Horizontal", "Vertical"],
                               key=f"{key_prefix}_refline_dir")
        try:
            value = float(col3.text_input("Value", value="0",
                                          key=f"{key_prefix}_refline_val"))
        except ValueError:
            value = 0.0
        label = col4.text_input("Label", value="Threshold",
                                key=f"{key_prefix}_refline_lbl")
        color = col4.color_picker("Line colour", value="#EF553B",
                                  key=f"{key_prefix}_refline_col")
        return {"direction": direction, "value": value, "label": label, "color": color}


def _apply_reference_line(fig: go.Figure, ref: Optional[dict]) -> go.Figure:
    if not ref:
        return fig
    if ref["direction"] == "Horizontal":
        fig.add_hline(
            y=ref["value"],
            line_dash="dash",
            line_color=ref["color"],
            annotation_text=ref["label"],
            annotation_position="top right",
        )
    else:
        fig.add_vline(
            x=ref["value"],
            line_dash="dash",
            line_color=ref["color"],
            annotation_text=ref["label"],
            annotation_position="top right",
        )
    return fig


# ============================================================================
# PRE-CHART DATA FILTER
# ============================================================================

def _data_filter_ui(df: pd.DataFrame, col_types: dict, key_prefix: str) -> pd.DataFrame:
    """
    Let the user slice the data before charting without touching session state.
    Returns filtered dataframe.
    """
    with st.expander("🔍 Pre-chart Data Filter (optional)", expanded=False):
        filtered = df.copy()
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Filter by category**")
            cat_col = st.selectbox(
                "Categorical column", [None] + col_types["categorical"],
                key=f"{key_prefix}_filter_cat_col"
            )
            if cat_col:
                unique_vals = sorted(df[cat_col].dropna().unique().tolist())
                selected_vals = st.multiselect(
                    f"Keep values", unique_vals, default=unique_vals,
                    key=f"{key_prefix}_filter_cat_vals"
                )
                if selected_vals:
                    filtered = filtered[filtered[cat_col].isin(selected_vals)]

        with col2:
            st.markdown("**Remove numeric outliers (IQR)**")
            num_col = st.selectbox(
                "Numeric column", [None] + col_types["numeric"],
                key=f"{key_prefix}_filter_num_col"
            )
            if num_col:
                multiplier = st.slider(
                    "IQR multiplier", 1.0, 3.0, 1.5, 0.1,
                    key=f"{key_prefix}_filter_iqr"
                )
                q1 = filtered[num_col].quantile(0.25)
                q3 = filtered[num_col].quantile(0.75)
                iqr = q3 - q1
                before = len(filtered)
                filtered = filtered[
                    (filtered[num_col] >= q1 - multiplier * iqr) &
                    (filtered[num_col] <= q3 + multiplier * iqr)
                ]
                removed = before - len(filtered)
                if removed:
                    st.info(f"Removed {removed:,} outlier rows from '{num_col}'")

        st.caption(f"Rows after filter: **{len(filtered):,}** of {len(df):,}")
    return filtered


# ============================================================================
# SUMMARY STATISTICS PANEL
# ============================================================================

def _summary_stats(df_full: pd.DataFrame, df_plot: pd.DataFrame,
                   numeric_cols: list, key_prefix: str):
    """Show summary statistics for numeric columns used in the chart."""
    if not numeric_cols:
        return
    with st.expander("📊 Summary Statistics", expanded=False):
        tab_full, tab_plot = st.tabs(["Full Dataset", "Plotted Data"])
        for tab, data, label in (
            (tab_full, df_full, "full"),
            (tab_plot, df_plot, "plotted"),
        ):
            with tab:
                cols_present = [c for c in numeric_cols if c in data.columns]
                if not cols_present:
                    st.info("No numeric columns to summarise")
                    continue
                stats = data[cols_present].agg(["mean", "median", "std", "min", "max"])
                stats.index = ["Mean", "Median", "Std Dev", "Min", "Max"]
                st.dataframe(stats.round(4), use_container_width=True)


# ============================================================================
# SAVED CHARTS GALLERY
# ============================================================================

def _init_gallery():
    if "chart_gallery" not in st.session_state:
        st.session_state["chart_gallery"] = []


def _save_to_gallery(fig: go.Figure, name: str):
    _init_gallery()
    gallery = st.session_state["chart_gallery"]
    if len(gallery) >= 6:
        gallery.pop(0)
    gallery.append({"name": name, "fig": fig})
    st.success(f"✅ Saved '{name}' to gallery ({len(gallery)}/6)")


def _render_gallery():
    _init_gallery()
    gallery = st.session_state["chart_gallery"]
    if not gallery:
        st.info("No charts saved yet. Click 'Pin to Gallery' after generating a chart.")
        return

    st.markdown(f"### 🖼️ Saved Charts ({len(gallery)}/6)")
    if st.button("🗑️ Clear Gallery", key="gallery_clear"):
        st.session_state["chart_gallery"] = []
        st.rerun()

    cols_per_row = 2
    for i in range(0, len(gallery), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for j, col in enumerate(row_cols):
            idx = i + j
            if idx < len(gallery):
                entry = gallery[idx]
                with col:
                    st.markdown(f"**{idx + 1}. {entry['name']}**")
                    st.plotly_chart(entry["fig"], key=f"gallery_chart_{idx}", use_container_width=True)
                    if st.button("🗑️ Remove", key=f"gallery_del_{idx}"):
                        gallery.pop(idx)
                        st.rerun()


# ============================================================================
# DOWNLOAD BUTTONS
# ============================================================================

def _add_download_buttons(fig: go.Figure, filename_base: str, key_suffix: str = ""):
    st.markdown("---")
    st.markdown("### 💾 Export")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        try:
            img_bytes = fig.to_image(format="png", scale=2)
            st.download_button(
                "📥 PNG", img_bytes,
                f"{filename_base}.png", "image/png",
                use_container_width=True, key=f"dl_png_{key_suffix}"
            )
        except Exception:
            st.caption("PNG: install kaleido")

    with col2:
        st.download_button(
            "📥 HTML", fig.to_html(),
            f"{filename_base}.html", "text/html",
            use_container_width=True, key=f"dl_html_{key_suffix}"
        )

    with col3:
        st.download_button(
            "📥 JSON", fig.to_json(),
            f"{filename_base}.json", "application/json",
            use_container_width=True, key=f"dl_json_{key_suffix}"
        )

    with col4:
        pin_name = st.text_input(
            "Pin name", value=filename_base, label_visibility="collapsed",
            placeholder="Chart name…", key=f"pin_name_{key_suffix}"
        )
        if st.button("📌 Pin to Gallery", use_container_width=True, key=f"pin_btn_{key_suffix}"):
            _save_to_gallery(fig, pin_name or filename_base)


# ============================================================================
# MAIN PAGE
# ============================================================================

def render_visualizations_page():
    st.markdown("""
        <div style='background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
                    padding: 2rem; border-radius: 12px; margin-bottom: 2rem;'>
            <h1 style='color: white; margin: 0; font-size: 2.5rem;'>📊 Data Visualizations</h1>
            <p style='color: rgba(255,255,255,0.9); margin-top: 0.5rem; font-size: 1.1rem;'>
                Create interactive visualizations to explore your data
            </p>
        </div>
    """, unsafe_allow_html=True)

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.info("📂 **No dataset loaded.** Load a dataset from the 🏠 Dataset page first.")
        return

    from utils_robust import get_column_types
    col_types = get_column_types(df)

    # Dashboard metrics
    st.markdown("### 📊 Dataset Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", f"{len(df):,}")
    c2.metric("Numeric Columns", len(col_types["numeric"]))
    c3.metric("Categorical Columns", len(col_types["categorical"]))
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    c4.metric("DateTime Columns", len(dt_cols))

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Univariate",
        "📊 Bivariate",
        "🔥 Multivariate",
        "📉 Time Series",
        "🖼️ Gallery",
    ])

    with tab1:
        render_univariate_charts(df, col_types)
    with tab2:
        render_bivariate_charts(df, col_types)
    with tab3:
        render_multivariate_charts(df, col_types)
    with tab4:
        render_timeseries_charts(df, col_types)
    with tab5:
        _render_gallery()


# ============================================================================
# UNIVARIATE
# ============================================================================

def render_univariate_charts(df: pd.DataFrame, col_types: Dict):
    st.markdown("### 📈 Univariate Analysis")
    st.caption("Analyze single variables to understand their distribution")

    with st.expander("ℹ️ Chart guide"):
        st.markdown("""
        **Numeric:** Histogram, Box Plot, Violin Plot, Density Plot, QQ Plot

        **Categorical:** Bar Chart, Pie Chart, Donut Chart, Treemap
        """)

    col1, col2 = st.columns([2, 1])
    with col1:
        chart_type = st.selectbox(
            "Chart type",
            ["Histogram", "Box Plot", "Violin Plot", "Density Plot", "QQ Plot",
             "Bar Chart", "Pie Chart", "Donut Chart", "Treemap"],
            key="uni_chart_type",
        )

    numeric_charts = {"Histogram", "Box Plot", "Violin Plot", "Density Plot", "QQ Plot"}

    if chart_type in numeric_charts:
        if not col_types["numeric"]:
            st.warning("⚠️ No numeric columns"); return
        avail = col_types["numeric"]; col_label = "Numeric column"
    else:
        if not col_types["categorical"]:
            st.warning("⚠️ No categorical columns"); return
        avail = col_types["categorical"]; col_label = "Categorical column"

    with col2:
        target_col = st.selectbox(col_label, avail, key="uni_target_col")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        title = st.text_input("Title", value=f"{chart_type} — {target_col}", key="uni_title")
    with c2:
        cs = st.selectbox("Colour scheme", list(COLOR_SCHEMES), key="uni_cs")
        st.markdown(_swatches_html(get_discrete_colors(cs)), unsafe_allow_html=True)
    with c3:
        height = st.slider("Height", 300, 800, 500, 50, key="uni_height")

    # Type-specific options
    bins, show_kde, top_n = 30, False, 10
    if chart_type == "Histogram":
        bins = st.slider("Bins", 10, 100, 30, key="uni_bins")
        show_kde = st.checkbox("Density curve overlay", value=True, key="uni_kde")
    elif chart_type in {"Bar Chart", "Pie Chart", "Donut Chart", "Treemap"}:
        top_n = st.slider("Top N categories", 5, 50, 10, key="uni_topn")

    # Axis controls (numeric only)
    axis_opts = {}
    if chart_type in {"Histogram", "Bar Chart"}:
        axis_opts = _axis_controls("uni", axes=["y"])
    elif chart_type in {"Box Plot", "Violin Plot", "Density Plot"}:
        axis_opts = _axis_controls("uni", axes=["y"])

    # Reference line
    ref = _reference_line_ui("uni")

    # Pre-chart filter
    df_plot = _data_filter_ui(df, col_types, "uni")
    df_plot = _safe_sample(df_plot, chart_type, "uni")

    st.markdown("---")
    if st.button("📊 Generate Chart", key="uni_gen", type="primary", use_container_width=True):
        try:
            fig = _create_univariate_chart(
                df_plot, target_col, chart_type, title, cs, height,
                bins=bins, show_kde=show_kde, top_n=top_n,
            )
            if fig:
                fig = _apply_axis_opts(fig, axis_opts)
                fig = _apply_reference_line(fig, ref)
                st.session_state["uni_last_fig"] = fig
                st.session_state["uni_last_name"] = f"{chart_type}_{target_col}"
        except Exception as e:
            st.error(f"Chart error: {e}")
            import traceback; st.code(traceback.format_exc())

    # Persistent display
    if st.session_state.get("uni_last_fig"):
        fig = st.session_state["uni_last_fig"]
        st.plotly_chart(fig, key = "u_l_f", use_container_width=True)
        _summary_stats(df, df_plot, [target_col] if target_col in col_types["numeric"] else [], "uni")
        _add_download_buttons(fig, st.session_state.get("uni_last_name", "chart"), "uni")


def _create_univariate_chart(df, col, chart_type, title, cs, height, **kw):
    disc = get_discrete_colors(cs)
    cont = get_continuous_scale(cs)
    primary = disc[0]

    if chart_type == "Histogram":
        fig = px.histogram(df, x=col, nbins=kw.get("bins", 30), title=title,
                           color_discrete_sequence=[primary])
        if kw.get("show_kde"):
            try:
                from scipy.stats import gaussian_kde
                data = df[col].dropna()
                kde = gaussian_kde(data)
                xs = np.linspace(data.min(), data.max(), 200)
                hist, edges = np.histogram(data, bins=kw.get("bins", 30))
                bw = edges[1] - edges[0]
                fig.add_trace(go.Scatter(
                    x=xs, y=kde(xs) * len(data) * bw,
                    mode="lines", name="Density",
                    line=dict(color=disc[1] if len(disc) > 1 else "#EF553B", width=2),
                ))
            except ImportError:
                st.caption("⚠️ scipy not installed — density curve skipped")

    elif chart_type == "Box Plot":
        fig = px.box(df, y=col, title=title, color_discrete_sequence=[primary])

    elif chart_type == "Violin Plot":
        fig = px.violin(df, y=col, box=True, points="outliers", title=title,
                        color_discrete_sequence=[primary])

    elif chart_type == "Density Plot":
        try:
            from scipy.stats import gaussian_kde
            data = df[col].dropna()
            kde = gaussian_kde(data)
            xs = np.linspace(data.min(), data.max(), 200)
            fill_rgba = _hex_to_rgba(primary, 0.25)          # FIX: proper rgba conversion
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=xs, y=kde(xs),
                mode="lines", fill="tozeroy", name="Density",
                line=dict(color=primary),
                fillcolor=fill_rgba,
            ))
            fig.update_layout(title=title, xaxis_title=col, yaxis_title="Density")
        except ImportError:
            st.error("Density Plot requires scipy: pip install scipy")
            return None

    elif chart_type == "QQ Plot":
        try:
            from scipy import stats
            data = df[col].dropna()
            qq = stats.probplot(data, dist="norm")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=qq[0][0], y=qq[0][1], mode="markers", name="Data",
                marker=dict(color=primary, size=5),
            ))
            # Reference line
            slope, intercept, _ = qq[1]
            xs = np.array([qq[0][0][0], qq[0][0][-1]])
            fig.add_trace(go.Scatter(
                x=xs, y=intercept + slope * xs,
                mode="lines", name="Normal",
                line=dict(color=disc[1] if len(disc) > 1 else "#EF553B", dash="dash"),
            ))
            fig.update_layout(title=title, xaxis_title="Theoretical Quantiles",
                              yaxis_title="Sample Quantiles")
        except ImportError:
            st.error("QQ Plot requires scipy: pip install scipy")
            return None

    elif chart_type == "Bar Chart":
        vc = df[col].value_counts().head(kw.get("top_n", 10))
        fig = px.bar(x=vc.index, y=vc.values, title=title,
                     labels={"x": col, "y": "Count"},
                     color=vc.index, color_discrete_sequence=disc)
        fig.update_layout(showlegend=False)

    elif chart_type == "Pie Chart":
        vc = df[col].value_counts().head(kw.get("top_n", 10))
        fig = px.pie(values=vc.values, names=vc.index, title=title,
                     color_discrete_sequence=disc)

    elif chart_type == "Donut Chart":
        vc = df[col].value_counts().head(kw.get("top_n", 10))
        fig = px.pie(values=vc.values, names=vc.index, title=title,
                     hole=0.4, color_discrete_sequence=disc)

    elif chart_type == "Treemap":
        vc = df[col].value_counts().head(kw.get("top_n", 10))
        fig = px.treemap(names=vc.index, parents=[""] * len(vc),
                         values=vc.values, title=title,
                         color_discrete_sequence=disc)
    else:
        return None

    fig.update_layout(height=height)
    return fig


# ============================================================================
# BIVARIATE
# ============================================================================

def render_bivariate_charts(df: pd.DataFrame, col_types: Dict):
    st.markdown("### 📊 Bivariate Analysis")
    st.caption("Explore relationships between two variables")

    with st.expander("ℹ️ Chart guide"):
        st.markdown("""
        **Numeric × Numeric:** Scatter, Line, Hexbin, Joint

        **Categorical × Numeric:** Box by Category, Violin by Category, Strip, Bar by Category

        **Categorical × Categorical:** Grouped Bar, Stacked Bar, Count Heatmap, Sunburst
        """)

    chart_type = st.selectbox(
        "Chart type",
        ["Scatter Plot", "Line Plot", "Hexbin Plot", "Joint Plot",
         "Box Plot by Category", "Violin Plot by Category", "Strip Plot", "Bar Plot by Category",
         "Grouped Bar Chart", "Stacked Bar Chart", "Count Heatmap", "Sunburst Chart"],
        key="bi_chart_type",
    )

    num_num  = {"Scatter Plot", "Line Plot", "Hexbin Plot", "Joint Plot"}
    cat_num  = {"Box Plot by Category", "Violin Plot by Category", "Strip Plot", "Bar Plot by Category"}
    cat_cat  = {"Grouped Bar Chart", "Stacked Bar Chart", "Count Heatmap", "Sunburst Chart"}

    # Safe defaults so x_col/y_col are always defined
    x_col = col_types["numeric"][0] if col_types["numeric"] else (col_types["categorical"][0] if col_types["categorical"] else None)
    y_col = x_col
    color_col = None

    st.markdown("---")
    c1, c2, c3 = st.columns(3)

    if chart_type in num_num:
        if len(col_types["numeric"]) < 2:
            st.warning("⚠️ Need at least 2 numeric columns"); return
        with c1: x_col = st.selectbox("X axis (numeric)", col_types["numeric"], key="bi_x")
        with c2: y_col = st.selectbox("Y axis (numeric)", [c for c in col_types["numeric"] if c != x_col] or col_types["numeric"], key="bi_y")
        with c3: color_col = st.selectbox("Colour by (optional)", [None] + col_types["categorical"], key="bi_color")

    elif chart_type in cat_num:
        if not col_types["numeric"] or not col_types["categorical"]:
            st.warning("⚠️ Need both numeric and categorical columns"); return
        with c1: x_col = st.selectbox("Category (X)", col_types["categorical"], key="bi_x_cat")
        with c2: y_col = st.selectbox("Numeric (Y)", col_types["numeric"], key="bi_y_num")

    elif chart_type in cat_cat:
        if len(col_types["categorical"]) < 2:
            st.warning("⚠️ Need at least 2 categorical columns"); return
        with c1: x_col = st.selectbox("First category", col_types["categorical"], key="bi_x_c1")
        with c2: y_col = st.selectbox("Second category", [c for c in col_types["categorical"] if c != x_col] or col_types["categorical"], key="bi_y_c2")

    if x_col is None or y_col is None:
        st.warning("⚠️ Not enough columns for this chart type"); return

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1: title = st.text_input("Title", value=f"{chart_type}: {x_col} vs {y_col}", key="bi_title")
    with c2:
        cs = st.selectbox("Colour scheme", list(COLOR_SCHEMES), key="bi_cs")
        st.markdown(_swatches_html(get_discrete_colors(cs)), unsafe_allow_html=True)
    with c3: height = st.slider("Height", 400, 900, 600, 50, key="bi_height")

    # Trendline with graceful fallback
    show_trendline = False
    if chart_type == "Scatter Plot":
        show_trendline = st.checkbox("Show OLS trendline", value=False, key="bi_trendline")
        if show_trendline:
            try:
                import statsmodels  # noqa: F401
            except ImportError:
                st.warning("⚠️ Trendline requires statsmodels: `pip install statsmodels`. Checkbox ignored.")
                show_trendline = False

    axis_opts = {}
    if chart_type in num_num | cat_num:
        axis_opts = _axis_controls("bi", axes=["x", "y"])

    ref = _reference_line_ui("bi")
    df_plot = _data_filter_ui(df, col_types, "bi")
    df_plot = _safe_sample(df_plot, chart_type, "bi")

    st.markdown("---")
    if st.button("📊 Generate Chart", key="bi_gen", type="primary", use_container_width=True):
        try:
            fig = _create_bivariate_chart(
                df_plot, x_col, y_col, chart_type, title, cs, height,
                color_col=color_col, show_trendline=show_trendline,
            )
            if fig:
                fig = _apply_axis_opts(fig, axis_opts)
                fig = _apply_reference_line(fig, ref)
                st.session_state["bi_last_fig"] = fig
                st.session_state["bi_last_name"] = chart_type.replace(" ", "_")
        except Exception as e:
            st.error(f"Chart error: {e}")
            import traceback; st.code(traceback.format_exc())

    if st.session_state.get("bi_last_fig"):
        fig = st.session_state["bi_last_fig"]
        st.plotly_chart(fig, key = "b_l_f", use_container_width=True)
        num_used = [c for c in [x_col, y_col] if c in col_types["numeric"]]
        _summary_stats(df, df_plot, num_used, "bi")
        _add_download_buttons(fig, st.session_state.get("bi_last_name", "chart"), "bi")


def _create_bivariate_chart(df, x_col, y_col, chart_type, title, cs, height, **kw):
    disc = get_discrete_colors(cs)
    cont = get_continuous_scale(cs)
    primary = disc[0]

    if chart_type == "Scatter Plot":
        fig = px.scatter(df, x=x_col, y=y_col, color=kw.get("color_col"), title=title,
                         trendline="ols" if kw.get("show_trendline") else None,
                         opacity=0.6, color_discrete_sequence=disc)

    elif chart_type == "Line Plot":
        fig = px.line(df.sort_values(x_col), x=x_col, y=y_col,
                      color=kw.get("color_col"), title=title, markers=True,
                      color_discrete_sequence=disc)

    elif chart_type == "Hexbin Plot":
        fig = px.density_heatmap(df, x=x_col, y=y_col, title=title,
                                 marginal_x="histogram", marginal_y="histogram",
                                 color_continuous_scale=cont)

    elif chart_type == "Joint Plot":
        fig = make_subplots(
            rows=2, cols=2,
            row_heights=[0.2, 0.8], column_widths=[0.8, 0.2],
            vertical_spacing=0.02, horizontal_spacing=0.02,
            specs=[[{"type": "xy"}, None], [{"type": "xy"}, {"type": "xy"}]],
        )
        fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode="markers",
                                  marker=dict(size=5, opacity=0.5, color=primary)), row=2, col=1)
        fig.add_trace(go.Histogram(x=df[x_col], nbinsx=30, marker_color=disc[0]), row=1, col=1)
        fig.add_trace(go.Histogram(y=df[y_col], nbinsy=30,
                                   marker_color=disc[1] if len(disc) > 1 else disc[0]), row=2, col=2)
        fig.update_layout(title=title, showlegend=False)

    elif chart_type == "Box Plot by Category":
        fig = px.box(df, x=x_col, y=y_col, title=title, color=x_col,
                     color_discrete_sequence=disc)
        fig.update_layout(showlegend=False)

    elif chart_type == "Violin Plot by Category":
        fig = px.violin(df, x=x_col, y=y_col, title=title, box=True,
                        points="outliers", color=x_col, color_discrete_sequence=disc)
        fig.update_layout(showlegend=False)

    elif chart_type == "Strip Plot":
        fig = px.strip(df, x=x_col, y=y_col, title=title, color=x_col,
                       color_discrete_sequence=disc)
        fig.update_layout(showlegend=False)

    elif chart_type == "Bar Plot by Category":
        grouped = df.groupby(x_col)[y_col].mean().reset_index()
        fig = px.bar(grouped, x=x_col, y=y_col, title=title,
                     color=y_col, color_continuous_scale=cont)

    elif chart_type == "Grouped Bar Chart":
        ct = pd.crosstab(df[x_col], df[y_col]).reset_index()
        melted = ct.melt(id_vars=x_col, var_name=y_col, value_name="count")
        fig = px.bar(melted, x=x_col, y="count", color=y_col,
                     barmode="group", title=title, color_discrete_sequence=disc)

    elif chart_type == "Stacked Bar Chart":
        ct = pd.crosstab(df[x_col], df[y_col]).reset_index()
        melted = ct.melt(id_vars=x_col, var_name=y_col, value_name="count")
        fig = px.bar(melted, x=x_col, y="count", color=y_col,
                     barmode="stack", title=title, color_discrete_sequence=disc)

    elif chart_type == "Count Heatmap":
        ct = pd.crosstab(df[x_col], df[y_col])
        fig = px.imshow(ct, title=title, color_continuous_scale=cont, text_auto=True)

    elif chart_type == "Sunburst Chart":
        combo = df.groupby([x_col, y_col]).size().reset_index(name="count")
        fig = px.sunburst(combo, path=[x_col, y_col], values="count",
                          title=title, color_discrete_sequence=disc)
    else:
        return None

    fig.update_layout(height=height)
    return fig


# ============================================================================
# MULTIVARIATE
# ============================================================================

def render_multivariate_charts(df: pd.DataFrame, col_types: Dict):
    st.markdown("### 🔥 Multivariate Analysis")
    st.caption("Explore relationships among multiple variables simultaneously")

    chart_type = st.selectbox(
        "Chart type",
        ["Correlation Heatmap", "Scatter Matrix", "Parallel Coordinates",
         "3D Scatter Plot", "Bubble Chart", "Radar Chart"],
        key="multi_chart_type",
    )

    st.markdown("---")

    if chart_type == "Correlation Heatmap":    _render_correlation_heatmap(df, col_types)
    elif chart_type == "Scatter Matrix":       _render_scatter_matrix(df, col_types)
    elif chart_type == "Parallel Coordinates": _render_parallel_coordinates(df, col_types)
    elif chart_type == "3D Scatter Plot":      _render_3d_scatter(df, col_types)
    elif chart_type == "Bubble Chart":         _render_bubble_chart(df, col_types)
    elif chart_type == "Radar Chart":          _render_radar_chart(df, col_types)


def _render_correlation_heatmap(df, col_types):
    numeric_cols = col_types["numeric"]
    if len(numeric_cols) < 2:
        st.warning("⚠️ Need at least 2 numeric columns"); return

    selected = st.multiselect("Columns (blank = all numeric)", numeric_cols,
                              default=numeric_cols[:min(10, len(numeric_cols))], key="corr_cols")
    c1, c2, c3 = st.columns(3)
    with c1: method = st.selectbox("Method", ["Pearson", "Spearman", "Kendall"], key="corr_method")
    with c2: annotate = st.checkbox("Show values", value=True, key="corr_ann")
    with c3:
        cs = st.selectbox("Colour scheme", list(COLOR_SCHEMES), key="corr_cs")
        st.markdown(_swatches_html(get_discrete_colors(cs)), unsafe_allow_html=True)

    ref = _reference_line_ui("corr")

    if st.button("🔥 Generate Heatmap", key="corr_btn", type="primary", use_container_width=True):
        cols = selected or numeric_cols
        corr = df[cols].corr(method=method.lower())
        fig = px.imshow(corr, text_auto=".2f" if annotate else False,
                        color_continuous_scale=get_continuous_scale(cs),
                        title=f"{method} Correlation Heatmap",
                        aspect="auto", zmin=-1, zmax=1)
        fig = _apply_reference_line(fig, ref)
        fig.update_layout(height=600)
        st.session_state["multi_last_fig"] = fig
        st.session_state["multi_last_name"] = "correlation_heatmap"

    if st.session_state.get("multi_last_fig"):
        fig = st.session_state["multi_last_fig"]
        st.plotly_chart(fig, key = "m_l_f", use_container_width=True)
        _add_download_buttons(fig, st.session_state.get("multi_last_name", "chart"), "multi")


def _render_scatter_matrix(df, col_types):
    numeric_cols = col_types["numeric"]
    if len(numeric_cols) < 2:
        st.warning("⚠️ Need at least 2 numeric columns"); return

    c1, c2 = st.columns(2)
    with c1:
        selected = st.multiselect("Columns (2–6 recommended)", numeric_cols,
                                  default=numeric_cols[:min(4, len(numeric_cols))], key="sm_cols")
    with c2:
        color_by = st.selectbox("Colour by", [None] + col_types["categorical"], key="sm_color")
        cs = st.selectbox("Colour scheme", list(COLOR_SCHEMES), key="sm_cs")

    if len(selected) < 2:
        st.info("Select at least 2 columns"); return

    df_plot = _safe_sample(df, "Scatter Matrix", "sm")

    if st.button("📊 Generate Scatter Matrix", key="sm_btn", type="primary", use_container_width=True):
        fig = px.scatter_matrix(df_plot, dimensions=selected, color=color_by,
                                title="Scatter Matrix",
                                color_discrete_sequence=get_discrete_colors(cs))
        fig.update_traces(diagonal_visible=False)
        fig.update_layout(height=800)
        st.session_state["multi_last_fig"] = fig
        st.session_state["multi_last_name"] = "scatter_matrix"

    if st.session_state.get("multi_last_fig"):
        fig = st.session_state["multi_last_fig"]
        st.plotly_chart(fig, key = "m_l_f_", use_container_width=True)
        _summary_stats(df, df_plot, selected, "sm")
        _add_download_buttons(fig, "scatter_matrix", "sm")


def _render_parallel_coordinates(df, col_types):
    numeric_cols = col_types["numeric"]
    if len(numeric_cols) < 2:
        st.warning("⚠️ Need at least 2 numeric columns"); return

    c1, c2 = st.columns(2)
    with c1:
        selected = st.multiselect("Dimensions", numeric_cols,
                                  default=numeric_cols[:min(5, len(numeric_cols))], key="para_cols")
    with c2:
        color_by = st.selectbox("Colour by", [None] + numeric_cols + col_types["categorical"], key="para_color")
        cs = st.selectbox("Colour scheme", list(COLOR_SCHEMES), key="para_cs")

    if len(selected) < 2:
        st.info("Select at least 2 columns"); return

    df_plot = _safe_sample(df, "Parallel Coordinates", "para")

    if st.button("📊 Generate Parallel Coordinates", key="para_btn", type="primary", use_container_width=True):
        fig = px.parallel_coordinates(df_plot, dimensions=selected, color=color_by,
                                      title="Parallel Coordinates",
                                      color_continuous_scale=get_continuous_scale(cs))
        fig.update_layout(height=600)
        st.session_state["multi_last_fig"] = fig
        st.session_state["multi_last_name"] = "parallel_coordinates"

    if st.session_state.get("multi_last_fig"):
        fig = st.session_state["multi_last_fig"]
        st.plotly_chart(fig, key = "m_l_f_3", use_container_width=True)
        _summary_stats(df, df_plot, selected, "para")
        _add_download_buttons(fig, "parallel_coordinates", "para")


def _render_3d_scatter(df, col_types):
    numeric_cols = col_types["numeric"]
    if len(numeric_cols) < 3:
        st.warning("⚠️ Need at least 3 numeric columns"); return

    c1, c2, c3 = st.columns(3)
    with c1: x = st.selectbox("X", numeric_cols, key="3d_x")
    with c2: y = st.selectbox("Y", [c for c in numeric_cols if c != x] or numeric_cols, key="3d_y")
    with c3: z = st.selectbox("Z", [c for c in numeric_cols if c not in [x, y]] or numeric_cols, key="3d_z")

    c1, c2, c3 = st.columns(3)
    with c1: color_by = st.selectbox("Colour by", [None] + col_types["categorical"] + numeric_cols, key="3d_color")
    with c2: size_by  = st.selectbox("Size by",   [None] + numeric_cols, key="3d_size")
    with c3: cs = st.selectbox("Colour scheme", list(COLOR_SCHEMES), key="3d_cs")

    axis_opts = _axis_controls("3d", axes=["x", "y", "z"])
    df_plot = _safe_sample(df, "3D Scatter Plot", "3d")

    if st.button("🚀 Generate 3D Plot", key="3d_btn", type="primary", use_container_width=True):
        fig = px.scatter_3d(df_plot, x=x, y=y, z=z, color=color_by, size=size_by,
                            title="3D Scatter", opacity=0.7,
                            color_discrete_sequence=get_discrete_colors(cs),
                            color_continuous_scale=get_continuous_scale(cs))
        fig = _apply_axis_opts(fig, axis_opts)
        fig.update_layout(height=700)
        st.session_state["multi_last_fig"] = fig
        st.session_state["multi_last_name"] = "3d_scatter"

    if st.session_state.get("multi_last_fig"):
        fig = st.session_state["multi_last_fig"]
        st.plotly_chart(fig, key = "m_l_f_4", use_container_width=True)
        _summary_stats(df, df_plot, [x, y, z], "3d")
        _add_download_buttons(fig, "3d_scatter", "3d")


def _render_bubble_chart(df, col_types):
    numeric_cols = col_types["numeric"]
    if len(numeric_cols) < 2:
        st.warning("⚠️ Need at least 2 numeric columns"); return

    c1, c2 = st.columns(2)
    with c1: x = st.selectbox("X", numeric_cols, key="bub_x")
    with c2: y = st.selectbox("Y", [c for c in numeric_cols if c != x] or numeric_cols, key="bub_y")

    c1, c2, c3 = st.columns(3)
    with c1: size_col  = st.selectbox("Bubble size", numeric_cols, key="bub_size")
    with c2: color_col = st.selectbox("Colour by",   [None] + col_types["categorical"] + numeric_cols, key="bub_color")
    with c3: cs = st.selectbox("Colour scheme", list(COLOR_SCHEMES), key="bub_cs")

    axis_opts = _axis_controls("bub", axes=["x", "y"])
    ref = _reference_line_ui("bub")

    if st.button("🎈 Generate Bubble Chart", key="bub_btn", type="primary", use_container_width=True):
        fig = px.scatter(df, x=x, y=y, size=size_col, color=color_col,
                         title="Bubble Chart", size_max=50, opacity=0.6,
                         color_discrete_sequence=get_discrete_colors(cs),
                         color_continuous_scale=get_continuous_scale(cs))
        fig = _apply_axis_opts(fig, axis_opts)
        fig = _apply_reference_line(fig, ref)
        fig.update_layout(height=600)
        st.session_state["multi_last_fig"] = fig
        st.session_state["multi_last_name"] = "bubble_chart"

    if st.session_state.get("multi_last_fig"):
        fig = st.session_state["multi_last_fig"]
        st.plotly_chart(fig, key = "m_L_f_5", use_container_width=True)
        _summary_stats(df, df, [x, y, size_col], "bub")
        _add_download_buttons(fig, "bubble_chart", "bub")


def _render_radar_chart(df, col_types):
    numeric_cols   = col_types["numeric"]
    categorical_cols = col_types["categorical"]
    if not numeric_cols or not categorical_cols:
        st.warning("⚠️ Need both numeric and categorical columns"); return

    c1, c2 = st.columns(2)
    with c1:
        cat_col = st.selectbox("Category column", categorical_cols, key="radar_cat")
        top_n   = st.slider("Top N categories", 3, 10, 5, key="radar_topn")
    with c2:
        metrics = st.multiselect("Metrics", numeric_cols,
                                 default=numeric_cols[:min(6, len(numeric_cols))], key="radar_metrics")
        cs = st.selectbox("Colour scheme", list(COLOR_SCHEMES), key="radar_cs")

    if len(metrics) < 3:
        st.info("Select at least 3 metrics"); return

    if st.button("🕸️ Generate Radar Chart", key="radar_btn", type="primary", use_container_width=True):
        disc = get_discrete_colors(cs)
        top_cats = df[cat_col].value_counts().head(top_n).index
        grouped = df[df[cat_col].isin(top_cats)].groupby(cat_col)[metrics].mean()
        fig = go.Figure()
        for i, cat in enumerate(grouped.index):
            fig.add_trace(go.Scatterpolar(
                r=grouped.loc[cat].values, theta=metrics,
                fill="toself", name=str(cat),
                line=dict(color=disc[i % len(disc)]),
            ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                          title="Radar Chart", height=600)
        st.session_state["multi_last_fig"] = fig
        st.session_state["multi_last_name"] = "radar_chart"

    if st.session_state.get("multi_last_fig"):
        fig = st.session_state["multi_last_fig"]
        st.plotly_chart(fig, "M_L_F_6", use_container_width=True)
        _summary_stats(df, df, metrics, "radar")
        _add_download_buttons(fig, "radar_chart", "radar")


# ============================================================================
# TIME SERIES  (FIX: never mutates session_state df)
# ============================================================================

def render_timeseries_charts(df: pd.DataFrame, col_types: Dict):
    st.markdown("### 📉 Time Series Analysis")
    st.caption("Analyze temporal patterns and trends")

    dt_cols  = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    num_cols = col_types["numeric"]

    if not dt_cols:
        st.warning("⚠️ No datetime columns found")
        st.info("💡 Convert a column to datetime on the Transformations page first")
        return
    if not num_cols:
        st.warning("⚠️ No numeric columns available"); return

    c1, c2, c3 = st.columns(3)
    with c1:
        chart_type = st.selectbox(
            "Chart type",
            ["Line Plot", "Area Plot", "Rolling Average", "Multiple Series", "Range Plot"],
            key="ts_type",
        )
    with c2:
        date_col = st.selectbox("Date column", dt_cols, key="ts_date")
    with c3:
        cs = st.selectbox("Colour scheme", list(COLOR_SCHEMES), key="ts_cs")
        st.markdown(_swatches_html(get_discrete_colors(cs)), unsafe_allow_html=True)

    if chart_type == "Multiple Series":
        val_cols = st.multiselect("Value columns", num_cols,
                                  default=num_cols[:min(3, len(num_cols))], key="ts_vals")
    else:
        val_col  = st.selectbox("Value column", num_cols, key="ts_val")
        val_cols = [val_col]

    window = 7
    if chart_type == "Rolling Average":
        window = st.slider("Rolling window", 3, 50, 7, key="ts_window")

    axis_opts = _axis_controls("ts", axes=["y"])
    ref = _reference_line_ui("ts")
    df_plot = _data_filter_ui(df, col_types, "ts")

    # FIX: work on local copy — never touch st.session_state['df']
    df_plot = df_plot.copy()
    df_plot[date_col] = pd.to_datetime(df_plot[date_col], errors="coerce")
    df_sorted = df_plot.sort_values(date_col).dropna(subset=[date_col])

    disc = get_discrete_colors(cs)

    st.markdown("---")
    if st.button("📈 Generate Time Series", key="ts_btn", type="primary", use_container_width=True):
        try:
            primary_val = val_cols[0] if val_cols else num_cols[0]
            fig = None

            if chart_type == "Line Plot":
                fig = px.line(df_sorted, x=date_col, y=primary_val,
                              title=f"Time Series: {primary_val}", markers=True,
                              color_discrete_sequence=disc)

            elif chart_type == "Area Plot":
                fig = px.area(df_sorted, x=date_col, y=primary_val,
                              title=f"Area Plot: {primary_val}",
                              color_discrete_sequence=disc)

            elif chart_type == "Rolling Average":
                tmp = df_sorted.set_index(date_col).copy()
                tmp["_rolling"] = tmp[primary_val].rolling(window=window).mean()
                tmp = tmp.reset_index()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=tmp[date_col], y=tmp[primary_val],
                    mode="lines", name="Original", opacity=0.45,
                    line=dict(color=disc[0]),
                ))
                fig.add_trace(go.Scatter(
                    x=tmp[date_col], y=tmp["_rolling"],
                    mode="lines", name=f"{window}-period MA",
                    line=dict(color=disc[1] if len(disc) > 1 else "#EF553B", width=2),
                ))
                fig.update_layout(title=f"Rolling Average ({window}-period)",
                                  xaxis_title=date_col, yaxis_title=primary_val)

            elif chart_type == "Multiple Series":
                if not val_cols:
                    st.warning("Select at least one value column"); return
                fig = go.Figure()
                for i, col in enumerate(val_cols):
                    fig.add_trace(go.Scatter(
                        x=df_sorted[date_col], y=df_sorted[col],
                        mode="lines+markers", name=col,
                        line=dict(color=disc[i % len(disc)]),
                    ))
                fig.update_layout(title="Multiple Time Series",
                                  xaxis_title=date_col, yaxis_title="Value")

            elif chart_type == "Range Plot":
                agg = df_sorted.groupby(date_col)[primary_val].agg(["min", "max", "mean"]).reset_index()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=agg[date_col], y=agg["max"],
                    mode="lines", name="Max", line=dict(width=0), showlegend=False,
                ))
                fig.add_trace(go.Scatter(
                    x=agg[date_col], y=agg["min"],
                    mode="lines", name="Range",
                    fill="tonexty", fillcolor="rgba(68,68,68,0.15)", line=dict(width=0),
                ))
                fig.add_trace(go.Scatter(
                    x=agg[date_col], y=agg["mean"],
                    mode="lines+markers", name="Mean",
                    line=dict(color=disc[0], width=2),
                ))
                fig.update_layout(title=f"Range Plot: {primary_val}",
                                  xaxis_title=date_col, yaxis_title=primary_val)

            if fig:
                fig = _apply_axis_opts(fig, axis_opts)
                fig = _apply_reference_line(fig, ref)
                fig.update_layout(height=600)
                st.session_state["ts_last_fig"] = fig
                st.session_state["ts_last_name"] = f"ts_{chart_type.replace(' ','_')}"
                st.session_state["ts_last_vals"] = val_cols

        except Exception as e:
            st.error(f"Chart error: {e}")
            import traceback; st.code(traceback.format_exc())

    if st.session_state.get("ts_last_fig"):
        fig = st.session_state["ts_last_fig"]
        st.plotly_chart(fig, key = "t_l_f", use_container_width=True)
        _summary_stats(df, df_sorted, st.session_state.get("ts_last_vals", []), "ts")
        _add_download_buttons(fig, st.session_state.get("ts_last_name", "timeseries"), "ts")


if __name__ == "__main__":
    render_visualizations_page()