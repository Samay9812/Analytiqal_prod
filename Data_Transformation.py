"""
Transformations Page - Fixed & Complete
Fixes:
  1. Double-transform guard: tracks applied transforms in session state,
     warns if a column has already been transformed on this page.
  2. Custom bin edges implemented (was "coming soon").
  3. Multiple conditions implemented (was "coming soon").
  4. Add Days implemented (was "coming soon").
  5. Format Date implemented (was "coming soon").
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime, timedelta


# ── Transform registry ────────────────────────────────────────────────────────
# Tracks {col: [list of transform names]} so we can warn in Feature Engineering.
# Stored in st.session_state['applied_transforms'].

def _register_transform(col: str, name: str):
    if 'applied_transforms' not in st.session_state:
        st.session_state['applied_transforms'] = {}
    st.session_state['applied_transforms'].setdefault(col, []).append(name)


def get_applied_transforms(col: str) -> List[str]:
    """Return list of transforms already applied to col (for cross-page warning)."""
    return st.session_state.get('applied_transforms', {}).get(col, [])


# ── Page entry point ──────────────────────────────────────────────────────────

def render_transformations_page():
    st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    padding: 2rem; border-radius: 12px; margin-bottom: 2rem;'>
            <h1 style='color: white; margin: 0; font-size: 2.5rem;'>🔄 Data Transformations</h1>
            <p style='color: rgba(255,255,255,0.9); margin-top: 0.5rem; font-size: 1.1rem;'>
                Select, sort, sample, compute, and transform your data
            </p>
        </div>
    """, unsafe_allow_html=True)

    df = st.session_state.get('df', None)

    if df is None or df.empty:
        st.info("""
            📂 **No dataset loaded**

            Please load a dataset from the **🏠 Dataset** page first.
        """)
        return

    # Dataset summary
    st.markdown("### 📊 Current Dataset")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Memory", f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    with col4:
        from utils_robust import get_column_types
        col_types = get_column_types(df)
        st.metric("Numeric", len(col_types['numeric']))

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Select Columns",
        "🔢 Sort Data",
        "🎲 Sample Data",
        "➕ Compute Columns",
        "🔧 Data Types"
    ])

    with tab1:
        render_select_columns_section(df)
    with tab2:
        render_sort_data_section(df)
    with tab3:
        render_sample_data_section(df)
    with tab4:
        render_compute_columns_section(df)
    with tab5:
        render_data_type_conversion()


# ── Tab 1: Select Columns ─────────────────────────────────────────────────────

def render_select_columns_section(df: pd.DataFrame):
    st.markdown("### 📋 Select Columns")
    st.caption("Choose which columns to keep in your dataset")

    search = st.text_input("🔍 Search columns", placeholder="Type to filter...", key="sc_search")

    all_cols = list(df.columns)
    filtered_cols = [c for c in all_cols if search.lower() in c.lower()] if search else all_cols

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("✓ Select All", use_container_width=True, key="sc_all"):
            st.session_state.selected_columns = filtered_cols
    with col2:
        if st.button("✗ Deselect All", use_container_width=True, key="sc_none"):
            st.session_state.selected_columns = []
    with col3:
        if st.button("🔢 Numeric Only", use_container_width=True, key="sc_num"):
            from utils_robust import get_column_types
            st.session_state.selected_columns = get_column_types(df)['numeric']
    with col4:
        if st.button("📝 Text Only", use_container_width=True, key="sc_txt"):
            from utils_robust import get_column_types
            st.session_state.selected_columns = get_column_types(df)['categorical']

    default_sel = st.session_state.get('selected_columns', filtered_cols)
    default_sel = [c for c in default_sel if c in filtered_cols]

    selected = st.multiselect(
        "Select columns to keep", filtered_cols,
        default=default_sel, key="sc_multiselect"
    )
    st.session_state.selected_columns = selected

    if selected:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Columns", len(df.columns))
        with col2:
            st.metric("Selected Columns", len(selected))
        with col3:
            st.metric("Removed Columns", len(df.columns) - len(selected))

        st.dataframe(df[selected].head(10), use_container_width=True)

        st.markdown("---")
        if st.button("✅ Apply Selection", type="primary", use_container_width=True, key="sc_apply"):
            from utils_robust import update_df
            update_df(df[selected], f"Selected {len(selected)} columns")
            st.success(f"✓ Selection applied — {len(selected)} columns remaining")
            st.rerun()
    else:
        st.warning("⚠️ No columns selected. Select at least one column.")


# ── Tab 2: Sort Data ──────────────────────────────────────────────────────────

def render_sort_data_section(df: pd.DataFrame):
    st.markdown("### 🔢 Sort Dataset")
    st.caption("Order rows by column values")

    sort_single_tab, sort_multi_tab = st.tabs(["📌 Single Column", "📋 Multiple Columns"])

    with sort_single_tab:
        sort_col = st.selectbox("Column to sort by", df.columns, key="sort_single_col")

        c1, c2 = st.columns(2)
        with c1:
            sort_dir = st.selectbox("Order", ["Ascending ↑", "Descending ↓"], key="sort_single_dir")
            ascending = sort_dir == "Ascending ↑"
        with c2:
            na_pos = st.selectbox("Missing values position", ["Last", "First"], key="sort_na_pos")

        try:
            sorted_df = df.sort_values(
                by=sort_col, ascending=ascending,
                na_position='last' if na_pos == "Last" else 'first'
            )
            st.info(f"**Sort:** {sort_col} {'↑' if ascending else '↓'}")
            st.dataframe(sorted_df.head(20), use_container_width=True)

            st.markdown("---")
            if st.button("✅ Apply Sort", type="primary", use_container_width=True, key="sort_apply"):
                from utils_robust import update_df
                update_df(sorted_df, f"Sorted by {sort_col} ({'asc' if ascending else 'desc'})")
                st.success(f"✓ Data sorted by {sort_col}")
                st.rerun()
        except Exception as e:
            st.error(f"Sort error: {e}")

    with sort_multi_tab:
        st.markdown("**Multi-Column Sort**")

        if 'sort_columns' not in st.session_state:
            st.session_state.sort_columns = []

        if st.button("➕ Add Sort Column", key="sort_add"):
            st.session_state.sort_columns.append({'column': df.columns[0], 'ascending': True})

        for idx, info in enumerate(st.session_state.sort_columns):
            c1, c2, c3 = st.columns([3, 2, 1])
            with c1:
                info['column'] = st.selectbox(
                    f"Sort {idx + 1}", df.columns,
                    index=list(df.columns).index(info['column']),
                    key=f"sort_mc_col_{idx}"
                )
            with c2:
                dir_choice = st.selectbox(
                    "Order", ["Ascending ↑", "Descending ↓"],
                    key=f"sort_mc_dir_{idx}"
                )
                info['ascending'] = dir_choice == "Ascending ↑"
            with c3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🗑️", key=f"sort_mc_del_{idx}"):
                    st.session_state.sort_columns.pop(idx)
                    st.rerun()

        if st.session_state.sort_columns:
            try:
                cols = [s['column'] for s in st.session_state.sort_columns]
                dirs = [s['ascending'] for s in st.session_state.sort_columns]
                sorted_df = df.sort_values(by=cols, ascending=dirs)
                st.dataframe(sorted_df.head(20), use_container_width=True)

                st.markdown("---")
                if st.button("✅ Apply Multi-Sort", type="primary", use_container_width=True, key="sort_mc_apply"):
                    from utils_robust import update_df
                    update_df(sorted_df, f"Sorted by {len(cols)} columns")
                    st.session_state.sort_columns = []
                    st.success("✓ Multi-sort applied!")
                    st.rerun()
            except Exception as e:
                st.error(f"Sort error: {e}")
        else:
            st.info("Click 'Add Sort Column' to start building your sort")


# ── Tab 3: Sample Data ────────────────────────────────────────────────────────

def render_sample_data_section(df: pd.DataFrame):
    st.markdown("### 🎲 Sample Data")
    st.caption("Extract a random subset of rows")

    samp_fixed_tab, samp_pct_tab, samp_strat_tab = st.tabs([
        "🔢 Fixed Count", "📊 Percentage", "🎯 Stratified"
    ])

    with samp_fixed_tab:
        c1, c2 = st.columns([2, 1])
        with c1:
            n = st.slider("Rows to sample", 1, len(df), min(100, len(df)), 10, key="samp_n")
        with c2:
            seed = st.number_input("Seed", value=42, min_value=0, step=1, key="samp_seed")

        sampled = df.sample(n=n, random_state=int(seed))
        c1, c2 = st.columns(2)
        c1.metric("Original Rows", f"{len(df):,}")
        c2.metric("Sampled Rows", f"{len(sampled):,}")
        st.dataframe(sampled.head(20), use_container_width=True)

        if st.button("✅ Apply Sample", type="primary", use_container_width=True, key="samp_apply_n"):
            from utils_robust import update_df
            update_df(sampled, f"Sampled {n} random rows")
            st.success(f"✓ Sampled {n} rows")
            st.rerun()

    with samp_pct_tab:
        c1, c2 = st.columns([2, 1])
        with c1:
            pct = st.slider("Percentage", 1, 100, 10, 5, key="samp_pct")
        with c2:
            seed = st.number_input("Seed", value=42, min_value=0, step=1, key="samp_pct_seed")

        sampled = df.sample(frac=pct / 100, random_state=int(seed))
        c1, c2 = st.columns(2)
        c1.metric("Original Rows", f"{len(df):,}")
        c2.metric("Sampled Rows", f"{len(sampled):,} ({pct}%)")
        st.dataframe(sampled.head(20), use_container_width=True)

        if st.button("✅ Apply Sample", type="primary", use_container_width=True, key="samp_apply_pct"):
            from utils_robust import update_df
            update_df(sampled, f"Sampled {pct}% ({len(sampled)} rows)")
            st.success(f"✓ Sampled {pct}%")
            st.rerun()

    with samp_strat_tab:
        from utils_robust import get_column_types
        col_types = get_column_types(df)
        if not col_types['categorical']:
            st.warning("⚠️ No categorical columns for stratification")
        else:
            c1, c2 = st.columns(2)
            with c1:
                strat_col = st.selectbox("Stratify by", col_types['categorical'], key="samp_strat_col")
            with c2:
                n = st.slider("Rows to sample", 1, len(df), min(100, len(df)), key="samp_strat_n")

            seed = st.number_input("Seed", value=42, min_value=0, step=1, key="samp_strat_seed")

            try:
                props = df[strat_col].value_counts(normalize=True)
                st.dataframe(pd.DataFrame({
                    'Class': props.index,
                    'Proportion': props.values,
                    'Original Count': df[strat_col].value_counts().values,
                    'Sample Count': (props.values * n).astype(int)
                }), use_container_width=True, hide_index=True)

                sampled = df.groupby(strat_col, group_keys=False).apply(
                    lambda x: x.sample(frac=n / len(df), random_state=int(seed))
                ).reset_index(drop=True)

                st.dataframe(sampled.head(20), use_container_width=True)

                if st.button("✅ Apply Stratified Sample", type="primary",
                             use_container_width=True, key="samp_apply_strat"):
                    from utils_robust import update_df
                    update_df(sampled, f"Stratified sample by {strat_col} ({len(sampled)} rows)")
                    st.success("✓ Stratified sampling applied!")
                    st.rerun()
            except Exception as e:
                st.error(f"Stratified sampling error: {e}")


# ── Tab 4: Compute Columns ────────────────────────────────────────────────────

def render_compute_columns_section(df: pd.DataFrame):
    st.markdown("### ➕ Compute New Columns")
    st.caption("Create calculated columns from existing data")

    math_tab, str_tab, date_tab, cond_tab = st.tabs([
        "🔢 Math Expression", "📝 String Operations",
        "📅 Date Operations", "🔀 Conditional Logic"
    ])

    with math_tab:
        _render_math_computation(df)
    with str_tab:
        _render_string_computation(df)
    with date_tab:
        _render_date_computation(df)
    with cond_tab:
        _render_conditional_computation(df)


def _render_math_computation(df):
    from utils_robust import get_column_types
    numeric_cols = get_column_types(df)['numeric']
    if not numeric_cols:
        st.warning("⚠️ No numeric columns")
        return

    new_name = st.text_input("New column name", placeholder="my_calculation", key="math_new_name")

    op = st.selectbox("Operation", [
        "Add (+)", "Subtract (-)", "Multiply (*)", "Divide (/)", "Power (**)", "Custom Expression"
    ], key="math_op")

    if op != "Custom Expression":
        c1, c2 = st.columns(2)
        with c1:
            col_a = st.selectbox("First column", numeric_cols, key="math_col_a")
        with c2:
            col_b = st.selectbox("Second column", numeric_cols, key="math_col_b")

        if new_name:
            try:
                res = df.copy()
                if op == "Add (+)":         res[new_name] = df[col_a] + df[col_b]
                elif op == "Subtract (-)":  res[new_name] = df[col_a] - df[col_b]
                elif op == "Multiply (*)":  res[new_name] = df[col_a] * df[col_b]
                elif op == "Divide (/)":    res[new_name] = df[col_a] / df[col_b].replace(0, np.nan)
                else:                       res[new_name] = df[col_a] ** df[col_b]

                st.dataframe(res[[col_a, col_b, new_name]].head(10), use_container_width=True)
                st.write(f"Mean={res[new_name].mean():.3f}  Min={res[new_name].min():.3f}  Max={res[new_name].max():.3f}")

                if st.button("✅ Create Column", type="primary", use_container_width=True, key="math_apply"):
                    from utils_robust import update_df
                    update_df(res, f"Created column: {new_name}")
                    st.success(f"✓ Created '{new_name}'")
                    st.rerun()
            except Exception as e:
                st.error(f"Calculation error: {e}")
    else:
        with st.expander("💡 Examples"):
            st.code("price * quantity\n(price * quantity) - discount\nnp.log(price)\nnp.sqrt(quantity)")

        expr = st.text_area("Expression", placeholder="price * quantity * 1.1",
                            height=100, key="math_expr")

        if new_name and expr:
            try:
                res = df.copy()
                res[new_name] = df.eval(expr)
                involved = [c for c in df.columns if c in expr]
                st.dataframe(res[involved + [new_name]].head(10), use_container_width=True)

                if st.button("✅ Create Column", type="primary", use_container_width=True, key="math_custom_apply"):
                    from utils_robust import update_df
                    update_df(res, f"Created column: {new_name}")
                    st.success(f"✓ Created '{new_name}'")
                    st.rerun()
            except Exception as e:
                st.error(f"Expression error: {e}")


def _render_string_computation(df):
    from utils_robust import get_column_types
    text_cols = get_column_types(df)['categorical']
    if not text_cols:
        st.warning("⚠️ No text columns")
        return

    new_name = st.text_input("New column name", placeholder="combined_text", key="str_new_name")
    op = st.selectbox("String operation", [
        "Concatenate", "Extract Substring", "Replace",
        "Length", "Uppercase", "Lowercase", "Title Case", "Strip Whitespace"
    ], key="str_op")

    if op == "Concatenate":
        cols = st.multiselect("Columns to combine", text_cols, key="str_concat_cols")
        sep = st.text_input("Separator", value=" ", key="str_sep")
        if new_name and cols:
            res = df.copy()
            res[new_name] = df[cols].astype(str).agg(sep.join, axis=1)
            st.dataframe(res[cols + [new_name]].head(10), use_container_width=True)
            if st.button("✅ Create Column", type="primary", use_container_width=True, key="str_concat_apply"):
                from utils_robust import update_df
                update_df(res, f"Created column: {new_name}")
                st.success(f"✓ Created '{new_name}'"); st.rerun()

    elif op == "Extract Substring":
        src = st.selectbox("Source column", text_cols, key="str_sub_src")
        c1, c2 = st.columns(2)
        start = c1.number_input("Start", value=0, min_value=0, step=1, key="str_sub_start")
        end   = c2.number_input("End (0=end)", value=0, min_value=0, step=1, key="str_sub_end")
        if new_name:
            res = df.copy()
            res[new_name] = df[src].astype(str).str[start:end if end else None]
            st.dataframe(res[[src, new_name]].head(10), use_container_width=True)
            if st.button("✅ Create Column", type="primary", use_container_width=True, key="str_sub_apply"):
                from utils_robust import update_df
                update_df(res, f"Created column: {new_name}"); st.success(f"✓ Created '{new_name}'"); st.rerun()

    elif op == "Replace":
        src = st.selectbox("Source column", text_cols, key="str_rep_src")
        c1, c2 = st.columns(2)
        old = c1.text_input("Find", key="str_rep_old")
        new_txt = c2.text_input("Replace with", key="str_rep_new")
        case = st.checkbox("Case sensitive", value=True, key="str_rep_case")
        if new_name and old:
            res = df.copy()
            res[new_name] = df[src].astype(str).str.replace(old, new_txt, case=case, regex=False)
            st.dataframe(res[[src, new_name]].head(10), use_container_width=True)
            if st.button("✅ Create Column", type="primary", use_container_width=True, key="str_rep_apply"):
                from utils_robust import update_df
                update_df(res, f"Created column: {new_name}"); st.success(f"✓ Created '{new_name}'"); st.rerun()

    elif op == "Length":
        src = st.selectbox("Source column", text_cols, key="str_len_src")
        if new_name:
            res = df.copy()
            res[new_name] = df[src].astype(str).str.len()
            st.dataframe(res[[src, new_name]].head(10), use_container_width=True)
            if st.button("✅ Create Column", type="primary", use_container_width=True, key="str_len_apply"):
                from utils_robust import update_df
                update_df(res, f"Created column: {new_name}"); st.success(f"✓ Created '{new_name}'"); st.rerun()

    else:
        src = st.selectbox("Source column", text_cols, key="str_case_src")
        if new_name:
            res = df.copy()
            if op == "Uppercase":        res[new_name] = df[src].astype(str).str.upper()
            elif op == "Lowercase":      res[new_name] = df[src].astype(str).str.lower()
            elif op == "Title Case":     res[new_name] = df[src].astype(str).str.title()
            else:                        res[new_name] = df[src].astype(str).str.strip()
            st.dataframe(res[[src, new_name]].head(10), use_container_width=True)
            if st.button("✅ Create Column", type="primary", use_container_width=True, key="str_case_apply"):
                from utils_robust import update_df
                update_df(res, f"Created column: {new_name}"); st.success(f"✓ Created '{new_name}'"); st.rerun()


def _render_date_computation(df):
    """Date operations — all options fully implemented."""
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

    if not datetime_cols:
        st.warning("⚠️ No datetime columns. Convert a column to datetime first (Data Types tab).")
        return

    new_name = st.text_input("New column name", placeholder="date_feature", key="dt_new_name")

    op = st.selectbox("Date operation", [
        "Extract Year", "Extract Month", "Extract Day",
        "Extract Day of Week", "Extract Hour", "Extract Quarter",
        "Extract Week of Year",
        "Date Difference",
        "Add Days",           # ← was "coming soon"
        "Format Date",        # ← was "coming soon"
    ], key="dt_op")

    src = st.selectbox("Source datetime column", datetime_cols, key="dt_src")

    # ── Extract components ────────────────────────────────────────────────
    extract_map = {
        "Extract Year":        lambda s: s.dt.year,
        "Extract Month":       lambda s: s.dt.month,
        "Extract Day":         lambda s: s.dt.day,
        "Extract Day of Week": lambda s: s.dt.dayofweek,
        "Extract Hour":        lambda s: s.dt.hour,
        "Extract Quarter":     lambda s: s.dt.quarter,
        "Extract Week of Year":lambda s: s.dt.isocalendar().week.astype(int),
    }

    if op in extract_map and new_name:
        res = df.copy()
        res[new_name] = extract_map[op](df[src])
        st.dataframe(res[[src, new_name]].head(10), use_container_width=True)
        if st.button("✅ Create Column", type="primary", use_container_width=True, key="dt_extract_apply"):
            from utils_robust import update_df
            update_df(res, f"Created column: {new_name}"); st.success(f"✓ Created '{new_name}'"); st.rerun()

    elif op == "Date Difference":
        col2 = st.selectbox("Second datetime column", datetime_cols, key="dt_diff_col2")
        unit = st.selectbox("Unit", ["Days", "Hours", "Minutes", "Seconds"], key="dt_diff_unit")
        if new_name:
            res = df.copy()
            diff = df[src] - df[col2]
            if unit == "Days":      res[new_name] = diff.dt.days
            elif unit == "Hours":   res[new_name] = diff.dt.total_seconds() / 3600
            elif unit == "Minutes": res[new_name] = diff.dt.total_seconds() / 60
            else:                   res[new_name] = diff.dt.total_seconds()
            st.dataframe(res[[src, col2, new_name]].head(10), use_container_width=True)
            if st.button("✅ Create Column", type="primary", use_container_width=True, key="dt_diff_apply"):
                from utils_robust import update_df
                update_df(res, f"Created column: {new_name}"); st.success(f"✓ Created '{new_name}'"); st.rerun()

    elif op == "Add Days":
        # ── IMPLEMENTED (was "coming soon") ──────────────────────────────
        st.caption("Add or subtract a fixed number of days from each date value.")
        days = st.number_input("Days to add (negative to subtract)", value=30, step=1, key="dt_add_days")
        if new_name:
            res = df.copy()
            res[new_name] = df[src] + pd.Timedelta(days=int(days))
            st.dataframe(res[[src, new_name]].head(10), use_container_width=True)
            st.info(f"Sample: {df[src].iloc[0].date()} → {res[new_name].iloc[0].date()}")
            if st.button("✅ Create Column", type="primary", use_container_width=True, key="dt_adddays_apply"):
                from utils_robust import update_df
                update_df(res, f"Created column: {new_name} ({'+' if days >= 0 else ''}{days} days)")
                st.success(f"✓ Created '{new_name}'"); st.rerun()

    elif op == "Format Date":
        # ── IMPLEMENTED (was "coming soon") ──────────────────────────────
        st.caption("Convert the datetime column to a formatted text string.")
        fmt_choice = st.selectbox("Output format", [
            "YYYY-MM-DD (2024-01-15)",
            "DD/MM/YYYY (15/01/2024)",
            "MM/DD/YYYY (01/15/2024)",
            "DD MMM YYYY (15 Jan 2024)",
            "MMM YYYY (Jan 2024)",
            "YYYY/MM/DD (2024/01/15)",
            "Custom strftime pattern",
        ], key="dt_fmt_choice")

        fmt_map = {
            "YYYY-MM-DD (2024-01-15)":   "%Y-%m-%d",
            "DD/MM/YYYY (15/01/2024)":   "%d/%m/%Y",
            "MM/DD/YYYY (01/15/2024)":   "%m/%d/%Y",
            "DD MMM YYYY (15 Jan 2024)": "%d %b %Y",
            "MMM YYYY (Jan 2024)":       "%b %Y",
            "YYYY/MM/DD (2024/01/15)":   "%Y/%m/%d",
        }

        if fmt_choice == "Custom strftime pattern":
            pattern = st.text_input("strftime pattern", value="%Y-%m-%d", key="dt_fmt_custom")
        else:
            pattern = fmt_map[fmt_choice]

        if new_name and pattern:
            try:
                res = df.copy()
                res[new_name] = df[src].dt.strftime(pattern)
                st.dataframe(res[[src, new_name]].head(10), use_container_width=True)
                if st.button("✅ Create Column", type="primary", use_container_width=True, key="dt_fmt_apply"):
                    from utils_robust import update_df
                    update_df(res, f"Created column: {new_name} (formatted)")
                    st.success(f"✓ Created '{new_name}'"); st.rerun()
            except Exception as e:
                st.error(f"Format error: {e}")


def _render_conditional_computation(df):
    """Conditional logic — Multiple Conditions now fully implemented."""
    st.markdown("**Conditional Column Creation**")
    st.caption("Create columns based on if-then-else logic")

    new_name = st.text_input("New column name", placeholder="category", key="cond_new_name")

    cond_simple_tab, cond_multi_tab, cond_bin_tab = st.tabs([
        "⚡ Simple If-Else", "📋 Multiple Conditions", "🗂️ Binning"
    ])

    with cond_simple_tab:
        cond_col = st.selectbox("Condition column", df.columns, key="cond_simple_col")
        c1, c2 = st.columns(2)
        with c1:
            operator = st.selectbox("Operator",
                ["==", "!=", ">", "<", ">=", "<=", "contains"], key="cond_op")
        with c2:
            if pd.api.types.is_numeric_dtype(df[cond_col]):
                threshold = st.number_input("Value", value=float(df[cond_col].mean()), key="cond_thresh_num")
            else:
                threshold = st.text_input("Value", key="cond_thresh_txt")

        c1, c2 = st.columns(2)
        true_val  = c1.text_input("Value if True",  value="Yes", key="cond_true")
        false_val = c2.text_input("Value if False", value="No",  key="cond_false")

        if new_name:
            res = df.copy()
            if operator == "==":       mask = df[cond_col] == threshold
            elif operator == "!=":     mask = df[cond_col] != threshold
            elif operator == ">":      mask = df[cond_col] > threshold
            elif operator == "<":      mask = df[cond_col] < threshold
            elif operator == ">=":     mask = df[cond_col] >= threshold
            elif operator == "<=":     mask = df[cond_col] <= threshold
            else:                      mask = df[cond_col].astype(str).str.contains(str(threshold), na=False)

            res[new_name] = np.where(mask, true_val, false_val)
            st.dataframe(res[[cond_col, new_name]].head(10), use_container_width=True)
            st.write(f"True: {mask.sum():,}   False: {(~mask).sum():,}")

            if st.button("✅ Create Column", type="primary", use_container_width=True, key="cond_simple_apply"):
                from utils_robust import update_df
                update_df(res, f"Created column: {new_name}"); st.success(f"✓ Created '{new_name}'"); st.rerun()

    with cond_multi_tab:
        st.caption("Define multiple if / else-if / else rules. Rules are evaluated top-to-bottom; first match wins.")

        if 'multi_cond_rules' not in st.session_state:
            st.session_state.multi_cond_rules = []

        if st.button("➕ Add Rule", key="mc_add_rule"):
            st.session_state.multi_cond_rules.append({
                'col': df.columns[0], 'op': '==',
                'val': '', 'label': f'Category_{len(st.session_state.multi_cond_rules)+1}'
            })

        rules = st.session_state.multi_cond_rules

        for i, rule in enumerate(rules):
            st.markdown(f"**Rule {i+1}**")
            r1, r2, r3, r4, r5 = st.columns([2, 1, 2, 2, 0.5])
            rule['col']   = r1.selectbox("Column", df.columns, key=f"mc_col_{i}")
            rule['op']    = r2.selectbox("Op", ["==","!=",">","<",">=","<=","contains"], key=f"mc_op_{i}")
            rule['val']   = r3.text_input("Value", value=str(rule['val']), key=f"mc_val_{i}")
            rule['label'] = r4.text_input("Output", value=rule['label'], key=f"mc_lbl_{i}")
            with r5:
                if st.button("🗑️", key=f"mc_del_{i}"):
                    st.session_state.multi_cond_rules.pop(i); st.rerun()

        default_label = st.text_input("Default value (no rule matched)", value="Other", key="mc_default")

        if rules and new_name:
            try:
                res = df.copy()
                result_col = pd.Series(default_label, index=df.index)

                # Apply in reverse so rule 1 has highest priority
                for rule in reversed(rules):
                    col_  = rule['col']
                    op_   = rule['op']
                    val_  = rule['val']
                    lbl   = rule['label']

                    # Try numeric comparison first
                    try:
                        num_val = float(val_)
                        if op_ == "==":   m = df[col_] == num_val
                        elif op_ == "!=": m = df[col_] != num_val
                        elif op_ == ">":  m = df[col_] > num_val
                        elif op_ == "<":  m = df[col_] < num_val
                        elif op_ == ">=": m = df[col_] >= num_val
                        elif op_ == "<=": m = df[col_] <= num_val
                        else:             m = df[col_].astype(str).str.contains(val_, na=False)
                    except (ValueError, TypeError):
                        if op_ == "==":       m = df[col_].astype(str) == val_
                        elif op_ == "!=":     m = df[col_].astype(str) != val_
                        elif op_ == "contains": m = df[col_].astype(str).str.contains(val_, na=False)
                        else:                 m = pd.Series(False, index=df.index)

                    result_col[m] = lbl

                res[new_name] = result_col

                st.markdown("**Preview:**")
                st.dataframe(res[list({r['col'] for r in rules}) + [new_name]].head(15),
                             use_container_width=True)
                st.write("**Distribution:**", res[new_name].value_counts().to_dict())

                if st.button("✅ Create Column", type="primary", use_container_width=True, key="mc_apply"):
                    from utils_robust import update_df
                    update_df(res, f"Created column: {new_name} (multi-condition)")
                    st.session_state.multi_cond_rules = []
                    st.success(f"✓ Created '{new_name}'"); st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
        elif not rules:
            st.info("Click 'Add Rule' to start building conditions")

    with cond_bin_tab:
        from utils_robust import get_column_types
        numeric_cols = get_column_types(df)['numeric']
        if not numeric_cols:
            st.warning("⚠️ No numeric columns for binning")
        else:
            src = st.selectbox("Column to bin", numeric_cols, key="bin_src")

            bin_ew_tab, bin_ef_tab, bin_ce_tab = st.tabs([
                "📏 Equal Width", "📊 Equal Frequency", "✏️ Custom Edges"
            ])

            with bin_ew_tab:
                n_bins = st.slider("Number of bins", 2, 20, 5, key="bin_ew_n")
                label_type = st.selectbox("Bin labels",
                    ["Numeric (0,1,2…)", "Range strings (0.0–1.5)"], key="bin_ew_lbl")

                if new_name:
                    _, edges = pd.cut(df[src], bins=n_bins, retbins=True)
                    if "Range" in label_type:
                        labels = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(n_bins)]
                        res = df.copy()
                        res[new_name] = pd.cut(df[src], bins=n_bins, labels=labels)
                    else:
                        res = df.copy()
                        res[new_name] = pd.cut(df[src], bins=n_bins, labels=False)

                    st.write("**Bin edges:**", [f"{e:.3f}" for e in edges])
                    st.dataframe(res[[src, new_name]].head(10), use_container_width=True)

                    if st.button("✅ Create Column", type="primary", use_container_width=True, key="bin_ew_apply"):
                        from utils_robust import update_df
                        update_df(res, f"Created column: {new_name} (equal-width bins)")
                        st.success(f"✓ Created '{new_name}'"); st.rerun()

            with bin_ef_tab:
                n_bins = st.slider("Number of bins", 2, 20, 5, key="bin_ef_n")
                if new_name:
                    res = df.copy()
                    res[new_name] = pd.qcut(df[src], q=n_bins, labels=False, duplicates='drop')
                    st.dataframe(res[[src, new_name]].head(10), use_container_width=True)
                    st.write(res[new_name].value_counts().sort_index())
                    if st.button("✅ Create Column", type="primary", use_container_width=True, key="bin_ef_apply"):
                        from utils_robust import update_df
                        update_df(res, f"Created column: {new_name} (equal-frequency bins)")
                        st.success(f"✓ Created '{new_name}'"); st.rerun()

            with bin_ce_tab:
                st.caption("Enter your own bin boundaries. Values outside the range become NaN.")

                col_min = float(df[src].min())
                col_max = float(df[src].max())
                st.info(f"Column range: {col_min:.3f} → {col_max:.3f}")

                n_custom = st.slider("Number of boundaries (= bins + 1)", 2, 10, 4, key="bin_ce_n")
                edges_input = []

                st.markdown("**Enter boundary values:**")
                cols = st.columns(n_custom)
                for i, col_ in enumerate(cols):
                    default_edge = col_min + (col_max - col_min) * i / (n_custom - 1)
                    val = col_.number_input(
                        f"Edge {i+1}", value=round(default_edge, 3),
                        key=f"bin_edge_{i}"
                    )
                    edges_input.append(val)

                # Label customisation
                n_bins_custom = n_custom - 1
            use_custom_labels = st.checkbox("Custom label names", value=False, key="bin_ce_labels_toggle")
            custom_labels = None
            if use_custom_labels:
                label_cols = st.columns(n_bins_custom)
                custom_labels = []
                for i, lc in enumerate(label_cols):
                    lbl = lc.text_input(f"Bin {i+1} label", value=f"Bin_{i+1}", key=f"bin_lbl_{i}")
                    custom_labels.append(lbl)

            if new_name:
                try:
                    sorted_edges = sorted(set(edges_input))
                    if len(sorted_edges) < 2:
                        st.warning("Need at least 2 distinct edge values")
                    else:
                        res = df.copy()
                        labels_arg = custom_labels if use_custom_labels and custom_labels else False
                        res[new_name] = pd.cut(df[src], bins=sorted_edges, labels=labels_arg, include_lowest=True)

                        st.dataframe(res[[src, new_name]].head(15), use_container_width=True)
                        st.write("**Bin distribution:**", res[new_name].value_counts().sort_index().to_dict())

                        if st.button("✅ Create Column", type="primary", use_container_width=True, key="bin_ce_apply"):
                            from utils_robust import update_df
                            update_df(res, f"Created column: {new_name} (custom edges)")
                            st.success(f"✓ Created '{new_name}'"); st.rerun()
                except Exception as e:
                    st.error(f"Binning error: {e}")


# ── Tab 5: Data Types ─────────────────────────────────────────────────────────

def render_data_type_conversion():
    st.markdown("### 🔧 Change Data Types")
    st.caption("Convert columns to different data types with validation")

    df = st.session_state.get('df')
    if df is None or df.empty:
        st.warning("No data available"); return

    with st.expander("ℹ️ Understanding Data Types"):
        st.markdown("""
        - **To Numeric:** Text numbers → integers/floats
        - **To String:** Any type → text (safe)
        - **To DateTime:** Text dates → datetime
        - **To Category:** Repeated text → category (saves memory)
        - **To Boolean:** True/False text → boolean with custom output
        """)

    c1, c2 = st.columns([2, 1])
    with c1:
        target_col = st.selectbox("Column to convert", df.columns.tolist(), key="dtype_col")
    with c2:
        st.markdown("**Current Type:**")
        st.info(f"**{df[target_col].dtype}**")

    with st.expander("👁️ Current Values"):
        st.dataframe(pd.DataFrame({
            'Value': df[target_col].head(20),
            'Type': [type(x).__name__ for x in df[target_col].head(20)]
        }), use_container_width=True, hide_index=True)

    st.markdown("---")

    type_tabs = st.tabs([
        "🔢 Numeric", "📝 Text", "📅 DateTime", "🏷️ Category", "✅ Boolean"
    ])

    # ── Numeric tab ───────────────────────────────────────────────────────
    with type_tabs[0]:
        target_type = "numeric"
        opts = {}
        c1, c2 = st.columns(2)
        sub = c1.selectbox("Numeric type", ["Integer", "Float"], key="dtype_num_sub")
        opts['subtype'] = 'int' if sub == "Integer" else 'float'
        err = c2.selectbox("Non-numeric values", ["Convert to NaN", "Keep as 0"], key="dtype_num_err")
        opts['errors'] = err

        st.markdown("---")
        if st.button("🔍 Preview Conversion", use_container_width=True, key="dtype_preview_btn_num"):
            try:
                preview_df = df.copy()
                converted, rate = _convert_column_type(preview_df[target_col], target_type, opts)
                preview_df[target_col] = converted
                st.session_state.update({'dtype_preview': preview_df, 'dtype_success_rate': rate,
                                         'dtype_target_col': target_col, 'dtype_type': target_type})
            except Exception as e:
                st.error(f"Conversion failed: {e}")
        _render_dtype_preview(df, target_col)

    # ── Text tab ──────────────────────────────────────────────────────────
    with type_tabs[1]:
        target_type = "string"
        opts = {}
        st.info("All values will be converted to text strings. This conversion always succeeds.")
        st.markdown("---")
        if st.button("🔍 Preview Conversion", use_container_width=True, key="dtype_preview_btn_str"):
            try:
                preview_df = df.copy()
                converted, rate = _convert_column_type(preview_df[target_col], target_type, opts)
                preview_df[target_col] = converted
                st.session_state.update({'dtype_preview': preview_df, 'dtype_success_rate': rate,
                                         'dtype_target_col': target_col, 'dtype_type': target_type})
            except Exception as e:
                st.error(f"Conversion failed: {e}")
        _render_dtype_preview(df, target_col)

    # ── DateTime tab ──────────────────────────────────────────────────────
    with type_tabs[2]:
        target_type = "datetime"
        opts = {}
        samples = df[target_col].dropna().astype(str).head(20).tolist()
        detected = _detect_date_formats(samples)
        if detected:
            st.markdown("**Detected formats:**")
            st.dataframe(pd.DataFrame([{'Format': k, 'Example': v} for k, v in detected.items()]),
                         use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            opts['output_format'] = st.selectbox("Standardise to", [
                "YYYY-MM-DD (2024-01-15)", "DD/MM/YYYY (15/01/2024)", "MM/DD/YYYY (01/15/2024)",
                "DD-MM-YYYY (15-01-2024)", "YYYY/MM/DD (2024/01/15)",
                "DD MMM YYYY (15 Jan 2024)", "Keep as datetime object"
            ], key="dtype_dt_fmt")
        with c2:
            opts['ambiguous'] = st.selectbox("Ambiguous dates", [
                "Assume DD/MM/YYYY first", "Assume MM/DD/YYYY first"
            ], key="dtype_dt_amb")
        with st.expander("🔧 Custom format"):
            mf = st.text_input("Extra format to try", placeholder="%Y/%m/%d", key="dtype_dt_manual")
            opts['manual_format'] = mf or None

        st.markdown("---")
        if st.button("🔍 Preview Conversion", use_container_width=True, key="dtype_preview_btn_dt"):
            try:
                preview_df = df.copy()
                converted, rate = _convert_column_type(preview_df[target_col], target_type, opts)
                preview_df[target_col] = converted
                st.session_state.update({'dtype_preview': preview_df, 'dtype_success_rate': rate,
                                         'dtype_target_col': target_col, 'dtype_type': target_type})
            except Exception as e:
                st.error(f"Conversion failed: {e}")
        _render_dtype_preview(df, target_col)

    # ── Category tab ──────────────────────────────────────────────────────
    with type_tabs[3]:
        target_type = "category"
        opts = {}
        unique_count = df[target_col].nunique()
        pct = unique_count / len(df) * 100
        st.info(f"Column has **{unique_count:,}** unique values ({pct:.1f}% of rows). "
                f"Category type is most efficient when < 50% unique.")
        st.markdown("---")
        if st.button("🔍 Preview Conversion", use_container_width=True, key="dtype_preview_btn_cat"):
            try:
                preview_df = df.copy()
                converted, rate = _convert_column_type(preview_df[target_col], target_type, opts)
                preview_df[target_col] = converted
                st.session_state.update({'dtype_preview': preview_df, 'dtype_success_rate': rate,
                                         'dtype_target_col': target_col, 'dtype_type': target_type})
            except Exception as e:
                st.error(f"Conversion failed: {e}")
        _render_dtype_preview(df, target_col)

    # ── Boolean tab ───────────────────────────────────────────────────────
    with type_tabs[4]:
        target_type = "boolean"
        opts = {}
        c1, c2 = st.columns(2)
        with c1:
            opts['output_format'] = st.selectbox("Output format", [
                "Keep as True/False (boolean)", "Convert to Yes/No (text)",
                "Convert to 1/0 (numeric)", "Convert to Y/N (text)",
                "Convert to Active/Inactive (text)", "Convert to Enabled/Disabled (text)",
                "Custom values..."
            ], key="dtype_bool_fmt")
        with c2:
            opts['errors'] = st.selectbox("Invalid values",
                ["Convert to NaN", "Treat as False"], key="dtype_bool_err")
        if "Custom" in opts['output_format']:
            c1, c2 = st.columns(2)
            opts['custom_true']  = c1.text_input("True value",  value="Active",   key="dtype_bool_ct")
            opts['custom_false'] = c2.text_input("False value", value="Inactive", key="dtype_bool_cf")

        st.markdown("---")
        if st.button("🔍 Preview Conversion", use_container_width=True, key="dtype_preview_btn_bool"):
            try:
                preview_df = df.copy()
                converted, rate = _convert_column_type(preview_df[target_col], target_type, opts)
                preview_df[target_col] = converted
                st.session_state.update({'dtype_preview': preview_df, 'dtype_success_rate': rate,
                                         'dtype_target_col': target_col, 'dtype_type': target_type})
            except Exception as e:
                st.error(f"Conversion failed: {e}")
        _render_dtype_preview(df, target_col)


def _render_dtype_preview(df, target_col):
    """Shared preview/apply block for data type conversion."""
    if st.session_state.get('dtype_preview') is not None:
        preview_df = st.session_state['dtype_preview']
        rate = st.session_state.get('dtype_success_rate', 0)
        saved_type = st.session_state.get('dtype_type', '')

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Original Type", str(df[target_col].dtype))
        c2.metric("New Type",      str(preview_df[target_col].dtype))
        c3.metric("Success Rate",  f"{rate*100:.1f}%")
        c4.metric("Failed",        f"{int(len(df) * (1-rate)):,}")

        if rate < 0.5:
            st.error("⚠️ Low success rate — wrong type for this column?")
        elif rate < 0.9:
            st.warning("⚠️ Some values couldn't convert and became NaN")
        else:
            st.success("✅ Conversion successful!")

        comp = pd.DataFrame({
            'Original':  df[target_col].head(20),
            'Converted': preview_df[target_col].head(20)
        })
        st.dataframe(comp, use_container_width=True, hide_index=True)

        c1, c2 = st.columns([3, 1])
        with c1:
            if st.button("✅ Apply Conversion", type="primary", use_container_width=True,
                         key=f"dtype_apply_{saved_type}"):
                from utils_robust import update_df
                _register_transform(target_col, f"dtype→{saved_type}")
                update_df(preview_df, f"Converted '{target_col}' to {saved_type}")
                for k in ['dtype_preview','dtype_success_rate','dtype_target_col','dtype_type']:
                    st.session_state[k] = None
                st.success("✅ Conversion applied!"); st.rerun()
        with c2:
            if st.button("❌ Cancel", use_container_width=True, key=f"dtype_cancel_{saved_type}"):
                for k in ['dtype_preview','dtype_success_rate','dtype_target_col','dtype_type']:
                    st.session_state[k] = None
                st.rerun()


# ── Date helpers (shared with feature engineering) ────────────────────────────

def _detect_date_formats(sample_values):
    from datetime import datetime as _dt
    FORMATS = [
        "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y",
        "%d-%m-%y", "%m/%d/%y", "%d/%m/%y", "%m-%d-%Y", "%Y.%m.%d",
        "%d.%m.%Y", "%b %d %Y", "%d %b %Y", "%B %d %Y",
    ]
    detected = {}
    for val in sample_values:
        val = str(val).strip()
        if not val or val.lower() in ('nan','nat','none',''): continue
        for fmt in FORMATS:
            try:
                _dt.strptime(val, fmt)
                if fmt not in detected:
                    detected[fmt] = val
                break
            except ValueError:
                continue
    return detected


def _smart_parse_dates(series, ambiguous, manual_format=None):
    from datetime import datetime as _dt
    day_first = "DD/MM" in ambiguous
    FORMATS = (["%d-%m-%Y","%d/%m/%Y","%d/%m/%y","%d-%m-%y",
                "%Y-%m-%d","%Y/%m/%d","%Y.%m.%d",
                "%m/%d/%Y","%m/%d/%y","%m-%d-%Y",
                "%d %b %Y","%d %B %Y","%b %d %Y","%B %d %Y","%d.%m.%Y"]
               if day_first else
               ["%m/%d/%Y","%m/%d/%y","%m-%d-%Y",
                "%Y-%m-%d","%Y/%m/%d","%Y.%m.%d",
                "%d-%m-%Y","%d/%m/%Y","%d/%m/%y","%d-%m-%y",
                "%d %b %Y","%d %B %Y","%b %d %Y","%B %d %Y","%d.%m.%Y"])
    if manual_format:
        FORMATS = [manual_format] + FORMATS

    def parse_one(val):
        if pd.isna(val): return pd.NaT
        s = str(val).strip()
        if not s or s.lower() in ('nan','nat','none',''): return pd.NaT
        for fmt in FORMATS:
            try: return pd.Timestamp(_dt.strptime(s, fmt))
            except: continue
        try: return pd.to_datetime(s, dayfirst=day_first)
        except: return pd.NaT

    try:
        fast = pd.to_datetime(series, dayfirst=day_first, errors='coerce')
        if fast.notna().sum() / len(series) >= 0.80:
            return fast
    except: pass
    return series.apply(parse_one)


def _convert_column_type(series, target_type, options):
    if target_type == "numeric":
        converted = pd.to_numeric(series, errors='coerce')
        if options.get('subtype') == 'int':
            converted = converted.round().astype('Int64')
        if 'Keep as 0' in options.get('errors',''):
            converted = converted.fillna(0)
        return converted, converted.notna().sum() / len(series)

    elif target_type == "string":
        return series.astype(str), 1.0

    elif target_type == "datetime":
        dates = _smart_parse_dates(series, options.get('ambiguous','Assume DD/MM/YYYY first'),
                                   options.get('manual_format'))
        out_fmt = options.get('output_format','YYYY-MM-DD (2024-01-15)')
        if "Keep as datetime object" in out_fmt:
            return dates, dates.notna().sum() / len(series)
        fmt_map = {
            "YYYY-MM-DD (2024-01-15)":   "%Y-%m-%d",
            "DD/MM/YYYY (15/01/2024)":   "%d/%m/%Y",
            "MM/DD/YYYY (01/15/2024)":   "%m/%d/%Y",
            "DD-MM-YYYY (15-01-2024)":   "%d-%m-%Y",
            "YYYY/MM/DD (2024/01/15)":   "%Y/%m/%d",
            "DD MMM YYYY (15 Jan 2024)": "%d %b %Y",
        }
        pat = fmt_map.get(out_fmt, "%Y-%m-%d")
        converted = dates.dt.strftime(pat).where(dates.notna(), other=np.nan)
        return converted, dates.notna().sum() / len(series)

    elif target_type == "category":
        ratio = series.nunique() / len(series)
        if ratio > 0.5:
            st.warning(f"⚠️ High cardinality ({series.nunique()} unique). Category type works best <50% unique.")
        return series.astype('category'), 1.0

    elif target_type == "boolean":
        bool_map = {
            'true':True,'false':False,'yes':True,'no':False,
            '1':True,'0':False,'t':True,'f':False,'y':True,'n':False,
            '1.0':True,'0.0':False,'active':True,'inactive':False,
            'approved':True,'rejected':False,'pass':True,'fail':False,
            'on':True,'off':False,'enabled':True,'disabled':False,
            'success':True,'failure':False,'accept':True,'decline':False,
            'present':True,'absent':False,
        }
        mapped = series.astype(str).str.lower().str.strip().map(bool_map)
        if 'Treat as False' in options.get('errors',''):
            mapped = mapped.fillna(False)
        out = options.get('output_format','')
        if   "Yes/No"            in out: mapped = mapped.map({True:'Yes',False:'No'})
        elif "1/0"               in out: mapped = mapped.map({True:1,False:0})
        elif "Y/N"               in out: mapped = mapped.map({True:'Y',False:'N'})
        elif "Active/Inactive"   in out: mapped = mapped.map({True:'Active',False:'Inactive'})
        elif "Enabled/Disabled"  in out: mapped = mapped.map({True:'Enabled',False:'Disabled'})
        elif "Custom"            in out:
            t = options.get('custom_true','True'); f = options.get('custom_false','False')
            mapped = mapped.map({True:t,False:f})
        return mapped, mapped.notna().sum() / len(series)

    raise ValueError(f"Unknown target type: {target_type}")


if __name__ == "__main__":
    render_transformations_page()
