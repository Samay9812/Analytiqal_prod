"""
Feature Engineering Page - Fixed & Complete
Fixes:
  1. Double-transform guard: checks applied_transforms registry (set by
     Transformations page) and warns before applying the same transform again.
  2. Memory usage estimate for numeric interaction feature explosion.
  3. All existing features preserved and working.
"""

import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from typing import List, Dict
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from utils_robust import update_df, get_column_types, log_action


# ── Double-transform guard ────────────────────────────────────────────────────

def _warn_if_already_transformed(col: str, transform_name: str):
    """
    Check the transform registry written by the Transformations page.
    If the same (or related) transform was already applied to col, show a warning.
    Returns True if a warning was shown.
    """
    registry: Dict[str, List[str]] = st.session_state.get('applied_transforms', {})
    previous = registry.get(col, [])
    if not previous:
        return False

    # Check for overlap: same family of transforms
    family_map = {
        'log':           ['Log', 'log'],
        'sqrt':          ['Square Root', 'sqrt'],
        'normalize':     ['Normalize', 'normalize', '0-1'],
        'standardize':   ['Standardize', 'standardize', 'z-score', 'Z-score'],
        'reciprocal':    ['Reciprocal', 'reciprocal'],
        'cube_root':     ['Cube Root', 'cube_root'],
        'dtype→numeric': ['dtype→numeric'],
        'dtype→category':['dtype→category'],
    }

    def in_family(name, family_key):
        return any(kw.lower() in name.lower() for kw in family_map.get(family_key, [family_key]))

    current_family = next((k for k in family_map if in_family(transform_name, k)), transform_name)
    conflicts = [p for p in previous if in_family(p, current_family)]

    if conflicts:
        st.warning(
            f"⚠️ **Double-transform risk:** `{col}` already had **{', '.join(conflicts)}** "
            f"applied on the Transformations page.\n\n"
            f"Applying **{transform_name}** again will stack on top of that. "
            f"This is usually unintentional. Check the Transformations page history before continuing."
        )
        return True
    return False


def _register_fe_transform(col: str, name: str):
    """Register a Feature Engineering transform in the shared registry."""
    if 'applied_transforms' not in st.session_state:
        st.session_state['applied_transforms'] = {}
    st.session_state['applied_transforms'].setdefault(col, []).append(f"FE:{name}")


# ── Memory estimate helper ────────────────────────────────────────────────────

def _estimate_interaction_memory(df: pd.DataFrame, cols: List[str], n_pairs: int) -> str:
    """Estimate memory impact of adding n_pairs float64 columns."""
    bytes_per_col = len(df) * 8  # float64 = 8 bytes
    total_mb = (n_pairs * bytes_per_col) / (1024 ** 2)
    current_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    return f"{total_mb:.1f} MB (current: {current_mb:.1f} MB, new total: {current_mb + total_mb:.1f} MB)"


# ── Page entry point ──────────────────────────────────────────────────────────

def render_feature_engineering_page():
    st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    padding: 2rem; border-radius: 12px; margin-bottom: 2rem;'>
            <h1 style='color: white; margin: 0; font-size: 2.5rem;'>📈 Feature Engineering</h1>
            <p style='color: rgba(255,255,255,0.9); margin-top: 0.5rem; font-size: 1.1rem;'>
                Create powerful features to enhance your analysis and models
            </p>
        </div>
    """, unsafe_allow_html=True)

    df = st.session_state.get('df', None)
    if df is None or df.empty:
        st.info("📂 **No dataset loaded.** Please load a dataset from the 🏠 Dataset page first.")
        return

    col_types = get_column_types(df)
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

    st.markdown("### 📊 Current Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Columns",  len(df.columns))
    c2.metric("Numeric",        len(col_types['numeric']))
    c3.metric("Categorical",    len(col_types['categorical']))
    c4.metric("DateTime",       len(datetime_cols))
    st.markdown("---")

    # Show transform registry summary if anything is registered
    registry = st.session_state.get('applied_transforms', {})
    if registry:
        with st.expander("🔒 Transform History (from Transformations page)", expanded=False):
            st.caption("The following transforms were applied on the Transformations page. "
                       "Feature Engineering will warn you before applying the same type again.")
            for col, transforms in registry.items():
                st.write(f"**{col}:** {', '.join(transforms)}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📅 DateTime Features",
        "🔢 Mathematical Features",
        "📊 Aggregation Features",
        "🔗 Interaction Features",
        "🔄 Transformations"
    ])

    with tab1: render_datetime_features(df, col_types)
    with tab2: render_mathematical_features(df, col_types)
    with tab3: render_aggregation_features(df, col_types)
    with tab4: render_interaction_features(df, col_types)
    with tab5: render_transformation_features(df, col_types)


# ── Tab 1: DateTime Features ──────────────────────────────────────────────────

def render_datetime_features(df, col_types):
    st.markdown("### 📅 DateTime Feature Extraction")
    st.caption("Extract temporal components from datetime columns")

    with st.expander("ℹ️ Understanding DateTime Features"):
        st.markdown("""
        **Date Components:** Year, Month, Day, Day of Year
        **Time Components:** Hour, Minute, Second
        **Temporal Units:** Day of Week (0=Mon), Week of Year, Quarter
        **Boolean Flags:** Is Weekend, Is Month/Quarter/Year Start or End
        **Named:** Day Name, Month Name, Days in Month
        """)

    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    potential     = [c for c in df.columns if ('date' in c.lower() or 'time' in c.lower()) and c not in datetime_cols]

    if not datetime_cols and not potential:
        st.warning("⚠️ No datetime columns found. Convert a text column using the Transformations → Data Types tab.")
        return

    target_col = st.selectbox("Datetime column", list(set(datetime_cols + potential)), key="fe_dt_col")

    if df[target_col].dtype != 'datetime64[ns]':
        st.warning(f"⚠️ '{target_col}' is not datetime type. Convert it first.")
        if st.button("🔄 Convert to DateTime", type="primary", key="fe_dt_convert"):
            try:
                df[target_col] = pd.to_datetime(df[target_col], errors='coerce')
                update_df(df, f"Converted {target_col} to datetime"); st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Min Date",    df[target_col].min().strftime('%Y-%m-%d'))
    c2.metric("Max Date",    df[target_col].max().strftime('%Y-%m-%d'))
    c3.metric("Range (days)", f"{(df[target_col].max() - df[target_col].min()).days:,}")

    st.markdown("---")
    st.markdown("### 🎯 Select Features")

    r1c1, r1c2, r1c3 = st.columns(3)
    date_feats  = r1c1.multiselect("Date Components",  ["Year","Month","Day","Day of Year"],
                                    default=["Year","Month","Day"], key="fe_dt_date")
    time_feats  = r1c2.multiselect("Time Components",  ["Hour","Minute","Second"], key="fe_dt_time")
    temp_feats  = r1c3.multiselect("Temporal Units",   ["Day of Week","Week of Year","Quarter"],
                                    default=["Day of Week"], key="fe_dt_temp")

    r2c1, r2c2 = st.columns(2)
    bool_feats  = r2c1.multiselect("Boolean Flags",
        ["Is Weekend","Is Month Start","Is Month End",
         "Is Quarter Start","Is Quarter End","Is Year Start","Is Year End"],
        default=["Is Weekend"], key="fe_dt_bool")
    named_feats = r2c2.multiselect("Named Features",
        ["Day Name","Month Name","Days in Month"], key="fe_dt_named")

    all_feats = date_feats + time_feats + temp_feats + bool_feats + named_feats

    if not all_feats:
        st.info("Select at least one feature"); return

    st.markdown("---")
    if st.button("🔍 Generate Preview", use_container_width=True, key="fe_dt_preview"):
        prev = _generate_datetime_features(df, target_col, all_feats)
        st.session_state['fe_dt_prev'] = prev
        st.session_state['fe_dt_feats'] = all_feats
        st.session_state['fe_dt_target'] = target_col

    if 'fe_dt_prev' in st.session_state:
        prev    = st.session_state['fe_dt_prev']
        new_cols = [c for c in prev.columns if c not in df.columns]
        st.info(f"Creating {len(new_cols)} features")
        st.dataframe(prev[[target_col] + new_cols].head(10), use_container_width=True)

        c1, c2 = st.columns([3, 1])
        with c1:
            if st.button("✅ Apply Features", type="primary", use_container_width=True, key="fe_dt_apply"):
                update_df(prev, f"Extracted {len(new_cols)} datetime features from {target_col}")
                for k in ['fe_dt_prev','fe_dt_feats','fe_dt_target']: del st.session_state[k]
                st.success(f"✅ Created {len(new_cols)} features!"); st.rerun()
        with c2:
            if st.button("❌ Cancel", use_container_width=True, key="fe_dt_cancel"):
                for k in ['fe_dt_prev','fe_dt_feats','fe_dt_target']: del st.session_state[k]
                st.rerun()


def _generate_datetime_features(df, col, features):
    p = df.copy()
    try:
        if "Year"            in features: p[f"{col}_year"]           = p[col].dt.year
        if "Month"           in features: p[f"{col}_month"]          = p[col].dt.month
        if "Day"             in features: p[f"{col}_day"]            = p[col].dt.day
        if "Day of Year"     in features: p[f"{col}_dayofyear"]      = p[col].dt.dayofyear
        if "Hour"            in features: p[f"{col}_hour"]           = p[col].dt.hour
        if "Minute"          in features: p[f"{col}_minute"]         = p[col].dt.minute
        if "Second"          in features: p[f"{col}_second"]         = p[col].dt.second
        if "Day of Week"     in features: p[f"{col}_dayofweek"]      = p[col].dt.dayofweek
        if "Week of Year"    in features: p[f"{col}_weekofyear"]     = p[col].dt.isocalendar().week.astype(int)
        if "Quarter"         in features: p[f"{col}_quarter"]        = p[col].dt.quarter
        if "Is Weekend"      in features: p[f"{col}_is_weekend"]     = p[col].dt.dayofweek.isin([5,6]).astype(int)
        if "Is Month Start"  in features: p[f"{col}_is_month_start"] = p[col].dt.is_month_start.astype(int)
        if "Is Month End"    in features: p[f"{col}_is_month_end"]   = p[col].dt.is_month_end.astype(int)
        if "Is Quarter Start"in features: p[f"{col}_is_qtr_start"]  = p[col].dt.is_quarter_start.astype(int)
        if "Is Quarter End"  in features: p[f"{col}_is_qtr_end"]    = p[col].dt.is_quarter_end.astype(int)
        if "Is Year Start"   in features: p[f"{col}_is_year_start"]  = p[col].dt.is_year_start.astype(int)
        if "Is Year End"     in features: p[f"{col}_is_year_end"]    = p[col].dt.is_year_end.astype(int)
        if "Day Name"        in features: p[f"{col}_day_name"]       = p[col].dt.day_name()
        if "Month Name"      in features: p[f"{col}_month_name"]     = p[col].dt.month_name()
        if "Days in Month"   in features: p[f"{col}_days_in_month"]  = p[col].dt.days_in_month
    except Exception as e:
        st.error(f"Feature generation error: {e}")
    return p


# ── Tab 2: Mathematical Features ─────────────────────────────────────────────

def render_mathematical_features(df, col_types):
    st.markdown("### 🔢 Mathematical Feature Engineering")

    with st.expander("ℹ️ Understanding Mathematical Features"):
        st.markdown("""
        **Binary operations:** Add, Subtract, Multiply, Divide, Power, Max, Min, Average of two columns.
        **Polynomial features:** Raise a column to powers 2–5.
        **Row-wise statistics:** Sum/Mean/Median/Std across a set of columns per row.
        """)

    numeric_cols = col_types['numeric']
    if len(numeric_cols) < 1:
        st.warning("⚠️ No numeric columns available"); return

    # Binary
    st.markdown("### ➕ Binary Operations")
    if len(numeric_cols) >= 2:
        c1, c2, c3 = st.columns(3)
        col1 = c1.selectbox("First column",  numeric_cols, key="fe_math_col1")
        op   = c2.selectbox("Operation", [
            "Addition (+)","Subtraction (-)","Multiplication (*)","Division (/)",
            "Power (**)","Maximum","Minimum","Average"
        ], key="fe_math_op")
        col2 = c3.selectbox("Second column", numeric_cols, key="fe_math_col2")

        sym_map = {"Addition (+)":"plus","Subtraction (-)":"minus","Multiplication (*)":"times",
                   "Division (/)":"div","Power (**)":"pow","Maximum":"max","Minimum":"min","Average":"avg"}
        new_name = st.text_input("New column name",
            value=f"{col1}_{sym_map[op]}_{col2}", key="fe_math_name")

        if st.button("👁️ Preview Binary", use_container_width=True, key="fe_math_preview"):
            try:
                p = df.copy()
                if op == "Addition (+)":      p[new_name] = df[col1] + df[col2]
                elif op == "Subtraction (-)": p[new_name] = df[col1] - df[col2]
                elif op == "Multiplication (*)":p[new_name] = df[col1] * df[col2]
                elif op == "Division (/)":    p[new_name] = df[col1] / df[col2].replace(0, np.nan)
                elif op == "Power (**)":      p[new_name] = df[col1] ** df[col2]
                elif op == "Maximum":         p[new_name] = df[[col1,col2]].max(axis=1)
                elif op == "Minimum":         p[new_name] = df[[col1,col2]].min(axis=1)
                else:                         p[new_name] = df[[col1,col2]].mean(axis=1)
                st.session_state['fe_math_prev'] = p
                st.session_state['fe_math_name_key'] = new_name
                st.session_state['fe_math_op_key'] = op
            except Exception as e:
                st.error(f"Error: {e}")

        if 'fe_math_prev' in st.session_state:
            p = st.session_state['fe_math_prev']
            nm = st.session_state['fe_math_name_key']
            st.dataframe(p[[col1, col2, nm]].head(10), use_container_width=True)
            rc1,rc2,rc3 = st.columns(3)
            rc1.metric("Mean", f"{p[nm].mean():.3f}")
            rc2.metric("Min",  f"{p[nm].min():.3f}")
            rc3.metric("Max",  f"{p[nm].max():.3f}")
            bc1, bc2 = st.columns([3,1])
            with bc1:
                if st.button("✅ Create Feature", type="primary", use_container_width=True, key="fe_math_apply"):
                    update_df(p, f"Created feature: {nm}")
                    for k in ['fe_math_prev','fe_math_name_key','fe_math_op_key']: del st.session_state[k]
                    st.success(f"✅ Created '{nm}'"); st.rerun()
            with bc2:
                if st.button("❌ Cancel", use_container_width=True, key="fe_math_cancel"):
                    for k in ['fe_math_prev','fe_math_name_key','fe_math_op_key']: del st.session_state[k]
                    st.rerun()
    else:
        st.info("Need at least 2 numeric columns for binary operations")

    st.markdown("---")

    # Polynomial
    st.markdown("### 🔢 Polynomial Features")
    c1, c2 = st.columns([2,1])
    poly_col = c1.selectbox("Column", numeric_cols, key="fe_poly_col")
    degree   = c2.slider("Degree", 2, 5, 2, key="fe_poly_deg")

    if st.button("👁️ Preview Polynomial", use_container_width=True, key="fe_poly_preview"):
        p = df.copy()
        for d in range(2, degree+1):
            p[f"{poly_col}_power{d}"] = df[poly_col] ** d
        st.session_state['fe_poly_prev'] = p
        st.session_state['fe_poly_col_key'] = poly_col
        st.session_state['fe_poly_deg_key'] = degree

    if 'fe_poly_prev' in st.session_state:
        p   = st.session_state['fe_poly_prev']
        pc  = st.session_state['fe_poly_col_key']
        pd_ = st.session_state['fe_poly_deg_key']
        new_cols = [f"{pc}_power{d}" for d in range(2, pd_+1)]
        st.dataframe(p[[pc]+new_cols].head(10), use_container_width=True)
        if pd_ == 2:
            fig = px.scatter(p.head(200), x=pc, y=f"{pc}_power2",
                             title=f"{pc} vs {pc}²", opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)
        bc1, bc2 = st.columns([3,1])
        with bc1:
            if st.button("✅ Create Polynomial Features", type="primary",
                         use_container_width=True, key="fe_poly_apply"):
                update_df(p, f"Polynomial features for {pc} (degree {pd_})")
                for k in ['fe_poly_prev','fe_poly_col_key','fe_poly_deg_key']: del st.session_state[k]
                st.success(f"✅ Created {pd_-1} polynomial features!"); st.rerun()
        with bc2:
            if st.button("❌ Cancel", use_container_width=True, key="fe_poly_cancel"):
                for k in ['fe_poly_prev','fe_poly_col_key','fe_poly_deg_key']: del st.session_state[k]
                st.rerun()

    st.markdown("---")

    # Row-wise statistics
    st.markdown("### 📊 Row-wise Statistics")
    if len(numeric_cols) >= 2:
        stat_cols = st.multiselect("Columns for statistics", numeric_cols, key="fe_stat_cols")
        if len(stat_cols) >= 2:
            stat_ops = st.multiselect("Statistics", ["Sum","Mean","Median","Min","Max","Std","Range"],
                                      default=["Mean"], key="fe_stat_ops")
            if stat_ops and st.button("📊 Create Statistical Features", type="primary", key="fe_stat_apply"):
                try:
                    p = df.copy()
                    for sop in stat_ops:
                        cname = f"{'_'.join(stat_cols[:2])}_rowwise_{sop.lower()}"
                        if sop == "Sum":    p[cname] = df[stat_cols].sum(axis=1)
                        elif sop == "Mean": p[cname] = df[stat_cols].mean(axis=1)
                        elif sop == "Median":p[cname]= df[stat_cols].median(axis=1)
                        elif sop == "Min":  p[cname] = df[stat_cols].min(axis=1)
                        elif sop == "Max":  p[cname] = df[stat_cols].max(axis=1)
                        elif sop == "Std":  p[cname] = df[stat_cols].std(axis=1)
                        else:               p[cname] = df[stat_cols].max(axis=1) - df[stat_cols].min(axis=1)
                    update_df(p, f"Created {len(stat_ops)} row-wise stats")
                    st.success(f"✅ Created {len(stat_ops)} features!"); st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("Select at least 2 columns")


# ── Tab 3: Aggregation Features ───────────────────────────────────────────────

def render_aggregation_features(df, col_types):
    st.markdown("### 📊 Aggregation Features")
    st.caption("Group-based statistical features assigned back to each row")

    with st.expander("ℹ️ How aggregation features work"):
        st.markdown("""
        1. Group rows by a categorical column (e.g. Region)
        2. Compute a statistic on a numeric column within each group (e.g. mean of Sales)
        3. Assign that group statistic back to every row in that group

        This lets models compare individual values against their group average.
        """)

    numeric_cols     = col_types['numeric']
    categorical_cols = col_types['categorical']

    if not numeric_cols or not categorical_cols:
        st.warning("⚠️ Need both numeric and categorical columns"); return

    st.markdown("### 🎯 Single Aggregation")
    c1, c2, c3 = st.columns(3)
    grp_col  = c1.selectbox("Group by",  categorical_cols, key="fe_agg_grp")
    agg_col  = c2.selectbox("Aggregate", numeric_cols,     key="fe_agg_col")
    agg_func = c3.selectbox("Function",  ["mean","median","sum","min","max","std","count"],
                             key="fe_agg_func")

    with st.expander("👁️ Group preview"):
        st.dataframe(df.groupby(grp_col)[agg_col].agg(['count','mean','min','max']),
                     use_container_width=True)

    if st.button("📊 Create Aggregation Feature", type="primary",
                 use_container_width=True, key="fe_agg_create"):
        try:
            p = df.copy()
            new_name = f"{agg_col}_{agg_func}_by_{grp_col}"
            p[new_name] = df.groupby(grp_col)[agg_col].transform(agg_func)
            st.dataframe(p[[grp_col, agg_col, new_name]].head(10), use_container_width=True)
            rc1,rc2,rc3 = st.columns(3)
            rc1.metric("Original Mean", f"{df[agg_col].mean():.3f}")
            rc2.metric("Feature Mean",  f"{p[new_name].mean():.3f}")
            rc3.metric("Unique Values", f"{p[new_name].nunique()}")
            if st.button("✅ Apply", type="primary", use_container_width=True, key="fe_agg_apply"):
                update_df(p, f"Created aggregation: {new_name}")
                st.success(f"✅ Created '{new_name}'!"); st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("### 📊 Multiple Aggregations")

    c1, c2 = st.columns(2)
    multi_grp = c1.selectbox("Group by", categorical_cols, key="fe_magg_grp")
    multi_col = c2.selectbox("Aggregate", numeric_cols,    key="fe_magg_col")
    multi_fns = st.multiselect("Functions", ["mean","median","sum","min","max","std","count"],
                               default=["mean","std"], key="fe_magg_fns")

    if multi_fns and st.button("📊 Create Multiple Aggregations", type="primary", key="fe_magg_apply"):
        try:
            p = df.copy()
            for fn in multi_fns:
                p[f"{multi_col}_{fn}_by_{multi_grp}"] = df.groupby(multi_grp)[multi_col].transform(fn)
            update_df(p, f"Created {len(multi_fns)} aggregation features")
            st.success(f"✅ Created {len(multi_fns)} features!"); st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")


# ── Tab 4: Interaction Features ───────────────────────────────────────────────

def render_interaction_features(df, col_types):
    st.markdown("### 🔗 Interaction Features")
    st.caption("Capture combined effects of multiple features")

    with st.expander("ℹ️ Understanding Interactions"):
        st.markdown("""
        **Numeric × Numeric:** Pairwise operations (multiply, add, subtract, divide).
        **Categorical × Categorical:** String concatenation creating a combined label.
        """)

    numeric_cols     = col_types['numeric']
    categorical_cols = col_types['categorical']

    # Numeric × Numeric
    st.markdown("### 🔢 Numeric × Numeric Interactions")

    if len(numeric_cols) >= 2:
        sel_nums = st.multiselect("Select columns (2–10)", numeric_cols,
                                  max_selections=10, key="fe_int_nums")

        if len(sel_nums) >= 2:
            op = st.selectbox("Operation", ["Multiplication","Addition","Subtraction","Division"],
                              key="fe_int_op")
            n_pairs = len(list(combinations(sel_nums, 2)))

            # ── Memory estimate ────────────────────────────────────────────
            mem_est = _estimate_interaction_memory(df, sel_nums, n_pairs)
            st.info(f"💡 Will create **{n_pairs}** interaction columns.  \n"
                    f"📦 Estimated memory impact: **{mem_est}**")

            if n_pairs > 50:
                st.warning("⚠️ Large number of interactions. Consider selecting fewer columns to avoid memory pressure.")

            if st.button("🔗 Create Numeric Interactions", type="primary", key="fe_int_nums_apply"):
                try:
                    p = df.copy()
                    for c1, c2 in combinations(sel_nums, 2):
                        if op == "Multiplication": p[f"{c1}_x_{c2}"]    = df[c1] * df[c2]
                        elif op == "Addition":     p[f"{c1}_plus_{c2}"] = df[c1] + df[c2]
                        elif op == "Subtraction":  p[f"{c1}_minus_{c2}"]= df[c1] - df[c2]
                        else:                      p[f"{c1}_div_{c2}"]  = df[c1] / df[c2].replace(0, np.nan)
                    update_df(p, f"Created {n_pairs} numeric interactions ({op})")
                    st.success(f"✅ Created {n_pairs} interaction features!"); st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("Select at least 2 columns")
    else:
        st.info("Need at least 2 numeric columns")

    st.markdown("---")

    # Categorical × Categorical
    st.markdown("### 🏷️ Categorical × Categorical Interactions")

    if len(categorical_cols) >= 2:
        c1, c2 = st.columns(2)
        cat1 = c1.selectbox("First categorical",  categorical_cols, key="fe_int_cat1")
        cat2 = c2.selectbox("Second categorical", categorical_cols, key="fe_int_cat2")

        card1 = df[cat1].nunique()
        card2 = df[cat2].nunique()
        combined = card1 * card2
        st.info(f"💡 Will create up to **{combined:,}** unique combinations ({card1} × {card2})")
        if combined > 1000:
            st.warning("⚠️ High cardinality interaction. May not be useful for tree models.")

        with st.expander("👁️ Top 10 combinations preview"):
            combo = (df[cat1].astype(str) + "_" + df[cat2].astype(str)).value_counts().head(10)
            st.dataframe(combo, use_container_width=True)

        if st.button("🔗 Create Categorical Interaction", type="primary", key="fe_int_cat_apply"):
            try:
                p = df.copy()
                cname = f"{cat1}_x_{cat2}"
                p[cname] = df[cat1].astype(str) + "_" + df[cat2].astype(str)
                update_df(p, f"Created categorical interaction: {cname}")
                st.success(f"✅ Created '{cname}'!"); st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("Need at least 2 categorical columns")


# ── Tab 5: Transformations ────────────────────────────────────────────────────

def render_transformation_features(df, col_types):
    st.markdown("### 🔄 Feature Transformations")
    st.caption("Apply mathematical transformations to reshape feature distributions")

    with st.expander("ℹ️ When to use each transformation"):
        st.markdown("""
        | Transformation | When to use |
        |---|---|
        | **Log** | Right-skewed data; compresses large values |
        | **Square Root** | Mild right skew; gentler than log |
        | **Cube Root** | Works on negative values unlike log/sqrt |
        | **Normalize (0–1)** | Different feature scales; tree models don't need this |
        | **Standardize (Z-score)** | SVM, KNN, linear models sensitive to scale |
        | **Reciprocal** | Rate features (e.g. 1/time) |

        ⚠️ **Double-transform risk:** If you already applied Log or Standardize on the
        Transformations page, applying it again here will stack the effect silently.
        The system will warn you if it detects this.
        """)

    numeric_cols = col_types.get('numeric', [])
    if not numeric_cols:
        st.warning("⚠️ No numeric columns available"); return

    c1, c2 = st.columns([2, 1])
    transform_col = c1.selectbox("Column", numeric_cols, key="fe_trans_col")
    transformation = c2.selectbox("Transformation", [
        "Log", "Square Root", "Cube Root",
        "Normalize (0-1)", "Standardize (Z-score)", "Reciprocal"
    ], key=f"fe_trans_type_{transform_col}")

    # ── Double-transform check ─────────────────────────────────────────────
    _warn_if_already_transformed(transform_col, transformation)

    # Current distribution
    st.markdown("### 📊 Current Distribution")
    fig = px.histogram(df, x=transform_col, nbins=50,
                       title=f"Original: {transform_col}", marginal='box')
    st.plotly_chart(fig, use_container_width=True)

    prev_key  = f"fe_trans_prev_{transform_col}"
    name_key  = f"fe_trans_name_{transform_col}"

    if st.button("👁️ Preview Transformation",
                 key=f"fe_trans_preview_{transform_col}", use_container_width=True):
        try:
            p = df.copy()
            slug = transformation.lower().replace(' ','_').replace('(','').replace(')','').replace('-','_')
            new_name = f"{transform_col}_{slug}"

            if transformation == "Log":
                min_v = df[transform_col].min()
                p[new_name] = np.log1p(df[transform_col] - min_v + 1) if min_v <= 0 else np.log1p(df[transform_col])
            elif transformation == "Square Root":
                min_v = df[transform_col].min()
                p[new_name] = np.sqrt(df[transform_col] - min_v) if min_v < 0 else np.sqrt(df[transform_col])
            elif transformation == "Cube Root":
                p[new_name] = np.cbrt(df[transform_col])
            elif transformation == "Normalize (0-1)":
                mn, mx = df[transform_col].min(), df[transform_col].max()
                p[new_name] = (df[transform_col] - mn) / (mx - mn)
            elif transformation == "Standardize (Z-score)":
                p[new_name] = (df[transform_col] - df[transform_col].mean()) / df[transform_col].std()
            else:  # Reciprocal
                p[new_name] = 1 / df[transform_col].replace(0, np.nan)

            st.session_state[prev_key] = p
            st.session_state[name_key] = new_name
        except Exception as e:
            st.error(f"Transformation error: {e}")

    if prev_key in st.session_state:
        p       = st.session_state[prev_key]
        new_name= st.session_state[name_key]

        fig = make_subplots(rows=1, cols=2, subplot_titles=('Original', 'Transformed'))
        fig.add_trace(go.Histogram(x=df[transform_col], name='Original', nbinsx=50), row=1, col=1)
        fig.add_trace(go.Histogram(x=p[new_name],       name='Transformed', nbinsx=50), row=1, col=2)
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("**Original:**")
            st.write(f"Mean: {df[transform_col].mean():.3f}  |  Std: {df[transform_col].std():.3f}  |  Skew: {df[transform_col].skew():.3f}")
        with sc2:
            st.markdown("**Transformed:**")
            st.write(f"Mean: {p[new_name].mean():.3f}  |  Std: {p[new_name].std():.3f}  |  Skew: {p[new_name].skew():.3f}")

        bc1, bc2 = st.columns([3, 1])
        with bc1:
            if st.button("✅ Apply Transformation", type="primary",
                         use_container_width=True, key=f"fe_trans_apply_{transform_col}"):
                _register_fe_transform(transform_col, transformation)
                update_df(p, f"Applied {transformation} to {transform_col}")
                del st.session_state[prev_key]; del st.session_state[name_key]
                st.success("✅ Transformation applied!"); st.rerun()
        with bc2:
            if st.button("❌ Cancel", use_container_width=True,
                         key=f"fe_trans_cancel_{transform_col}"):
                del st.session_state[prev_key]; del st.session_state[name_key]
                st.rerun()


if __name__ == "__main__":
    render_feature_engineering_page()