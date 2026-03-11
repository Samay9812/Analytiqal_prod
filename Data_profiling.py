"""
Data Profiling Page - Professional & Robust
Comprehensive data profiling with interactive visualizations and insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import chi2_contingency
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def render_data_profiling_page():
    """
    Main data profiling page with comprehensive analysis.

    Features:
    - Interactive Plotly visualizations
    - Automated insights generation
    - Statistical tests and recommendations
    - Beautiful UI with metrics
    - Export capabilities
    """

    # ========================================================================
    # PAGE HEADER
    # ========================================================================

    st.markdown("""
        <div style='background: linear-gradient(135deg, #6B73FF 0%, #000DFF 100%);
                    padding: 2rem; border-radius: 12px; margin-bottom: 2rem;'>
            <h1 style='color: white; margin: 0; font-size: 2.5rem;'>🔍 Data Profiling</h1>
            <p style='color: rgba(255,255,255,0.9); margin-top: 0.5rem; font-size: 1.1rem;'>
                Comprehensive analysis and quality assessment of your dataset
            </p>
        </div>
    """, unsafe_allow_html=True)


    # ========================================================================
    # CHECK IF DATA EXISTS
    # ========================================================================

    df = st.session_state.get('df', None)

    if df is None or df.empty:
        st.info("""
            📂 **No dataset loaded**

            Please load a dataset from the **🏠 Dataset** page first.

            Once loaded, you'll be able to:
            - View statistical summaries
            - Analyze individual columns
            - Explore distributions
            - Examine correlations
            - Get automated insights
        """)
        return


    # ========================================================================
    # DATASET OVERVIEW DASHBOARD
    # ========================================================================

    st.markdown("### 📊 Dataset Overview")

    from utils_robust import get_column_types
    col_types = get_column_types(df)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Rows", f"{len(df):,}")

    with col2:
        st.metric("Total Columns", len(df.columns))

    with col3:
        st.metric("Numeric", len(col_types['numeric']))

    with col4:
        st.metric("Categorical", len(col_types['categorical']))

    with col5:
        memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
        st.metric("Memory", f"{memory_mb:.2f} MB")

    # Quick insights
    with st.expander("🔍 Quick Insights"):
        insights = _generate_quick_insights(df, col_types)
        for insight in insights:
            st.write(f"• {insight}")

    st.markdown("---")


    # ========================================================================
    # PROFILING TABS
    # ========================================================================

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Statistical Summary",
        "🔍 Column Analysis",
        "📉 Distribution Analysis",
        "🔗 Correlation Analysis",
        "🎯 Advanced Insights"
    ])

    with tab1:
        render_statistical_summary(df, col_types)

    with tab2:
        render_column_analysis(df, col_types)

    with tab3:
        render_distribution_analysis(df, col_types)

    with tab4:
        render_correlation_analysis(df, col_types)

    with tab5:
        render_advanced_insights(df, col_types)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _generate_quick_insights(df: pd.DataFrame, col_types: Dict) -> List[str]:
    """Generate automated quick insights about the dataset."""

    insights = []

    # Missing values
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    if missing_pct > 10:
        insights.append(f"⚠️ High missing values: {missing_pct:.1f}% of data is missing")
    elif missing_pct > 0:
        insights.append(f"Missing values detected: {missing_pct:.1f}% of data")
    else:
        insights.append("✅ No missing values found")

    # Duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        dup_pct = (dup_count / len(df)) * 100
        insights.append(f"⚠️ {dup_count:,} duplicate rows found ({dup_pct:.1f}%)")

    # Constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        insights.append(f"⚠️ {len(constant_cols)} constant columns detected")

    # High cardinality
    high_card = [col for col in col_types['categorical'] if df[col].nunique() > len(df) * 0.9]
    if high_card:
        insights.append(f"📊 {len(high_card)} high-cardinality categorical columns")

    # Skewed distributions
    if col_types['numeric']:
        skewed = [col for col in col_types['numeric'] if abs(df[col].skew()) > 2]
        if skewed:
            insights.append(f"📈 {len(skewed)} highly skewed numeric columns")

    return insights


# ============================================================================
# SECTION RENDERERS
# ============================================================================

def render_statistical_summary(df: pd.DataFrame, col_types: Dict):
    """Enhanced statistical summary with interactive tables."""

    st.markdown("### 📊 Statistical Summary")
    st.caption("Comprehensive statistics for numeric and categorical variables")

    with st.expander("ℹ️ Understanding Statistical Summary"):
        st.markdown("""
        The **Statistical Summary** provides quantitative insights into your dataset:

        **Numeric Columns:**
        - **Descriptive Statistics:** Mean, median, std, min/max, quartiles
        - **Skewness:** Measures asymmetry (< -1 or > 1 indicates high skew)
        - **Kurtosis:** Measures tail heaviness (> 3 indicates heavy tails/outliers)
        - **Missing %:** Proportion of missing values
        - **Zeros %:** Proportion of zero values (important for some models)

        **Categorical Columns:**
        - **Cardinality:** Number of unique values
        - **Mode:** Most frequent value
        - **Mode Frequency:** How often the mode appears
        - **Missing %:** Proportion of missing values

        👉 Use this to identify data quality issues and distribution patterns.
        """)

    # Numeric summary
    if col_types['numeric']:
        st.markdown("#### 🔢 Numeric Columns")

        numeric_stats = df[col_types['numeric']].describe().T
        numeric_stats['missing_%'] = (df[col_types['numeric']].isnull().sum() / len(df) * 100)
        numeric_stats['zeros_%'] = ((df[col_types['numeric']] == 0).sum() / len(df) * 100)
        numeric_stats['skewness'] = df[col_types['numeric']].skew()
        numeric_stats['kurtosis'] = df[col_types['numeric']].kurtosis()

        def interpret_skew(x):
            if abs(x) < 0.5:
                return "✅ Normal"
            elif abs(x) < 1:
                return "🟡 Moderate"
            else:
                return "🔴 High"

        numeric_stats['skew_level'] = numeric_stats['skewness'].apply(interpret_skew)

        display_cols = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max',
                       'skewness', 'skew_level', 'kurtosis', 'missing_%', 'zeros_%']

        st.dataframe(
            numeric_stats[display_cols],
            use_container_width=True,
            column_config={
                'count':     st.column_config.NumberColumn('Count',   format='%d'),
                'mean':      st.column_config.NumberColumn('Mean',    format='%.2f'),
                'std':       st.column_config.NumberColumn('Std Dev', format='%.2f'),
                'min':       st.column_config.NumberColumn('Min',     format='%.2f'),
                '25%':       st.column_config.NumberColumn('Q1',      format='%.2f'),
                '50%':       st.column_config.NumberColumn('Median',  format='%.2f'),
                '75%':       st.column_config.NumberColumn('Q3',      format='%.2f'),
                'max':       st.column_config.NumberColumn('Max',     format='%.2f'),
                'skewness':  st.column_config.NumberColumn('Skewness', format='%.2f'),
                'kurtosis':  st.column_config.NumberColumn('Kurtosis', format='%.2f'),
                'missing_%': st.column_config.ProgressColumn('Missing %', format='%.1f%%', min_value=0, max_value=100),
                'zeros_%':   st.column_config.ProgressColumn('Zeros %',   format='%.1f%%', min_value=0, max_value=100),
            }
        )

    st.markdown("---")

    # Categorical summary
    if col_types['categorical']:
        st.markdown("#### 📝 Categorical Columns")

        cat_info = []
        for col in col_types['categorical']:
            value_counts = df[col].value_counts()
            mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A'

            cat_info.append({
                'Column':         col,
                'Unique Values':  df[col].nunique(),
                'Most Common':    str(mode_val)[:30] + '...' if len(str(mode_val)) > 30 else str(mode_val),
                'Mode Frequency': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'Mode %':         (value_counts.iloc[0] / len(df) * 100) if len(value_counts) > 0 else 0,
                'Missing':        df[col].isnull().sum(),
                'Missing %':      (df[col].isnull().sum() / len(df) * 100)
            })

        cat_df = pd.DataFrame(cat_info)

        st.dataframe(
            cat_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Unique Values':  st.column_config.NumberColumn('Unique',     format='%d'),
                'Mode Frequency': st.column_config.NumberColumn('Mode Count', format='%d'),
                'Mode %':         st.column_config.ProgressColumn('Mode %',    format='%.1f%%', min_value=0, max_value=100),
                'Missing':        st.column_config.NumberColumn('Missing',    format='%d'),
                'Missing %':      st.column_config.ProgressColumn('Missing %', format='%.1f%%', min_value=0, max_value=100),
            }
        )


def render_column_analysis(df: pd.DataFrame, col_types: Dict):
    """Enhanced individual column analysis with visualizations."""

    st.markdown("### 🔍 Detailed Column Analysis")
    st.caption("Deep dive into individual column characteristics")

    with st.expander("ℹ️ Understanding Column Analysis"):
        st.markdown("""
        Analyze individual columns in depth:

        **Numeric Columns:**
        - Statistical measures and distribution metrics
        - Interactive histogram with statistics overlay
        - Box plot for outlier detection
        - Value distribution table

        **Categorical Columns:**
        - Value frequency analysis
        - Interactive bar chart
        - Top/bottom values
        - Unique value insights

        👉 Use this to understand column-specific patterns and anomalies.
        """)

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_col = st.selectbox(
            "Select column for detailed analysis",
            df.columns,
            help="Choose any column to analyze"
        )

    with col2:
        col_dtype = str(df[selected_col].dtype)
        st.metric("Data Type", col_dtype)

    st.markdown("---")

    st.markdown("### 📋 Basic Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Values", f"{len(df):,}")

    with col2:
        non_null = df[selected_col].count()
        st.metric("Non-Null", f"{non_null:,}")

    with col3:
        null_count = df[selected_col].isnull().sum()
        st.metric("Null", f"{null_count:,}")

    with col4:
        unique_count = df[selected_col].nunique()
        st.metric("Unique", f"{unique_count:,}")

    st.markdown("---")

    if pd.api.types.is_numeric_dtype(df[selected_col]):
        _render_numeric_column_analysis(df, selected_col)
    elif pd.api.types.is_datetime64_any_dtype(df[selected_col]):
        _render_datetime_column_analysis(df, selected_col)
    else:
        _render_categorical_column_analysis(df, selected_col)


def _render_numeric_column_analysis(df: pd.DataFrame, col: str):
    """Detailed numeric column analysis."""

    st.markdown("### 📊 Statistical Measures")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Mean",    f"{df[col].mean():.3f}")
        st.metric("Std Dev", f"{df[col].std():.3f}")

    with col2:
        st.metric("Median", f"{df[col].median():.3f}")
        st.metric("MAD",    f"{(df[col] - df[col].median()).abs().median():.3f}")

    with col3:
        st.metric("Min", f"{df[col].min():.3f}")
        st.metric("Max", f"{df[col].max():.3f}")

    with col4:
        st.metric("Skewness", f"{df[col].skew():.3f}")
        st.metric("Kurtosis", f"{df[col].kurtosis():.3f}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Range", f"{df[col].max() - df[col].min():.3f}")

    with col2:
        iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
        st.metric("IQR", f"{iqr:.3f}")

    with col3:
        cv = (df[col].std() / df[col].mean() * 100) if df[col].mean() != 0 else 0
        st.metric("CV %", f"{cv:.1f}%")

    with col4:
        zeros = (df[col] == 0).sum()
        st.metric("Zeros", f"{zeros:,}")

    st.markdown("---")

    st.markdown("### 📈 Distribution Visualization")

    tab1, tab2, tab3 = st.tabs(["Histogram", "Box Plot", "Statistics Table"])

    with tab1:
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=df[col].dropna(),
            nbinsx=50,
            name='Distribution',
            marker_color='lightblue',
            opacity=0.7
        ))

        fig.add_vline(
            x=df[col].mean(),
            line_dash="dash", line_color="red",
            annotation_text="Mean", annotation_position="top"
        )
        fig.add_vline(
            x=df[col].median(),
            line_dash="dash", line_color="green",
            annotation_text="Median", annotation_position="top"
        )

        fig.update_layout(
            title=f'Distribution of {col}',
            xaxis_title=col, yaxis_title='Frequency',
            height=500, showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = go.Figure()

        fig.add_trace(go.Box(
            y=df[col].dropna(),
            name=col,
            marker_color='coral',
            boxmean='sd'
        ))

        fig.update_layout(
            title=f'Box Plot of {col}',
            yaxis_title=col,
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers      = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)

        if outlier_count > 0:
            st.warning(f"⚠️ Detected {outlier_count:,} potential outliers using IQR method")
            st.write(f"**Outlier Bounds:** [{lower_bound:.3f}, {upper_bound:.3f}]")
        else:
            st.success("✅ No outliers detected using IQR method")

    with tab3:
        stats_data = {
            'Metric': [
                'Count', 'Mean', 'Median', 'Mode', 'Std Dev', 'Variance',
                'Min', 'Q1', 'Q2', 'Q3', 'Max', 'Range', 'IQR',
                'Skewness', 'Kurtosis', 'CV %'
            ],
            'Value': [
                f"{df[col].count():,}",
                f"{df[col].mean():.3f}",
                f"{df[col].median():.3f}",
                f"{df[col].mode()[0]:.3f}" if len(df[col].mode()) > 0 else "N/A",
                f"{df[col].std():.3f}",
                f"{df[col].var():.3f}",
                f"{df[col].min():.3f}",
                f"{df[col].quantile(0.25):.3f}",
                f"{df[col].quantile(0.50):.3f}",
                f"{df[col].quantile(0.75):.3f}",
                f"{df[col].max():.3f}",
                f"{df[col].max() - df[col].min():.3f}",
                f"{iqr:.3f}",
                f"{df[col].skew():.3f}",
                f"{df[col].kurtosis():.3f}",
                f"{cv:.2f}%"
            ]
        }

        st.dataframe(
            pd.DataFrame(stats_data),
            use_container_width=True,
            hide_index=True
        )


def _render_categorical_column_analysis(df: pd.DataFrame, col: str):
    """Detailed categorical column analysis."""

    st.markdown("### 📊 Value Distribution")

    value_counts = df[col].value_counts()
    value_pcts   = df[col].value_counts(normalize=True) * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Unique Values", f"{len(value_counts):,}")

    with col2:
        most_common = value_counts.index[0] if len(value_counts) > 0 else "N/A"
        st.metric("Most Common", str(most_common)[:20])

    with col3:
        mode_freq = value_counts.iloc[0] if len(value_counts) > 0 else 0
        st.metric("Mode Frequency", f"{mode_freq:,}")

    with col4:
        mode_pct = value_pcts.iloc[0] if len(value_pcts) > 0 else 0
        st.metric("Mode %", f"{mode_pct:.1f}%")

    st.markdown("---")

    st.markdown("### 📊 Top Values")

    tab1, tab2 = st.tabs(["Bar Chart", "Frequency Table"])

    with tab1:
        top_n      = st.slider("Number of top values to display", 5, 50, 10)
        top_values = value_counts.head(top_n)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=top_values.index.astype(str),
            y=top_values.values,
            text=top_values.values,
            textposition='outside',
            marker_color='lightblue'
        ))

        fig.update_layout(
            title=f'Top {top_n} Values in {col}',
            xaxis_title='Value', yaxis_title='Frequency',
            height=500, xaxis_tickangle=-45
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        freq_df = pd.DataFrame({
            'Value':      value_counts.index.astype(str),
            'Count':      value_counts.values,
            'Percentage': value_pcts.values
        })

        st.dataframe(
            freq_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Count':      st.column_config.NumberColumn('Count', format='%d'),
                'Percentage': st.column_config.ProgressColumn('Percentage', format='%.2f%%', min_value=0, max_value=100)
            }
        )

    st.markdown("---")
    st.markdown("### 🎯 Cardinality Analysis")

    cardinality_ratio = len(value_counts) / len(df)

    col1, col2 = st.columns([2, 1])

    with col1:
        if cardinality_ratio > 0.95:
            st.warning(f"⚠️ **High Cardinality** ({cardinality_ratio*100:.1f}%): Almost all values are unique")
            st.caption("Consider: This might be an ID column or require encoding strategies")
        elif cardinality_ratio > 0.5:
            st.info(f"🟡 **Medium Cardinality** ({cardinality_ratio*100:.1f}%): Moderate number of unique values")
            st.caption("Consider: May need careful encoding or grouping")
        else:
            st.success(f"✅ **Low Cardinality** ({cardinality_ratio*100:.1f}%): Good for categorical analysis")
            st.caption("Suitable for: One-hot encoding, grouping, visualization")

    with col2:
        st.metric("Cardinality Ratio", f"{cardinality_ratio*100:.1f}%")


def _render_datetime_column_analysis(df: pd.DataFrame, col: str):
    """Detailed datetime column analysis."""

    st.markdown("### 📅 Date/Time Analysis")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Min Date", df[col].min().strftime('%Y-%m-%d'))

    with col2:
        st.metric("Max Date", df[col].max().strftime('%Y-%m-%d'))

    with col3:
        date_range = (df[col].max() - df[col].min()).days
        st.metric("Range (days)", f"{date_range:,}")

    with col4:
        unique_dates = df[col].nunique()
        st.metric("Unique Dates", f"{unique_dates:,}")

    st.markdown("---")

    date_counts = df[col].value_counts().sort_index()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=date_counts.index,
        y=date_counts.values,
        mode='lines+markers',
        name='Frequency',
        line=dict(color='blue')
    ))

    fig.update_layout(
        title=f'Frequency Over Time: {col}',
        xaxis_title='Date', yaxis_title='Frequency',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def render_distribution_analysis(df: pd.DataFrame, col_types: Dict):
    """Enhanced distribution analysis with multiple plot types."""

    st.markdown("### 📉 Distribution Analysis")
    st.caption("Visualize and analyze data distributions")

    with st.expander("ℹ️ Understanding Distribution Analysis"):
        st.markdown("""
        Analyze how numeric data is distributed:

        **Visualizations:**
        - **Histogram:** Frequency distribution across bins
        - **Box Plot:** Quartiles, median, and outliers
        - **Violin Plot:** Distribution density and quartiles
        - **Q-Q Plot:** Test for normality
        - **ECDF:** Empirical cumulative distribution

        **Normality Tests:**
        - **Shapiro-Wilk:** Tests if data follows normal distribution
        - **Anderson-Darling:** More sensitive normality test

        👉 Use these to understand distribution shape and identify transformations needed.
        """)

    numeric_cols = col_types['numeric']

    if not numeric_cols:
        st.warning("⚠️ No numeric columns available for distribution analysis")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        dist_col = st.selectbox(
            "Select column to analyze",
            numeric_cols,
            key="dist_col"
        )

    with col2:
        st.markdown("**Quick Stats:**")
        st.write(f"Mean: {df[dist_col].mean():.2f}")
        st.write(f"Std: {df[dist_col].std():.2f}")

    st.markdown("---")

    st.markdown("### 📊 Distribution Plots")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Histogram & Density",
        "📦 Box & Violin",
        "📈 Q-Q Plot",
        "📉 ECDF"
    ])

    with tab1:
        _render_histogram_density(df, dist_col)

    with tab2:
        _render_box_violin(df, dist_col)

    with tab3:
        _render_qq_plot(df, dist_col)

    with tab4:
        _render_ecdf(df, dist_col)

    st.markdown("---")

    st.markdown("### 🔬 Normality Tests")

    col1, col2 = st.columns(2)

    with col1:
        sample = df[dist_col].dropna().sample(min(5000, len(df[dist_col].dropna())))
        shapiro_stat, shapiro_p = stats.shapiro(sample)

        st.markdown("**Shapiro-Wilk Test**")
        st.write(f"Statistic: {shapiro_stat:.4f}")
        st.write(f"P-value: {shapiro_p:.4f}")

        if shapiro_p > 0.05:
            st.success("✅ Data appears normally distributed (p > 0.05)")
        else:
            st.warning("⚠️ Data does not appear normally distributed (p ≤ 0.05)")

    with col2:
        anderson_result = stats.anderson(df[dist_col].dropna())

        st.markdown("**Anderson-Darling Test**")
        st.write(f"Statistic: {anderson_result.statistic:.4f}")
        st.write(f"Critical Value (5%): {anderson_result.critical_values[2]:.4f}")

        if anderson_result.statistic < anderson_result.critical_values[2]:
            st.success("✅ Data appears normally distributed")
        else:
            st.warning("⚠️ Data does not appear normally distributed")


def _render_histogram_density(df: pd.DataFrame, col: str):
    """Render histogram with density overlay."""

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Histogram(
            x=df[col].dropna(),
            name='Histogram',
            nbinsx=50,
            marker_color='lightblue',
            opacity=0.7
        ),
        secondary_y=False
    )

    from scipy.stats import gaussian_kde

    data    = df[col].dropna()
    kde     = gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 100)

    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=kde(x_range) * len(data) * (data.max() - data.min()) / 50,
            name='Density',
            line=dict(color='red', width=2)
        ),
        secondary_y=True
    )

    fig.add_vline(x=df[col].mean(),   line_dash="dash", line_color="green",  annotation_text="Mean")
    fig.add_vline(x=df[col].median(), line_dash="dash", line_color="orange", annotation_text="Median")

    fig.update_layout(title=f'Distribution of {col}', height=500, showlegend=True)
    fig.update_xaxes(title_text=col)
    fig.update_yaxes(title_text="Frequency", secondary_y=False)
    fig.update_yaxes(title_text="Density",   secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)


def _render_box_violin(df: pd.DataFrame, col: str):
    """Render box plot and violin plot side by side."""

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Box Plot', 'Violin Plot'))

    fig.add_trace(
        go.Box(y=df[col].dropna(), name='Box Plot', marker_color='coral', boxmean='sd'),
        row=1, col=1
    )

    fig.add_trace(
        go.Violin(
            y=df[col].dropna(),
            name='Violin Plot',
            box_visible=True,
            meanline_visible=True,
            fillcolor='lightseagreen',
            opacity=0.6
        ),
        row=1, col=2
    )

    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def _render_qq_plot(df: pd.DataFrame, col: str):
    """Render Q-Q plot for normality assessment."""

    data = df[col].dropna()
    qq   = stats.probplot(data, dist="norm")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=qq[0][0], y=qq[0][1],
        mode='markers', name='Data',
        marker=dict(color='blue', size=5)
    ))

    fig.add_trace(go.Scatter(
        x=qq[0][0],
        y=qq[1][1] + qq[1][0] * qq[0][0],
        mode='lines', name='Normal Distribution',
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(
        title=f'Q-Q Plot: {col}',
        xaxis_title='Theoretical Quantiles',
        yaxis_title='Sample Quantiles',
        height=500, showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
        **Interpretation:**
        - Points close to the line → Data is normally distributed
        - Points curve above/below → Skewed distribution
        - Points diverge at ends → Heavy or light tails
    """)


def _render_ecdf(df: pd.DataFrame, col: str):
    """Render empirical cumulative distribution function."""

    data = np.sort(df[col].dropna())
    y    = np.arange(1, len(data) + 1) / len(data)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data, y=y,
        mode='lines', name='ECDF',
        line=dict(color='blue', width=2)
    ))

    fig.add_hline(y=0.25, line_dash="dash", line_color="gray", annotation_text="Q1")
    fig.add_hline(y=0.50, line_dash="dash", line_color="gray", annotation_text="Median")
    fig.add_hline(y=0.75, line_dash="dash", line_color="gray", annotation_text="Q3")

    fig.update_layout(
        title=f'Empirical Cumulative Distribution: {col}',
        xaxis_title=col, yaxis_title='Cumulative Probability',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# CORRELATION ANALYSIS  (numeric + categorical)
# ============================================================================

def _compute_cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Compute Cramér's V association statistic between two categorical series.
    Returns a value in [0, 1] where 0 = no association, 1 = perfect association.
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2, _, _, _    = chi2_contingency(confusion_matrix)
    n                = confusion_matrix.sum().sum()
    phi2             = chi2 / n
    r, k             = confusion_matrix.shape
    # Bias-corrected version
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr    = r - (r - 1)**2 / (n - 1)
    kcorr    = k - (k - 1)**2 / (n - 1)
    denom    = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2corr / denom))


def render_correlation_analysis(df: pd.DataFrame, col_types: Dict):
    """Enhanced correlation analysis — numeric heatmap, pairplot, and Cramér's V."""

    st.markdown("### 🔗 Correlation Analysis")
    st.caption("Explore relationships between variables — numeric and categorical")

    with st.expander("ℹ️ Understanding Correlation Analysis"):
        st.markdown("""
        **Numeric correlations:**
        - **Pearson:** Linear relationships (−1 to 1)
        - **Spearman:** Monotonic relationships (rank-based)
        - **Kendall:** Ordinal associations (robust to outliers)

        **Categorical associations:**
        - **Cramér's V:** Measures association between two categorical columns (0 to 1).
          Based on the chi-square statistic. 0 = no association, 1 = perfect association.
          Values above 0.3 are generally considered meaningful.

        **Pairplot:**
        - Scatter matrix of all numeric columns.
        - Shows clusters, nonlinear patterns, and outliers simultaneously.

        **Strength guide (numeric):**
        |r| > 0.7 → Strong · 0.3–0.7 → Moderate · < 0.3 → Weak
        """)

    # ── Three sub-tabs ──────────────────────────────────────────────────────
    corr_tab1, corr_tab2, corr_tab3 = st.tabs([
        "🔢 Numeric Correlations",
        "📊 Scatter Matrix (Pairplot)",
        "🗂️ Categorical Associations (Cramér's V)"
    ])

    # ── Tab 1: Numeric correlation heatmap ──────────────────────────────────
    with corr_tab1:
        numeric_cols = col_types['numeric']

        if len(numeric_cols) < 2:
            st.warning("⚠️ Need at least 2 numeric columns for correlation analysis")
        else:
            c1, c2 = st.columns([2, 1])

            with c1:
                corr_method = st.selectbox(
                    "Correlation Method",
                    ["Pearson", "Spearman", "Kendall"],
                    help="Choose the correlation coefficient to compute",
                    key="corr_method"
                )

            with c2:
                threshold = st.slider(
                    "Highlight threshold",
                    min_value=0.0, max_value=1.0,
                    value=0.7, step=0.1,
                    help="Highlight correlations above this value",
                    key="corr_threshold"
                )

            method_map   = {"Pearson": "pearson", "Spearman": "spearman", "Kendall": "kendall"}
            corr_matrix  = df[numeric_cols].corr(method=method_map[corr_method])

            st.markdown("#### 🔥 Correlation Heatmap")

            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu_r',
                zmid=0,
                text=corr_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))

            fig.update_layout(
                title=f'{corr_method} Correlation Matrix',
                height=600,
                xaxis_tickangle=-45
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"#### 🔍 High Correlations (|r| > {threshold})")

            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > threshold:
                        high_corr.append({
                            'Variable 1':      corr_matrix.columns[i],
                            'Variable 2':      corr_matrix.columns[j],
                            'Correlation':     corr_val,
                            'Abs Correlation': abs(corr_val),
                            'Strength':        'Strong Positive' if corr_val > threshold else 'Strong Negative'
                        })

            if high_corr:
                high_corr_df = pd.DataFrame(high_corr).sort_values('Abs Correlation', ascending=False)

                st.dataframe(
                    high_corr_df[['Variable 1', 'Variable 2', 'Correlation', 'Strength']],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Correlation': st.column_config.NumberColumn('Correlation', format='%.3f')
                    }
                )

                st.markdown("---")
                st.markdown("#### 📊 Pairwise Scatter Plot")

                pair_idx = st.selectbox(
                    "Select correlation pair to visualize",
                    range(len(high_corr)),
                    format_func=lambda i: (
                        f"{high_corr[i]['Variable 1']} vs {high_corr[i]['Variable 2']} "
                        f"(r={high_corr[i]['Correlation']:.3f})"
                    ),
                    key="corr_pair_select"
                )

                var1 = high_corr[pair_idx]['Variable 1']
                var2 = high_corr[pair_idx]['Variable 2']

                fig = px.scatter(
                    df, x=var1, y=var2,
                    title=f'{var1} vs {var2}',
                    trendline='ols',
                    opacity=0.6
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.info(f"✅ No correlations found above threshold {threshold}")

    # ── Tab 2: Scatter matrix / pairplot ────────────────────────────────────
    with corr_tab2:
        numeric_cols = col_types['numeric']

        if len(numeric_cols) < 2:
            st.warning("⚠️ Need at least 2 numeric columns for a scatter matrix")
        else:
            st.markdown("#### 📊 Scatter Matrix")
            st.caption(
                "Each cell shows the relationship between two numeric variables. "
                "Diagonal cells show the distribution of each variable. "
                "Look for clusters, curves, and outliers."
            )

            # Let user pick which columns to include (cap default at 6 to avoid overload)
            default_cols = numeric_cols[:min(6, len(numeric_cols))]
            selected_pair_cols = st.multiselect(
                "Select columns to include (recommended: up to 6)",
                options=numeric_cols,
                default=default_cols,
                key="pairplot_cols"
            )

            if len(selected_pair_cols) < 2:
                st.warning("Select at least 2 columns.")
            else:
                # Optional: colour by a categorical column
                color_options = ["None"] + col_types['categorical']
                color_by      = st.selectbox(
                    "Colour points by (optional)",
                    color_options,
                    key="pairplot_color"
                )
                color_col = None if color_by == "None" else color_by

                with st.spinner("Building scatter matrix…"):
                    fig = px.scatter_matrix(
                        df,
                        dimensions=selected_pair_cols,
                        color=color_col,
                        opacity=0.5,
                        title="Scatter Matrix — Numeric Variables",
                    )
                    fig.update_traces(diagonal_visible=True, showupperhalf=True)
                    fig.update_layout(height=700)

                st.plotly_chart(fig, use_container_width=True)

                st.info("""
                    **How to read this:**
                    - **Diagonal** — distribution of each variable
                    - **Off-diagonal** — scatter plot between each pair
                    - **Linear pattern** → strong linear relationship
                    - **Curved pattern** → nonlinear relationship
                    - **Spread cloud** → weak or no relationship
                    - **Clusters** → possible grouping in the data
                """)

    # ── Tab 3: Cramér's V (categorical associations) ────────────────────────
    with corr_tab3:
        cat_cols = col_types['categorical']

        if len(cat_cols) < 2:
            st.warning("⚠️ Need at least 2 categorical columns for association analysis")
        else:
            st.markdown("#### 🗂️ Cramér's V Association Matrix")
            st.caption(
                "Cramér's V measures the strength of association between categorical columns. "
                "Values range from 0 (no association) to 1 (perfect association). "
                "Values above 0.3 are generally considered meaningful."
            )

            # Cap at 15 columns to keep computation fast
            MAX_CAT = 15
            if len(cat_cols) > MAX_CAT:
                st.info(
                    f"ℹ️ {len(cat_cols)} categorical columns found. "
                    f"Showing first {MAX_CAT} for performance. "
                    "Use the selector below to customise."
                )

            default_cat = cat_cols[:min(MAX_CAT, len(cat_cols))]
            selected_cat_cols = st.multiselect(
                "Select categorical columns to include",
                options=cat_cols,
                default=default_cat,
                key="cramers_cols"
            )

            if len(selected_cat_cols) < 2:
                st.warning("Select at least 2 columns.")
            else:
                with st.spinner("Computing Cramér's V matrix…"):
                    n   = len(selected_cat_cols)
                    mat = np.zeros((n, n))

                    for i in range(n):
                        for j in range(n):
                            if i == j:
                                mat[i, j] = 1.0
                            elif j > i:
                                v          = _compute_cramers_v(
                                    df[selected_cat_cols[i]].astype(str),
                                    df[selected_cat_cols[j]].astype(str)
                                )
                                mat[i, j] = v
                                mat[j, i] = v

                    cramers_df = pd.DataFrame(mat, index=selected_cat_cols, columns=selected_cat_cols)

                fig = go.Figure(data=go.Heatmap(
                    z=cramers_df.values,
                    x=cramers_df.columns,
                    y=cramers_df.columns,
                    colorscale='Blues',
                    zmin=0, zmax=1,
                    text=cramers_df.values,
                    texttemplate='%{text:.2f}',
                    textfont={"size": 10},
                    colorbar=dict(title="Cramér's V")
                ))

                fig.update_layout(
                    title="Cramér's V — Categorical Association Matrix",
                    height=max(400, 60 * len(selected_cat_cols)),
                    xaxis_tickangle=-45
                )

                st.plotly_chart(fig, use_container_width=True)

                # Strong associations table
                v_threshold = st.slider(
                    "Show pairs with Cramér's V above",
                    min_value=0.0, max_value=1.0,
                    value=0.3, step=0.05,
                    key="cramers_threshold"
                )

                strong_pairs = []
                for i in range(n):
                    for j in range(i + 1, n):
                        v = cramers_df.iloc[i, j]
                        if v >= v_threshold:
                            strong_pairs.append({
                                'Variable 1':  selected_cat_cols[i],
                                'Variable 2':  selected_cat_cols[j],
                                "Cramér's V":  round(v, 3),
                                'Association': (
                                    "🔴 Strong"   if v >= 0.6
                                    else "🟡 Moderate" if v >= 0.3
                                    else "🟢 Weak"
                                )
                            })

                if strong_pairs:
                    st.markdown(f"#### Pairs with Cramér's V ≥ {v_threshold}")
                    st.dataframe(
                        pd.DataFrame(strong_pairs).sort_values("Cramér's V", ascending=False),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Cramér's V": st.column_config.NumberColumn("Cramér's V", format="%.3f")
                        }
                    )
                else:
                    st.info(f"✅ No categorical associations found above threshold {v_threshold}")

                st.info("""
                    **Interpretation guide:**
                    - **V ≥ 0.6** → Strong association — variables may be redundant or structurally linked
                    - **0.3 ≤ V < 0.6** → Moderate association — worth investigating
                    - **V < 0.3** → Weak association — likely independent
                """)


# ============================================================================
# ADVANCED INSIGHTS
# ============================================================================

def render_advanced_insights(df: pd.DataFrame, col_types: Dict):
    """Advanced insights and automated recommendations."""

    st.markdown("### 🎯 Advanced Insights")
    st.caption("Automated analysis and recommendations")

    st.markdown("### 🏆 Data Quality Score")

    quality_metrics = _calculate_advanced_quality(df, col_types)

    col1, col2, col3 = st.columns(3)

    with col1:
        completeness = quality_metrics['completeness']
        color = "🟢" if completeness > 95 else "🟡" if completeness > 80 else "🔴"
        st.metric("Completeness", f"{completeness:.1f}%", help="% of non-missing values")
        st.caption(f"{color} Score")

    with col2:
        uniqueness = quality_metrics['uniqueness']
        color = "🟢" if uniqueness > 90 else "🟡" if uniqueness > 70 else "🔴"
        st.metric("Uniqueness", f"{uniqueness:.1f}%", help="% of unique rows")
        st.caption(f"{color} Score")

    with col3:
        consistency = quality_metrics['consistency']
        color = "🟢" if consistency > 90 else "🟡" if consistency > 70 else "🔴"
        st.metric("Consistency", f"{consistency:.1f}%", help="Proper data types & formats")
        st.caption(f"{color} Score")

    overall_score = (completeness + uniqueness + consistency) / 3

    st.markdown("---")

    progress_color = "🟢" if overall_score > 85 else "🟡" if overall_score > 70 else "🔴"
    st.markdown(f"### {progress_color} Overall Data Quality: {overall_score:.1f}/100")
    st.progress(overall_score / 100)

    st.markdown("---")

    st.markdown("### 💡 Automated Recommendations")

    recommendations = _generate_recommendations(df, col_types)

    for category, recs in recommendations.items():
        if recs:
            with st.expander(f"{category} ({len(recs)} recommendations)"):
                for rec in recs:
                    st.write(f"• {rec}")

    st.markdown("---")

    st.markdown("### 🔧 Feature Engineering Suggestions")

    suggestions = _generate_feature_suggestions(df, col_types)

    for suggestion in suggestions:
        st.info(suggestion)


def _calculate_advanced_quality(df: pd.DataFrame, col_types: Dict) -> Dict:
    """Calculate advanced quality metrics."""

    total_cells  = df.shape[0] * df.shape[1]
    non_missing  = total_cells - df.isnull().sum().sum()
    completeness = (non_missing / total_cells) * 100

    duplicates  = df.duplicated().sum()
    uniqueness  = ((len(df) - duplicates) / len(df)) * 100

    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    consistency   = ((len(df.columns) - len(constant_cols)) / len(df.columns)) * 100

    return {
        'completeness': completeness,
        'uniqueness':   uniqueness,
        'consistency':  consistency
    }


def _generate_recommendations(df: pd.DataFrame, col_types: Dict) -> Dict:
    """Generate automated recommendations."""

    recs = {
        '🧹 Data Cleaning':        [],
        '🔧 Data Transformation':  [],
        '📊 Feature Engineering':  [],
        '⚠️ Data Quality Warnings': []
    }

    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        recs['🧹 Data Cleaning'].append(
            f"Handle missing values in {len(missing_cols)} columns: "
            f"{', '.join(missing_cols[:3])}{'...' if len(missing_cols) > 3 else ''}"
        )

    if df.duplicated().sum() > 0:
        recs['🧹 Data Cleaning'].append(
            f"Remove {df.duplicated().sum():,} duplicate rows"
        )

    if col_types['numeric']:
        skewed = [col for col in col_types['numeric'] if abs(df[col].skew()) > 2]
        if skewed:
            recs['🔧 Data Transformation'].append(
                f"Apply log/sqrt transformation to {len(skewed)} skewed columns: {', '.join(skewed[:3])}"
            )

    high_card = [col for col in col_types['categorical'] if df[col].nunique() > len(df) * 0.9]
    if high_card:
        recs['⚠️ Data Quality Warnings'].append(
            f"Review high-cardinality columns: {', '.join(high_card[:3])} (may be IDs)"
        )

    constant = [col for col in df.columns if df[col].nunique() <= 1]
    if constant:
        recs['🧹 Data Cleaning'].append(
            f"Remove {len(constant)} constant columns: {', '.join(constant)}"
        )

    if len(col_types['numeric']) >= 2:
        corr_matrix = df[col_types['numeric']].corr().abs()
        high_corr   = (corr_matrix > 0.9).sum().sum() - len(corr_matrix)
        if high_corr > 0:
            recs['📊 Feature Engineering'].append(
                f"Consider removing {high_corr} highly correlated features"
            )

    return recs


def _generate_feature_suggestions(df: pd.DataFrame, col_types: Dict) -> List[str]:
    """Generate feature engineering suggestions."""

    suggestions = []

    datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    if datetime_cols:
        suggestions.append(
            f"💡 Extract features from datetime columns: year, month, day, "
            f"day_of_week, is_weekend from {', '.join(datetime_cols[:2])}"
        )

    if col_types['numeric']:
        suggestions.append(
            "💡 Create binned/categorical versions of continuous variables for grouping and analysis"
        )

    text_cols = [col for col in col_types['categorical'] if df[col].astype(str).str.len().mean() > 20]
    if text_cols:
        suggestions.append(
            f"💡 Extract text length features from: {', '.join(text_cols[:2])}"
        )

    if len(col_types['numeric']) >= 2:
        suggestions.append(
            "💡 Consider creating interaction features (products, ratios) between numeric variables"
        )

    return suggestions


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    render_data_profiling_page()