"""
Export & Reports Page - Professional & Comprehensive
Beautiful reporting dashboard with exports, summaries, and quality analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import json


def render_export_reports_page():
    """
    Main export and reports page with 4 comprehensive sections.
    
    Features:
    - Multi-format exports with options
    - Visual processing summary
    - Interactive quality dashboard
    - Downloadable HTML report
    """
    
    # ========================================================================
    # PAGE HEADER
    # ========================================================================
    
    st.markdown("""
        <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                    padding: 2rem; border-radius: 12px; margin-bottom: 2rem;'>
            <h1 style='color: white; margin: 0; font-size: 2.5rem;'>💾 Export & Reports</h1>
            <p style='color: rgba(255,255,255,0.9); margin-top: 0.5rem; font-size: 1.1rem;'>
                Export your data and generate comprehensive quality reports
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
            
            Please load and process a dataset first, then return here to:
            - Export in multiple formats (CSV, Excel, JSON, Parquet)
            - View processing summary
            - Analyze data quality
            - Generate comprehensive reports
        """)
        return
    
    
    # ========================================================================
    # TABS
    # ========================================================================
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Export Data",
        "📊 Dataset Summary",
        "🔍 Quality Analysis",
        "📋 Full Report"
    ])
    
    
    # ========================================================================
    # TAB 1: EXPORT DATA
    # ========================================================================
    
    with tab1:
        render_export_section(df)
    
    
    # ========================================================================
    # TAB 2: DATASET SUMMARY
    # ========================================================================
    
    with tab2:
        render_summary_section(df)
    
    
    # ========================================================================
    # TAB 3: QUALITY ANALYSIS
    # ========================================================================
    
    with tab3:
        render_quality_section(df)
    
    
    # ========================================================================
    # TAB 4: FULL REPORT
    # ========================================================================
    
    with tab4:
        render_full_report_section(df)


# ============================================================================
# SECTION RENDERERS
# ============================================================================

def render_export_section(df: pd.DataFrame):
    """Export data in multiple formats with options."""
    
    st.markdown("### 📤 Export Your Dataset")
    st.caption("Download your processed data in various formats")
    
    # ====================================================================
    # EXPORT OPTIONS
    # ====================================================================
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        export_format = st.selectbox(
            "Export format",
            ["CSV", "Excel (XLSX)", "JSON", "Parquet", "SQL"],
            help="Choose the format that best suits your needs"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"**File size:** ~{_estimate_file_size(df, export_format)} MB")
    
    # Filename
    default_filename = f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    filename = st.text_input(
        "Filename (without extension)",
        value=default_filename,
        max_chars=100
    )
    
    
    # ====================================================================
    # FORMAT-SPECIFIC OPTIONS
    # ====================================================================
    
    st.markdown("---")
    st.markdown("**Export Options**")
    
    if export_format == "CSV":
        _render_csv_options()
    elif export_format == "Excel (XLSX)":
        _render_excel_options()
    elif export_format == "JSON":
        _render_json_options()
    elif export_format == "Parquet":
        _render_parquet_options()
    elif export_format == "SQL":
        _render_sql_options()
    
    
    # ====================================================================
    # PREVIEW
    # ====================================================================
    
    st.markdown("---")
    st.markdown("**📊 Export Preview**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        st.metric("Memory", f"{memory_mb:.2f} MB")
    with col4:
        st.metric("Format", export_format.split()[0])
    
    # Show first few rows
    st.dataframe(df.head(5), use_container_width=True)
    
    
    # ====================================================================
    # EXPORT BUTTON
    # ====================================================================
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("📥 Generate & Download", type="primary", use_container_width=True):
            _generate_export(df, filename, export_format)


def _render_csv_options():
    """CSV export options."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.csv_separator = st.selectbox(
            "Delimiter",
            [",", ";", "\t", "|"],
            format_func=lambda x: {"," : "Comma (,)", ";" : "Semicolon (;)", 
                                   "\t": "Tab", "|": "Pipe (|)"}[x]
        )
    
    with col2:
        st.session_state.csv_encoding = st.selectbox(
            "Encoding",
            ["utf-8", "latin-1", "cp1252"],
            help="UTF-8 is recommended for maximum compatibility"
        )
    
    st.session_state.csv_index = st.checkbox("Include row index", value=False)
    st.session_state.csv_header = st.checkbox("Include column headers", value=True)


def _render_excel_options():
    """Excel export options."""
    
    st.session_state.excel_sheet_name = st.text_input(
        "Sheet name",
        value="Data",
        max_chars=31
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.excel_index = st.checkbox("Include row index", value=False)
    
    with col2:
        st.session_state.excel_freeze = st.checkbox("Freeze header row", value=True)
    
    st.session_state.excel_autofilter = st.checkbox("Add auto-filter", value=True)


def _render_json_options():
    """JSON export options."""
    
    st.session_state.json_orient = st.selectbox(
        "JSON format",
        ["records", "columns", "index", "values"],
        format_func=lambda x: {
            "records": "Records (list of dicts)",
            "columns": "Columns (dict of lists)",
            "index": "Index (dict of dicts)",
            "values": "Values (list of lists)"
        }[x],
        help="'records' is most common for APIs"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.json_indent = st.checkbox("Pretty print (indented)", value=True)
    
    with col2:
        st.session_state.json_date_format = st.selectbox(
            "Date format",
            ["iso", "epoch"],
            format_func=lambda x: {"iso": "ISO 8601", "epoch": "Unix timestamp"}[x]
        )


def _render_parquet_options():
    """Parquet export options."""
    
    st.session_state.parquet_compression = st.selectbox(
        "Compression",
        ["snappy", "gzip", "brotli", "none"],
        help="Snappy offers good balance of speed and compression"
    )
    
    st.info("💡 Parquet is highly efficient for large datasets and preserves data types perfectly")


def _render_sql_options():
    """SQL export options."""
    
    st.session_state.sql_table_name = st.text_input(
        "Table name",
        value="data_table"
    )
    
    st.session_state.sql_dialect = st.selectbox(
        "SQL dialect",
        ["sqlite", "mysql", "postgresql"],
        format_func=lambda x: x.upper()
    )
    
    st.session_state.sql_if_exists = st.radio(
        "If table exists",
        ["replace", "append", "fail"],
        horizontal=True
    )


def _estimate_file_size(df: pd.DataFrame, format: str) -> str:
    """Estimate export file size."""
    
    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    
    # Rough estimates
    if format == "CSV":
        size = memory_mb * 0.8
    elif format == "Excel (XLSX)":
        size = memory_mb * 0.9
    elif format == "JSON":
        size = memory_mb * 1.2
    elif format == "Parquet":
        size = memory_mb * 0.3  # Highly compressed
    elif format == "SQL":
        size = memory_mb * 1.0
    else:
        size = memory_mb
    
    return f"{size:.2f}"


def _generate_export(df: pd.DataFrame, filename: str, format: str):
    """Generate and trigger download."""
    
    try:
        with st.spinner(f"Generating {format} file..."):
            
            if format == "CSV":
                csv_data = df.to_csv(
                    index=st.session_state.get('csv_index', False),
                    sep=st.session_state.get('csv_separator', ','),
                    encoding=st.session_state.get('csv_encoding', 'utf-8'),
                    header=st.session_state.get('csv_header', True)
                )
                st.download_button(
                    "⬇️ Download CSV File",
                    csv_data,
                    f"{filename}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            elif format == "Excel (XLSX)":
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(
                        writer,
                        sheet_name=st.session_state.get('excel_sheet_name', 'Data'),
                        index=st.session_state.get('excel_index', False)
                    )
                    
                    # Apply formatting if requested
                    if st.session_state.get('excel_freeze', True):
                        worksheet = writer.sheets[st.session_state.get('excel_sheet_name', 'Data')]
                        worksheet.freeze_panes = 'A2'
                
                st.download_button(
                    "⬇️ Download Excel File",
                    buffer.getvalue(),
                    f"{filename}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            elif format == "JSON":
                json_data = df.to_json(
                    orient=st.session_state.get('json_orient', 'records'),
                    indent=2 if st.session_state.get('json_indent', True) else None,
                    date_format=st.session_state.get('json_date_format', 'iso')
                )
                st.download_button(
                    "⬇️ Download JSON File",
                    json_data,
                    f"{filename}.json",
                    "application/json",
                    use_container_width=True
                )
            
            elif format == "Parquet":
                buffer = BytesIO()
                df.to_parquet(
                    buffer,
                    index=False,
                    compression=st.session_state.get('parquet_compression', 'snappy')
                )
                st.download_button(
                    "⬇️ Download Parquet File",
                    buffer.getvalue(),
                    f"{filename}.parquet",
                    "application/octet-stream",
                    use_container_width=True
                )
            
            elif format == "SQL":
                # Generate SQL CREATE and INSERT statements
                sql_statements = _generate_sql_statements(
                    df,
                    st.session_state.get('sql_table_name', 'data_table'),
                    st.session_state.get('sql_dialect', 'sqlite')
                )
                st.download_button(
                    "⬇️ Download SQL File",
                    sql_statements,
                    f"{filename}.sql",
                    "text/plain",
                    use_container_width=True
                )
        
        st.success(f"✅ {format} file ready for download!")
        
        from utils_robust import log_action
        log_action(f"Exported dataset as {format}: {filename}")
    
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")
        from utils_robust import log_action
        log_action(f"Export failed: {str(e)}", "ERROR")


def _generate_sql_statements(df: pd.DataFrame, table_name: str, dialect: str) -> str:
    """Generate SQL CREATE TABLE and INSERT statements."""
    
    # SQL type mapping
    type_map = {
        'int64': 'INTEGER',
        'float64': 'REAL',
        'object': 'TEXT',
        'bool': 'BOOLEAN',
        'datetime64[ns]': 'TIMESTAMP'
    }
    
    # CREATE TABLE
    sql = f"-- SQL Export for {table_name}\n"
    sql += f"-- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    sql += f"DROP TABLE IF EXISTS {table_name};\n\n"
    sql += f"CREATE TABLE {table_name} (\n"
    
    columns = []
    for col, dtype in df.dtypes.items():
        sql_type = type_map.get(str(dtype), 'TEXT')
        columns.append(f"    {col} {sql_type}")
    
    sql += ",\n".join(columns)
    sql += "\n);\n\n"
    
    # INSERT statements
    sql += f"-- Insert {len(df)} rows\n"
    
    for idx, row in df.head(1000).iterrows():  # Limit to 1000 rows for practicality
        values = []
        for val in row:
            if pd.isna(val):
                values.append("NULL")
            elif isinstance(val, str):
                # Escape single quotes by doubling them
                escaped_val = val.replace("'", "''")
                values.append(f"'{escaped_val}'")
            elif isinstance(val, (int, float)):
                values.append(str(val))
            else:
                values.append(f"'{str(val)}'")
        
        sql += f"INSERT INTO {table_name} VALUES ({', '.join(values)});\n"
    
    if len(df) > 1000:
        sql += f"\n-- Note: Only first 1000 rows exported. Total rows: {len(df)}\n"
    
    return sql


def render_summary_section(df: pd.DataFrame):
    """Render dataset summary with before/after comparison."""
    
    st.markdown("### 📊 Dataset Summary")
    st.caption("Compare original and current dataset state")
    
    
    # ====================================================================
    # COMPARISON METRICS
    # ====================================================================
    
    original_df = st.session_state.get('original_df', None)
    
    if original_df is not None:
        col1, col2 = st.columns(2)
        
        # Original dataset
        with col1:
            st.markdown("""
                <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 12px; 
                            border-left: 4px solid #6c757d;'>
                    <h4 style='margin-top: 0; color: #6c757d;'>📁 Original Dataset</h4>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            orig_rows = len(original_df)
            orig_cols = len(original_df.columns)
            orig_missing = original_df.isnull().sum().sum()
            orig_dups = original_df.duplicated().sum()
            orig_memory = original_df.memory_usage(deep=True).sum() / (1024 ** 2)
            
            st.metric("Rows", f"{orig_rows:,}")
            st.metric("Columns", orig_cols)
            st.metric("Missing Values", f"{orig_missing:,}")
            st.metric("Duplicates", f"{orig_dups:,}")
            st.metric("Memory", f"{orig_memory:.2f} MB")
        
        # Current dataset
        with col2:
            st.markdown("""
                <div style='background: #e7f3ff; padding: 1.5rem; border-radius: 12px; 
                            border-left: 4px solid #1976d2;'>
                    <h4 style='margin-top: 0; color: #1976d2;'>📊 Current Dataset</h4>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            curr_rows = len(df)
            curr_cols = len(df.columns)
            curr_missing = df.isnull().sum().sum()
            curr_dups = df.duplicated().sum()
            curr_memory = df.memory_usage(deep=True).sum() / (1024 ** 2)
            
            st.metric("Rows", f"{curr_rows:,}", delta=f"{curr_rows - orig_rows:+,}")
            st.metric("Columns", curr_cols, delta=f"{curr_cols - orig_cols:+,}")
            st.metric("Missing Values", f"{curr_missing:,}", delta=f"{curr_missing - orig_missing:+,}")
            st.metric("Duplicates", f"{curr_dups:,}", delta=f"{curr_dups - orig_dups:+,}")
            st.metric("Memory", f"{curr_memory:.2f} MB", delta=f"{curr_memory - orig_memory:+.2f} MB")
        
        
        # ================================================================
        # VISUAL COMPARISON
        # ================================================================
        
        st.markdown("---")
        st.markdown("**📈 Visual Comparison**")
        
        import plotly.graph_objects as go
        
        # Create comparison bar chart
        categories = ['Rows', 'Columns', 'Missing', 'Duplicates']
        original_values = [orig_rows, orig_cols, orig_missing, orig_dups]
        current_values = [curr_rows, curr_cols, curr_missing, curr_dups]
        
        fig = go.Figure(data=[
            go.Bar(name='Original', x=categories, y=original_values, marker_color='#6c757d'),
            go.Bar(name='Current', x=categories, y=current_values, marker_color='#1976d2')
        ])
        
        fig.update_layout(
            barmode='group',
            title='Original vs Current Dataset',
            xaxis_title='Metric',
            yaxis_title='Count',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Only current dataset available
        st.info("📌 Original dataset not available. Showing current dataset only.")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Missing", f"{df.isnull().sum().sum():,}")
        with col4:
            st.metric("Duplicates", f"{df.duplicated().sum():,}")
        with col5:
            memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
            st.metric("Memory", f"{memory_mb:.2f} MB")
    
    
    # ====================================================================
    # PROCESSING LOG
    # ====================================================================
    
    st.markdown("---")
    st.markdown("### 📋 Processing History")
    
    processing_log = st.session_state.get('processing_log', [])
    
    if processing_log:
        # Show stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Actions", len(processing_log))
        with col2:
            # Count action types if structured
            if isinstance(processing_log[0], dict):
                info_count = sum(1 for log in processing_log if log.get('level') == 'INFO')
                st.metric("INFO Actions", info_count)
            else:
                st.metric("Actions Today", len([l for l in processing_log if datetime.now().strftime('%Y-%m-%d') in str(l)]))
        with col3:
            if isinstance(processing_log[0], dict):
                error_count = sum(1 for log in processing_log if log.get('level') == 'ERROR')
                st.metric("Errors", error_count)
            else:
                st.metric("Last 24h", len(processing_log[-20:]))
        
        st.markdown("---")
        
        # Display log entries
        st.markdown("**Recent Actions (Last 20):**")
        
        # Create dataframe for better display
        if isinstance(processing_log[0], dict):
            log_df = pd.DataFrame(processing_log[-20:])
            
            # Color code by level
            def color_level(row):
                if row['level'] == 'ERROR':
                    return ['background-color: #ffebee'] * len(row)
                elif row['level'] == 'WARNING':
                    return ['background-color: #fff3e0'] * len(row)
                else:
                    return ['background-color: #e8f5e9'] * len(row)
            
            st.dataframe(
                log_df.style.apply(color_level, axis=1),
                use_container_width=True,
                hide_index=True
            )
        else:
            # Plain text logs
            for log_entry in processing_log[-20:]:
                st.text(log_entry)
        
        # Export log
        st.markdown("---")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("📥 Export Log", use_container_width=True):
                from utils_robust import export_log
                
                log_text = export_log('text')
                st.download_button(
                    "⬇️ Download Log",
                    log_text,
                    f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    "text/plain"
                )
    
    else:
        st.info("📝 No processing actions logged yet")


def render_quality_section(df: pd.DataFrame):
    """Render comprehensive quality analysis."""
    
    st.markdown("### 🔍 Data Quality Analysis")
    st.caption("Comprehensive quality assessment across multiple dimensions")
    
    
    # ====================================================================
    # OVERALL QUALITY SCORE
    # ====================================================================
    
    from utils_robust import calculate_data_quality_score
    
    with st.spinner("Calculating quality metrics..."):
        overall_score, dimensions = calculate_data_quality_score(df)
    
    # Score display with color coding
    if overall_score >= 90:
        color = "#28a745"
        status = "Excellent"
        icon = "🌟"
    elif overall_score >= 75:
        color = "#17a2b8"
        status = "Good"
        icon = "✓"
    elif overall_score >= 60:
        color = "#ffc107"
        status = "Fair"
        icon = "⚠️"
    else:
        color = "#dc3545"
        status = "Needs Improvement"
        icon = "❌"
    
    st.markdown(f"""
        <div style='background: {color}; padding: 2rem; border-radius: 12px; 
                    text-align: center; color: white; margin-bottom: 2rem;'>
            <div style='font-size: 4rem; font-weight: bold;'>{overall_score:.1f}</div>
            <div style='font-size: 1.5rem; margin-top: 0.5rem;'>{icon} {status}</div>
            <div style='font-size: 1rem; opacity: 0.9; margin-top: 0.5rem;'>Overall Quality Score</div>
        </div>
    """, unsafe_allow_html=True)
    
    
    # ====================================================================
    # QUALITY DIMENSIONS
    # ====================================================================
    
    st.markdown("**📊 Quality Dimensions**")
    
    col1, col2, col3 = st.columns(3)
    
    for i, (dimension, score) in enumerate(dimensions.items()):
        with [col1, col2, col3][i % 3]:
            # Progress bar with color
            if score >= 90:
                bar_color = "green"
            elif score >= 70:
                bar_color = "orange"
            else:
                bar_color = "red"
            
            st.markdown(f"**{dimension}**")
            st.progress(score / 100)
            st.caption(f"{score:.1f}%")
    
    
    # ====================================================================
    # DETAILED CHECKS
    # ====================================================================
    
    st.markdown("---")
    st.markdown("### 📋 Detailed Quality Checks")
    
    checks_tab1, checks_tab2, checks_tab3 = st.tabs([
        "Missing Values", "Duplicates & Outliers", "Data Types"
    ])
    
    # Missing Values Tab
    with checks_tab1:
        _render_missing_analysis(df)
    
    # Duplicates & Outliers Tab
    with checks_tab2:
        _render_duplicates_outliers(df)
    
    # Data Types Tab
    with checks_tab3:
        _render_data_types_analysis(df)


def _render_missing_analysis(df: pd.DataFrame):
    """Analyze missing values."""
    
    st.markdown("**Missing Values Analysis**")
    
    missing_summary = df.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0]
    
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Missing Cells", f"{missing_cells:,}")
    with col2:
        st.metric("Missing Percentage", f"{missing_pct:.2f}%")
    with col3:
        st.metric("Affected Columns", len(missing_cols))
    
    if len(missing_cols) > 0:
        st.markdown("---")
        st.markdown("**Columns with Missing Values:**")
        
        missing_df = pd.DataFrame({
            'Column': missing_cols.index,
            'Missing Count': missing_cols.values,
            'Missing %': (missing_cols.values / len(df) * 100).round(2),
            'Complete Rows': len(df) - missing_cols.values
        }).sort_values('Missing %', ascending=False)
        
        # Color code by severity
        def highlight_missing(row):
            pct = row['Missing %']
            if pct > 50:
                return ['background-color: #ffebee'] * len(row)
            elif pct > 20:
                return ['background-color: #fff3e0'] * len(row)
            else:
                return ['background-color: #fff9e0'] * len(row)
        
        st.dataframe(
            missing_df.style.apply(highlight_missing, axis=1),
            use_container_width=True,
            hide_index=True
        )
        
        # Visualization
        import plotly.express as px
        
        fig = px.bar(
            missing_df.head(10),
            x='Column',
            y='Missing %',
            title='Top 10 Columns by Missing Percentage',
            color='Missing %',
            color_continuous_scale='Reds'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.success("✅ No missing values detected!")


def _render_duplicates_outliers(df: pd.DataFrame):
    """Analyze duplicates and outliers."""
    
    # Duplicates
    st.markdown("**Duplicate Rows**")
    
    dup_count = df.duplicated().sum()
    dup_pct = (dup_count / len(df)) * 100 if len(df) > 0 else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Duplicate Rows", f"{dup_count:,}")
    with col2:
        st.metric("Duplicate %", f"{dup_pct:.2f}%")
    
    if dup_count > 0:
        st.warning(f"⚠️ Found {dup_count:,} duplicate rows ({dup_pct:.2f}%)")
        
        # Show sample duplicates
        with st.expander("👀 View Sample Duplicates"):
            duplicates = df[df.duplicated(keep=False)].sort_values(by=list(df.columns))
            st.dataframe(duplicates.head(10), use_container_width=True)
    else:
        st.success("✅ No duplicate rows found!")
    
    # Outliers
    st.markdown("---")
    st.markdown("**Outlier Detection**")
    
    from utils_robust import get_column_types, detect_outliers_iqr
    
    col_types = get_column_types(df)
    numeric_cols = col_types['numeric']
    
    if numeric_cols:
        outlier_summary = []
        
        for col in numeric_cols:
            result = detect_outliers_iqr(df, col)
            if result:
                count, lower, upper = result
                if count > 0:
                    outlier_summary.append({
                        'Column': col,
                        'Outliers': count,
                        'Percentage': f"{(count/len(df)*100):.2f}%",
                        'Lower Bound': f"{lower:.2f}",
                        'Upper Bound': f"{upper:.2f}"
                    })
        
        if outlier_summary:
            st.warning(f"⚠️ Outliers detected in {len(outlier_summary)} columns")
            
            outlier_df = pd.DataFrame(outlier_summary)
            st.dataframe(outlier_df, use_container_width=True, hide_index=True)
            
            # Visualization
            import plotly.express as px
            
            fig = px.bar(
                outlier_df,
                x='Column',
                y='Outliers',
                title='Outlier Count by Column',
                color='Outliers',
                color_continuous_scale='Oranges'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No significant outliers detected!")
    else:
        st.info("ℹ️ No numeric columns available for outlier detection")


def _render_data_types_analysis(df: pd.DataFrame):
    """Analyze data types distribution."""
    
    st.markdown("**Data Types Distribution**")
    
    from utils_robust import get_column_types
    
    col_types = get_column_types(df)
    
    # Count by type
    type_counts = {
        'Numeric': len(col_types['numeric']),
        'Categorical': len(col_types['categorical']),
        'DateTime': len(col_types['datetime']),
        'Boolean': len(col_types['boolean'])
    }
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Numeric", type_counts['Numeric'])
    with col2:
        st.metric("Categorical", type_counts['Categorical'])
    with col3:
        st.metric("DateTime", type_counts['DateTime'])
    with col4:
        st.metric("Boolean", type_counts['Boolean'])
    
    # Pie chart
    import plotly.graph_objects as go
    
    fig = go.Figure(data=[go.Pie(
        labels=list(type_counts.keys()),
        values=list(type_counts.values()),
        hole=0.4,
        marker=dict(colors=['#667eea', '#764ba2', '#f093fb', '#f5576c'])
    )])
    
    fig.update_layout(
        title='Column Type Distribution',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed breakdown
    st.markdown("---")
    st.markdown("**Detailed Type Breakdown:**")
    
    for type_name, columns in col_types.items():
        if columns:
            with st.expander(f"{type_name.title()} Columns ({len(columns)})"):
                st.write(", ".join(columns))


def render_full_report_section(df: pd.DataFrame):
    """Generate comprehensive HTML report."""
    
    st.markdown("### 📋 Comprehensive Data Report")
    st.caption("Generate a detailed HTML report with all analysis")
    
    st.info("""
        The full report includes:
        - Executive summary
        - Dataset statistics
        - Quality analysis
        - Column profiles
        - Visualizations
        - Recommendations
    """)
    
    if st.button("📊 Generate HTML Report", type="primary", use_container_width=False):
        with st.spinner("Generating comprehensive report..."):
            html_report = _generate_html_report(df)
            
            st.download_button(
                "⬇️ Download HTML Report",
                html_report,
                f"data_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                "text/html",
                use_container_width=False
            )
            
            st.success("✅ Report generated successfully!")


def _generate_html_report(df: pd.DataFrame) -> str:
    """Generate comprehensive HTML report."""
    
    from utils_robust import calculate_data_quality_score, get_column_types
    
    overall_score, dimensions = calculate_data_quality_score(df)
    col_types = get_column_types(df)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; }}
            h1 {{ color: #667eea; border-bottom: 3px solid #667eea; padding-bottom: 10px; }}
            h2 {{ color: #2c3e50; margin-top: 30px; }}
            .metric {{ display: inline-block; margin: 20px; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
            .metric-value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
            .metric-label {{ color: #666; margin-top: 5px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #667eea; color: white; }}
            .quality-score {{ font-size: 4em; text-align: center; color: #28a745; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Data Analysis Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Executive Summary</h2>
            <div class="metric">
                <div class="metric-value">{len(df):,}</div>
                <div class="metric-label">Total Rows</div>
            </div>
            <div class="metric">
                <div class="metric-value">{len(df.columns)}</div>
                <div class="metric-label">Total Columns</div>
            </div>
            <div class="metric">
                <div class="metric-value">{df.memory_usage(deep=True).sum()/(1024**2):.2f} MB</div>
                <div class="metric-label">Memory Usage</div>
            </div>
            
            <h2>Data Quality Score</h2>
            <div class="quality-score">{overall_score:.1f}/100</div>
            
            <h2>Quality Dimensions</h2>
            <table>
                <tr><th>Dimension</th><th>Score</th></tr>
    """
    
    for dim, score in dimensions.items():
        html += f"<tr><td>{dim}</td><td>{score:.1f}%</td></tr>"
    
    html += """
            </table>
            
            <h2>Column Types</h2>
            <table>
                <tr><th>Type</th><th>Count</th></tr>
    """
    
    for type_name, columns in col_types.items():
        html += f"<tr><td>{type_name.title()}</td><td>{len(columns)}</td></tr>"
    
    html += """
            </table>
            
            <h2>Missing Values</h2>
            <table>
                <tr><th>Column</th><th>Missing</th><th>Percentage</th></tr>
    """
    
    missing = df.isnull().sum()
    for col, count in missing[missing > 0].items():
        pct = (count / len(df)) * 100
        html += f"<tr><td>{col}</td><td>{count:,}</td><td>{pct:.2f}%</td></tr>"
    
    html += """
            </table>
        </div>
    </body>
    </html>
    """
    
    return html


# ============================================================================
# MAIN USAGE
# ============================================================================

if __name__ == "__main__":
    render_export_reports_page()