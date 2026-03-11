"""
Analytics Assistant UI Component - UPDATED VERSION
Enhanced with data-aware recommendations display
"""

import streamlit as st
from typing import Dict, List, Any, Optional
from assistant_engine import AnalyticsAssistant, format_workflow_for_display
from assistant_config import EXAMPLE_QUERIES, PAGE_CAPABILITIES

# Try to import profiler for enhanced display
try:
    from assistant_profiler import format_recommendations_for_display
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False


# ============================================================================
# ASSISTANT STYLES - UPDATED
# ============================================================================

ASSISTANT_CSS = """
<style>
/* Assistant container */
.assistant-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

/* Assistant header */
.assistant-header {
    color: white;
    font-size: 1.3rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Assistant subtitle */
.assistant-subtitle {
    color: rgba(255,255,255,0.9);
    font-size: 0.85rem;
    margin: 0 0 1rem 0;
}

/* Quality Score Badge (NEW!) */
.quality-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 1.2rem;
    font-weight: 700;
    margin: 1rem 0;
}

.quality-grade-A {
    background: #28a745;
    color: white;
}

.quality-grade-B {
    background: #28a745;
    color: white;
}

.quality-grade-C {
    background: #ffc107;
    color: #333;
}

.quality-grade-D {
    background: #fd7e14;
    color: white;
}

.quality-grade-F {
    background: #dc3545;
    color: white;
}

/* Recommendation Card (NEW!) */
.recommendation-card {
    background: white;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    border-left: 4px solid #ccc;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.rec-critical {
    border-left-color: #dc3545;
    background: #fff5f5;
}

.rec-high {
    border-left-color: #fd7e14;
    background: #fff8f0;
}

.rec-medium {
    border-left-color: #ffc107;
    background: #fffbf0;
}

.rec-low {
    border-left-color: #28a745;
    background: #f0fff4;
}

.rec-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-weight: 600;
    color: #2c3e50;
    margin-bottom: 0.5rem;
}

.rec-icon {
    font-size: 1.2rem;
}

.rec-details {
    color: #555;
    font-size: 0.9rem;
    margin: 0.25rem 0;
}

.rec-action {
    color: #667eea;
    font-size: 0.9rem;
    font-weight: 500;
    margin-top: 0.5rem;
}

/* Workflow step */
.workflow-step {
    background: white;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    border-left: 4px solid #667eea;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.workflow-step-number {
    display: inline-block;
    background: #667eea;
    color: white;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    text-align: center;
    line-height: 24px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-right: 0.5rem;
}

.workflow-step-page {
    font-weight: 600;
    color: #2c3e50;
    margin-bottom: 0.25rem;
}

.workflow-step-action {
    color: #34495e;
    margin-left: 1.5rem;
}

.workflow-step-details {
    color: #7f8c8d;
    font-size: 0.85rem;
    margin-left: 1.5rem;
    margin-top: 0.25rem;
}

.workflow-step-guidance {
    color: #667eea;
    font-size: 0.85rem;
    margin-left: 1.5rem;
    margin-top: 0.25rem;
    font-style: italic;
}

/* Summary Stats (NEW!) */
.summary-stats {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.5rem;
    margin: 1rem 0;
}

.stat-box {
    background: rgba(255,255,255,0.1);
    padding: 0.75rem;
    border-radius: 8px;
    text-align: center;
}

.stat-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: white;
}

.stat-label {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.9);
    margin-top: 0.25rem;
}

/* Warning message */
.assistant-warning {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 0.75rem;
    border-radius: 6px;
    margin-bottom: 0.75rem;
}

/* Suggestion */
.assistant-suggestion {
    background: #d4edda;
    border-left: 4px solid #28a745;
    padding: 0.75rem;
    border-radius: 6px;
    margin-bottom: 0.75rem;
    font-size: 0.85rem;
}

/* Example query */
.example-query {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 6px;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.5rem;
    color: white;
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.3s;
}

.example-query:hover {
    background: rgba(255,255,255,0.2);
    border-color: rgba(255,255,255,0.4);
    transform: translateX(5px);
}
</style>
"""


# ============================================================================
# INITIALIZE ASSISTANT IN SESSION STATE
# ============================================================================

def initialize_assistant():
    """Initialize assistant in session state."""
    
    if 'assistant' not in st.session_state:
        st.session_state.assistant = None
    
    if 'assistant_history' not in st.session_state:
        st.session_state.assistant_history = []
    
    if 'show_examples' not in st.session_state:
        st.session_state.show_examples = True


# ============================================================================
# MAIN ASSISTANT UI COMPONENT - UPDATED
# ============================================================================

def render_assistant_sidebar():
    """
    Render the Analytics Assistant in the sidebar.
    UPDATED with data-aware features
    """
    
    # Apply styles
    st.markdown(ASSISTANT_CSS, unsafe_allow_html=True)
    
    # Initialize
    initialize_assistant()
    
    # Update assistant with current dataset
    df = st.session_state.get('df')
    
    if df is not None:
        from utils_robust import get_column_types
        col_types = get_column_types(df)
        
        if st.session_state.assistant is None:
            st.session_state.assistant = AnalyticsAssistant(df, col_types)
        else:
            st.session_state.assistant.update_dataset(df, col_types)
    
    # Render UI
    st.markdown("---")
    
    st.markdown("""
        <div class='assistant-container'>
            <div class='assistant-header'>
                🤖 Analytics Assistant
            </div>
            <div class='assistant-subtitle'>
                I'll analyze your data and guide you through the steps
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Dataset status
    if df is None:
        st.warning("⚠️ Please upload a dataset first to use the assistant")
        return
    
    # Show dataset info with stats
    col_types = get_column_types(df)
    
    with st.expander("📊 Dataset Info", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", f"{len(df):,}")
            st.metric("Numeric", len(col_types['numeric']))
        with col2:
            st.metric("Columns", len(df.columns))
            st.metric("Categorical", len(col_types['categorical']))
    
    # Query input
    st.markdown("**💬 What would you like to do?**")
    
    user_query = st.text_area(
        "Describe your goal",
        placeholder="Try: 'What should I do to clean my data?' or 'Check my data quality'",
        height=80,
        key="assistant_query",
        label_visibility="collapsed"
    )
    
    # Action buttons
    col1, col2 = st.columns([3, 1])
    
    with col1:
        generate_btn = st.button("✨ Analyze & Guide", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🔄", use_container_width=True, help="Clear"):
            st.session_state.assistant_query = ""
            st.rerun()
    
    # Process query
    if generate_btn and user_query.strip():
        with st.spinner("Analyzing your dataset..."):
            result = st.session_state.assistant.process_query(user_query)
            
            # Store in history
            st.session_state.assistant_history.append({
                'query': user_query,
                'result': result
            })
            
            # Display result
            _display_workflow_result(result)
    
    # Show examples
    if st.session_state.show_examples:
        with st.expander("💡 Example Queries", expanded=False):
            st.markdown("**Try asking:**")
            
            for example in EXAMPLE_QUERIES[:6]:
                if st.button(f"📝 {example}", key=f"example_{example[:30]}", use_container_width=True):
                    st.session_state.assistant_query = example
                    st.rerun()
    
    # Page help
    st.markdown("---")
    
    with st.expander("📚 Page Capabilities", expanded=False):
        for page, info in PAGE_CAPABILITIES.items():
            st.markdown(f"**{page}**")
            st.caption(info['description'])


def _display_workflow_result(result: Dict[str, Any]):
    """
    Display the workflow generation result.
    UPDATED with data-aware visualization
    """
    
    if not result['success']:
        st.error(result.get('error', 'Could not generate workflow'))
        
        if result.get('suggestions'):
            st.markdown("**Try these examples:**")
            for suggestion in result['suggestions']:
                st.info(suggestion)
        
        return
    
    # Check if this is a data-aware response (NEW!)
    if result.get('is_data_aware') and result.get('data_summary'):
        _display_data_aware_results(result)
    else:
        _display_regular_workflow(result)


def _display_data_aware_results(result: Dict[str, Any]):
    """
    NEW: Display data-aware analysis results
    """
    
    summary = result['data_summary']
    recommendations = result.get('recommendations', [])
    
    st.markdown("---")
    st.markdown("### 🔍 Data Quality Analysis")
    
    # Quality score
    score = summary['quality_score']
    grade = summary['quality_grade']
    
    st.markdown(f"""
        <div class='quality-badge quality-grade-{grade}'>
            Data Quality: {score:.0f}/100 (Grade {grade})
        </div>
    """, unsafe_allow_html=True)
    
    # Issue summary
    st.markdown("**📊 Issues Found:**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        critical = summary['by_priority']['critical']
        st.metric("🔴 Critical", critical)
    
    with col2:
        high = summary['by_priority']['high']
        st.metric("🟠 High", high)
    
    with col3:
        medium = summary['by_priority']['medium']
        st.metric("🟡 Medium", medium)
    
    with col4:
        low = summary['by_priority']['low']
        st.metric("🟢 Low", low)
    
    # Display recommendations by priority
    if recommendations:
        st.markdown("---")
        st.markdown("### 📋 Recommendations")
        
        # Group by priority
        critical_recs = [r for r in recommendations if r['priority'] == 'critical']
        high_recs = [r for r in recommendations if r['priority'] == 'high']
        medium_recs = [r for r in recommendations if r['priority'] == 'medium']
        low_recs = [r for r in recommendations if r['priority'] == 'low']
        
        # Show critical
        if critical_recs:
            st.markdown("#### 🔴 Critical Priority")
            for rec in critical_recs[:3]:
                _render_recommendation_card(rec, 'critical')
        
        # Show high
        if high_recs:
            st.markdown("#### 🟠 High Priority")
            for rec in high_recs[:3]:
                _render_recommendation_card(rec, 'high')
        
        # Show medium (collapsible)
        if medium_recs:
            with st.expander(f"🟡 Medium Priority ({len(medium_recs)} issues)"):
                for rec in medium_recs:
                    _render_recommendation_card(rec, 'medium')
        
        # Show low (collapsible)
        if low_recs:
            with st.expander(f"🟢 Low Priority ({len(low_recs)} optional improvements)"):
                for rec in low_recs:
                    _render_recommendation_card(rec, 'low')
    
    else:
        st.success("✅ No issues found! Your data looks clean.")
    
    # Suggestions
    if result.get('suggestions'):
        st.markdown("---")
        st.markdown("### 💡 Suggestions")
        for suggestion in result['suggestions']:
            st.markdown(f"""
                <div class='assistant-suggestion'>
                    {suggestion}
                </div>
            """, unsafe_allow_html=True)


def _render_recommendation_card(rec: Dict, priority: str):
    """Render a single recommendation card."""
    
    card_html = f"""
    <div class='recommendation-card rec-{priority}'>
        <div class='rec-header'>
            <span class='rec-icon'>{rec.get('icon', '•')}</span>
            <span>{rec['issue']}</span>
        </div>
        {f"<div class='rec-details'><strong>Column:</strong> {rec['column']}</div>" if rec.get('column') else ""}
        <div class='rec-details'>{rec['details']}</div>
        <div class='rec-action'>→ {rec['action']}</div>
        <div class='rec-details' style='margin-top: 0.5rem;'>
            <strong>Go to:</strong> {rec['page']} → {rec.get('tab', 'Main')}
        </div>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)


def _display_regular_workflow(result: Dict[str, Any]):
    """Display regular (non-data-aware) workflow results."""
    
    # Display warnings
    if result.get('warnings'):
        for warning in result['warnings']:
            st.warning(warning)
    
    # Display workflow
    workflow = result.get('workflow', [])
    
    if not workflow:
        st.info("No specific steps identified. Please be more specific.")
        return
    
    st.success(f"✅ Generated {len(workflow)}-step workflow")
    
    # Display confidence
    if result.get('confidence'):
        confidence = result['confidence']
        if confidence < 0.5:
            st.caption("⚠️ Low confidence - please review steps carefully")
        elif confidence < 0.8:
            st.caption("🟡 Medium confidence")
        else:
            st.caption("🟢 High confidence")
    
    st.markdown("---")
    st.markdown("### 📋 Your Workflow")
    
    for step in workflow:
        _render_workflow_step(step)
    
    # Display suggestions
    if result.get('suggestions'):
        st.markdown("---")
        st.markdown("### 💡 Additional Suggestions")
        for suggestion in result['suggestions']:
            st.markdown(f"""
                <div class='assistant-suggestion'>
                    {suggestion}
                </div>
            """, unsafe_allow_html=True)


def _render_workflow_step(step: Dict):
    """Render a single workflow step."""
    
    step_html = f"""
    <div class='workflow-step'>
        <div class='workflow-step-page'>
            <span class='workflow-step-number'>{step['step']}</span>
            <strong>{step['page']}</strong>
        </div>
        <div class='workflow-step-action'>
            → {step['action']}
        </div>
    """
    
    if 'details' in step:
        step_html += f"""
        <div class='workflow-step-details'>
            ℹ️ {step['details']}
        </div>
        """
    
    if 'guidance' in step:
        step_html += f"""
        <div class='workflow-step-guidance'>
            💡 {step['guidance']}
        </div>
        """
    
    step_html += "</div>"
    
    st.markdown(step_html, unsafe_allow_html=True)


# ============================================================================
# COMPACT VERSION
# ============================================================================

def render_compact_assistant():
    """Compact version for limited space."""
    
    initialize_assistant()
    
    df = st.session_state.get('df')
    
    if df is None:
        st.info("🤖 Upload data to use assistant")
        return
    
    from utils_robust import get_column_types
    col_types = get_column_types(df)
    
    if st.session_state.assistant is None:
        st.session_state.assistant = AnalyticsAssistant(df, col_types)
    else:
        st.session_state.assistant.update_dataset(df, col_types)
    
    st.markdown("---")
    st.markdown("### 🤖 Assistant")
    
    user_query = st.text_input(
        "What to do?",
        placeholder="e.g., What should I do?",
        key="compact_assistant_query"
    )
    
    if st.button("Analyze", use_container_width=True):
        if user_query.strip():
            result = st.session_state.assistant.process_query(user_query)
            
            if result['success']:
                if result.get('is_data_aware'):
                    summary = result.get('data_summary', {})
                    score = summary.get('quality_score', 0)
                    st.metric("Quality", f"{score:.0f}/100")
                    st.caption(f"{summary.get('total_issues', 0)} issues found")
                else:
                    workflow = result['workflow']
                    st.success(f"{len(workflow)} steps")
                    for step in workflow[:3]:
                        st.caption(f"{step['step']}. {step['page']}")
            else:
                st.error("Could not understand")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    
    with st.sidebar:
        render_assistant_sidebar()