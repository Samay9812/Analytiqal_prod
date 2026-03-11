"""
Modern Navigation Component for Streamlit
Beautiful tab-style navigation with icons and active states
"""

import streamlit as st
from typing import List, Optional


# ============================================================================
# NAVIGATION STYLES
# ============================================================================

NAVIGATION_CSS = """
<style>
/* Hide default radio buttons */
div[data-testid="stRadio"] > div {
    display: none;
}

/* Navigation container */
.nav-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem 1.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
}

/* Navigation title */
.nav-title {
    color: white;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0 0 1.5rem 0;
    text-align: center;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

/* Navigation subtitle */
.nav-subtitle {
    color: rgba(255,255,255,0.9);
    font-size: 0.95rem;
    text-align: center;
    margin: -1rem 0 1.5rem 0;
}

/* Navigation pills container */
.nav-pills {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}

/* Navigation pill item */
.nav-pill {
    background: rgba(255, 255, 255, 0.1);
    border: 2px solid rgba(255, 255, 255, 0.2);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    color: white;
    cursor: pointer;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-size: 1rem;
    font-weight: 500;
    backdrop-filter: blur(10px);
}

.nav-pill:hover {
    background: rgba(255, 255, 255, 0.2);
    border-color: rgba(255, 255, 255, 0.4);
    transform: translateX(5px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}

/* Active pill */
.nav-pill.active {
    background: white;
    color: #667eea;
    border-color: white;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.nav-pill.active:hover {
    transform: translateX(8px);
}

/* Icon in pill */
.nav-pill-icon {
    font-size: 1.5rem;
    line-height: 1;
    min-width: 1.5rem;
    text-align: center;
}

/* Badge (optional - for notifications) */
.nav-badge {
    background: #ff4757;
    color: white;
    border-radius: 12px;
    padding: 0.2rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: auto;
}

/* Divider */
.nav-divider {
    height: 1px;
    background: rgba(255, 255, 255, 0.2);
    margin: 0.5rem 0;
}

/* Footer in navigation */
.nav-footer {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255, 255, 255, 0.2);
    color: rgba(255,255,255,0.8);
    font-size: 0.85rem;
    text-align: center;
}

/* Compact mode for smaller screens */
@media (max-width: 768px) {
    .nav-pill {
        padding: 0.75rem 1rem;
        font-size: 0.9rem;
    }
    
    .nav-pill-icon {
        font-size: 1.25rem;
    }
}
</style>
"""


# ============================================================================
# NAVIGATION COMPONENT
# ============================================================================

def render_modern_navigation(
    pages: List[str],
    current_page: Optional[str] = None,
    show_footer: bool = True
) -> str:
    """
    Render modern tab-style navigation with beautiful design.
    
    Args:
        pages: List of page names with emojis (e.g., "🏠 Dataset")
        current_page: Currently selected page
        show_footer: Whether to show footer with app info
        
    Returns:
        Selected page name
        
    Example:
        pages = ["🏠 Dataset", "🔄 Transformations", "📊 Visualizations"]
        selected = render_modern_navigation(pages)
    """
    
    # Apply CSS
    st.markdown(NAVIGATION_CSS, unsafe_allow_html=True)
    
    # Get current selection
    if current_page is None:
        current_page = st.session_state.get('current_page', pages[0])
    
    # Navigation container
    st.markdown("""
        <div class='nav-container'>
            <h1 class='nav-title'>📊 Data Analysis Pro</h1>
            <p class='nav-subtitle'>Professional Data Analysis Toolkit</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Create clickable pills (using actual Streamlit radio but styled)
    selected_page = st.radio(
        "Navigation",
        pages,
        index=pages.index(current_page) if current_page in pages else 0,
        key="main_navigation",
        label_visibility="collapsed"
    )
    
    # Custom pills rendering (visual only - for display)
    pills_html = "<div class='nav-pills'>"
    
    for page in pages:
        # Extract emoji and text
        parts = page.split(" ", 1)
        if len(parts) == 2:
            icon, text = parts
        else:
            icon, text = "•", page
        
        # Active state
        active_class = "active" if page == selected_page else ""
        
        # Create pill
        pills_html += f"""
        <div class='nav-pill {active_class}'>
            <span class='nav-pill-icon'>{icon}</span>
            <span>{text}</span>
        </div>
        """
    
    pills_html += "</div>"
    
    # Render pills (visual representation)
    st.markdown(pills_html, unsafe_allow_html=True)
    
    # Footer
    if show_footer:
        st.markdown("""
            <div class='nav-footer'>
                <strong>💡 Quick Tips</strong><br/>
                • Use Ctrl+Z for undo<br/>
                • All data stays on your device<br/>
                • Export anytime from Reports page
            </div>
        """, unsafe_allow_html=True)
    
    # Update session state
    if selected_page != st.session_state.get('current_page'):
        st.session_state.current_page = selected_page
    
    return selected_page


# ============================================================================
# ALTERNATIVE: HORIZONTAL TABS (for main content area)
# ============================================================================

def render_horizontal_tabs(tabs: List[str], default_index: int = 0) -> int:
    """
    Render horizontal tabs for content sections.
    
    Args:
        tabs: List of tab names
        default_index: Default selected tab index
        
    Returns:
        Selected tab index
        
    Example:
        tab_index = render_horizontal_tabs(["Overview", "Details", "Settings"])
        if tab_index == 0:
            # Show overview
        elif tab_index == 1:
            # Show details
    """
    
    tab_css = """
    <style>
    /* Horizontal tabs */
    .h-tabs {
        display: flex;
        gap: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }
    
    .h-tab {
        padding: 0.75rem 1.5rem;
        cursor: pointer;
        border: none;
        background: transparent;
        color: #666;
        font-weight: 500;
        border-bottom: 3px solid transparent;
        transition: all 0.3s ease;
        white-space: nowrap;
    }
    
    .h-tab:hover {
        color: #667eea;
        background: rgba(102, 126, 234, 0.05);
    }
    
    .h-tab.active {
        color: #667eea;
        border-bottom-color: #667eea;
        font-weight: 600;
    }
    </style>
    """
    
    st.markdown(tab_css, unsafe_allow_html=True)
    
    # Use Streamlit's built-in tabs (styled)
    return st.tabs(tabs)


# ============================================================================
# COMPACT NAVIGATION (for smaller sidebars)
# ============================================================================

def render_compact_navigation(pages: List[str], current_page: Optional[str] = None) -> str:
    """
    Compact navigation for narrow sidebars - icon-only with tooltips.
    
    Args:
        pages: List of page names with emojis
        current_page: Currently selected page
        
    Returns:
        Selected page name
    """
    
    compact_css = """
    <style>
    .compact-nav {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    
    .compact-nav-item {
        width: 100%;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 12px;
        border: 2px solid transparent;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
        font-size: 1.5rem;
    }
    
    .compact-nav-item:hover {
        background: #e9ecef;
        border-color: #667eea;
        transform: scale(1.05);
    }
    
    .compact-nav-item.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    </style>
    """
    
    st.markdown(compact_css, unsafe_allow_html=True)
    
    if current_page is None:
        current_page = st.session_state.get('current_page', pages[0])
    
    # Create compact nav
    nav_html = "<div class='compact-nav'>"
    
    for page in pages:
        icon = page.split(" ")[0] if " " in page else page
        active_class = "active" if page == current_page else ""
        
        nav_html += f"""
        <div class='compact-nav-item {active_class}' title='{page}'>
            {icon}
        </div>
        """
    
    nav_html += "</div>"
    
    st.markdown(nav_html, unsafe_allow_html=True)
    
    # Actual selection (hidden radio)
    selected = st.radio(
        "Navigation",
        pages,
        index=pages.index(current_page) if current_page in pages else 0,
        label_visibility="collapsed",
        key="compact_nav"
    )
    
    return selected


# ============================================================================
# MOBILE-FRIENDLY DROPDOWN NAVIGATION
# ============================================================================

def render_dropdown_navigation(pages: List[str], current_page: Optional[str] = None) -> str:
    """
    Dropdown-style navigation for mobile or compact layouts.
    
    Args:
        pages: List of page names
        current_page: Currently selected page
        
    Returns:
        Selected page name
    """
    
    st.markdown("### 📑 Navigation")
    
    if current_page is None:
        current_page = st.session_state.get('current_page', pages[0])
    
    # Use selectbox with custom styling
    selected = st.selectbox(
        "Go to page",
        pages,
        index=pages.index(current_page) if current_page in pages else 0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    return selected


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    
    # Define pages
    PAGES = [
        "🏠 Dataset",
        "🔄 Transformations", 
        "🔍 Data Profiling",
        "🧹 Data Cleaning",
        "📈 Feature Engineering",
        "📊 Visualizations",
        "💾 Export & Reports"
    ]
    
    # Example 1: Modern pill navigation (recommended)
    st.title("Navigation Examples")
    
    with st.sidebar:
        st.subheader("Style 1: Modern Pills (Recommended)")
        selected_page = render_modern_navigation(PAGES)
    
    st.write(f"Current page: **{selected_page}**")
    
    # Example 2: Horizontal tabs for sub-sections
    st.markdown("---")
    st.subheader("Style 2: Horizontal Tabs (for content sections)")
    
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📋 Details", "⚙️ Settings"])
    
    with tab1:
        st.write("Overview content here")
    
    with tab2:
        st.write("Detailed information")
    
    with tab3:
        st.write("Settings and configuration")