"""
Session Persistence Module for Data Analysis Pro
Allows users to save and load their analysis sessions
"""

import streamlit as st
import pickle
import os
import json
from datetime import datetime
from pathlib import Path
import shutil

# Create projects directory if it doesn't exist
PROJECTS_DIR = Path("saved_projects")
PROJECTS_DIR.mkdir(exist_ok=True)

# Metadata file to track all projects
PROJECTS_METADATA = PROJECTS_DIR / "projects_metadata.json"


def initialize_session_persistence():
    """Initialize session persistence on app startup"""
    if 'current_project_name' not in st.session_state:
        st.session_state.current_project_name = None
    if 'last_saved' not in st.session_state:
        st.session_state.last_saved = None
    if 'auto_save_enabled' not in st.session_state:
        st.session_state.auto_save_enabled = True


def get_session_data_to_save():
    """
    Extract all important session state data for saving
    Returns: Dictionary of session data
    """
    # List of session state keys to save
    keys_to_save = [
        'df',  # Main dataframe
        'original_df',  # Original dataframe backup
        'history',  # Undo history
        'redo_stack',  # Redo stack
        'processing_log',  # Action log
        'column_metadata',  # Column information
        'data_quality_score',  # Quality metrics
        'current_page',  # Which page user was on
        # Add any other custom keys your app uses
    ]
    
    session_data = {}
    for key in keys_to_save:
        if key in st.session_state:
            session_data[key] = st.session_state[key]
    
    return session_data


def save_project(project_name):
    """
    Save current session to disk
    
    Args:
        project_name: Name of the project to save
    
    Returns:
        bool: True if save successful, False otherwise
    """
    try:
        # Create safe filename
        safe_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        
        # Create project folder
        project_folder = PROJECTS_DIR / safe_name
        project_folder.mkdir(exist_ok=True)
        
        # Get session data
        session_data = get_session_data_to_save()
        
        # Save session data as pickle
        session_file = project_folder / "session_data.pkl"
        with open(session_file, 'wb') as f:
            pickle.dump(session_data, f)
        
        # Save metadata
        metadata = {
            'project_name': project_name,
            'safe_name': safe_name,
            'created_date': datetime.now().isoformat(),
            'last_modified': datetime.now().isoformat(),
            'row_count': len(session_data.get('df', [])) if 'df' in session_data else 0,
            'column_count': len(session_data.get('df', {}).columns) if 'df' in session_data else 0,
        }
        
        metadata_file = project_folder / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update global projects list
        update_projects_metadata(metadata)
        
        # Update session state
        st.session_state.current_project_name = project_name
        st.session_state.last_saved = datetime.now()
        
        return True
        
    except Exception as e:
        st.error(f"Error saving project: {str(e)}")
        return False


def load_project(project_name):
    """
    Load a saved project
    
    Args:
        project_name: Name of the project to load
    
    Returns:
        bool: True if load successful, False otherwise
    """
    try:
        # Find project folder
        projects = list_projects()
        project_meta = next((p for p in projects if p['project_name'] == project_name), None)
        
        if not project_meta:
            st.error(f"Project '{project_name}' not found")
            return False
        
        safe_name = project_meta['safe_name']
        project_folder = PROJECTS_DIR / safe_name
        session_file = project_folder / "session_data.pkl"
        
        if not session_file.exists():
            st.error(f"Project data file not found")
            return False
        
        # Load session data
        with open(session_file, 'rb') as f:
            session_data = pickle.load(f)
        
        # Restore session state
        for key, value in session_data.items():
            st.session_state[key] = value
        
        # Update metadata
        st.session_state.current_project_name = project_name
        st.session_state.last_saved = datetime.fromisoformat(project_meta['last_modified'])
        
        return True
        
    except Exception as e:
        st.error(f"Error loading project: {str(e)}")
        return False


def delete_project(project_name):
    """
    Delete a saved project
    
    Args:
        project_name: Name of the project to delete
    
    Returns:
        bool: True if delete successful, False otherwise
    """
    try:
        projects = list_projects()
        project_meta = next((p for p in projects if p['project_name'] == project_name), None)
        
        if not project_meta:
            return False
        
        safe_name = project_meta['safe_name']
        project_folder = PROJECTS_DIR / safe_name
        
        if project_folder.exists():
            shutil.rmtree(project_folder)
        
        # Update global metadata
        remove_from_projects_metadata(safe_name)
        
        return True
        
    except Exception as e:
        st.error(f"Error deleting project: {str(e)}")
        return False


def list_projects():
    """
    Get list of all saved projects
    
    Returns:
        list: List of project metadata dictionaries
    """
    try:
        if PROJECTS_METADATA.exists():
            with open(PROJECTS_METADATA, 'r') as f:
                metadata = json.load(f)
                return metadata.get('projects', [])
        return []
    except:
        return []


def update_projects_metadata(project_meta):
    """Update the global projects metadata file"""
    try:
        if PROJECTS_METADATA.exists():
            with open(PROJECTS_METADATA, 'r') as f:
                data = json.load(f)
        else:
            data = {'projects': []}
        
        # Update or add project
        existing = next((i for i, p in enumerate(data['projects']) 
                        if p['safe_name'] == project_meta['safe_name']), None)
        
        if existing is not None:
            data['projects'][existing] = project_meta
        else:
            data['projects'].append(project_meta)
        
        with open(PROJECTS_METADATA, 'w') as f:
            json.dump(data, f, indent=2)
            
    except Exception as e:
        print(f"Error updating metadata: {e}")


def remove_from_projects_metadata(safe_name):
    """Remove project from global metadata"""
    try:
        if PROJECTS_METADATA.exists():
            with open(PROJECTS_METADATA, 'r') as f:
                data = json.load(f)
            
            data['projects'] = [p for p in data['projects'] if p['safe_name'] != safe_name]
            
            with open(PROJECTS_METADATA, 'w') as f:
                json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error removing from metadata: {e}")


def auto_save():
    """Auto-save current project if enabled and project is loaded"""
    if (st.session_state.get('auto_save_enabled', False) and 
        st.session_state.get('current_project_name')):
        save_project(st.session_state.current_project_name)


def render_project_manager_ui():
    """
    Render the project management UI in sidebar
    Should be called in sidebar.py
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💾 Project Manager")
    
    # Current project info
    if st.session_state.get('current_project_name'):
        st.sidebar.success(f"📂 {st.session_state.current_project_name}")
        if st.session_state.get('last_saved'):
            time_diff = datetime.now() - st.session_state.last_saved
            minutes = int(time_diff.total_seconds() / 60)
            st.sidebar.caption(f"Last saved: {minutes} min ago")
    else:
        st.sidebar.info("No project loaded")
    
    # Save/Load buttons
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("💾 Save", use_container_width=True):
            if st.session_state.get('current_project_name'):
                if save_project(st.session_state.current_project_name):
                    st.sidebar.success("Saved!")
                    st.rerun()
            else:
                st.session_state.show_save_dialog = True
    
    with col2:
        if st.button("📂 Load", use_container_width=True):
            st.session_state.show_load_dialog = True
    
    # Auto-save toggle
    auto_save_enabled = st.sidebar.checkbox(
        "Auto-save (every 5 min)",
        value=st.session_state.get('auto_save_enabled', True),
        key='auto_save_checkbox'
    )
    st.session_state.auto_save_enabled = auto_save_enabled
    
    # Save dialog
    if st.session_state.get('show_save_dialog', False):
        with st.sidebar.expander("💾 Save Project", expanded=True):
            project_name = st.text_input(
                "Project Name",
                value=st.session_state.get('current_project_name', ''),
                placeholder="My Analysis Project"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save", use_container_width=True, type="primary"):
                    if project_name.strip():
                        if save_project(project_name):
                            st.success(f"✅ Saved '{project_name}'")
                            st.session_state.show_save_dialog = False
                            st.rerun()
                    else:
                        st.error("Please enter a project name")
            
            with col2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.show_save_dialog = False
                    st.rerun()
    
    # Load dialog
    if st.session_state.get('show_load_dialog', False):
        with st.sidebar.expander("📂 Load Project", expanded=True):
            projects = list_projects()
            
            if projects:
                # Sort by last modified
                projects.sort(key=lambda x: x['last_modified'], reverse=True)
                
                for project in projects:
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{project['project_name']}**")
                            st.caption(f"{project['row_count']} rows × {project['column_count']} cols")
                            modified = datetime.fromisoformat(project['last_modified'])
                            st.caption(f"Modified: {modified.strftime('%Y-%m-%d %H:%M')}")
                        
                        with col2:
                            if st.button("Load", key=f"load_{project['safe_name']}", use_container_width=True):
                                if load_project(project['project_name']):
                                    st.success(f"✅ Loaded '{project['project_name']}'")
                                    st.session_state.show_load_dialog = False
                                    st.rerun()
                        
                        with col3:
                            if st.button("🗑️", key=f"del_{project['safe_name']}", use_container_width=True):
                                if delete_project(project['project_name']):
                                    st.success("Deleted")
                                    st.rerun()
                        
                        st.divider()
            else:
                st.info("No saved projects yet")
            
            if st.button("Close", use_container_width=True):
                st.session_state.show_load_dialog = False
                st.rerun()


# Export functions that should be called from main app
__all__ = [
    'initialize_session_persistence',
    'save_project',
    'load_project',
    'delete_project',
    'list_projects',
    'auto_save',
    'render_project_manager_ui'
]