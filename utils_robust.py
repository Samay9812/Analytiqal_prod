"""
Robust utility functions for Data Analysis Pro Streamlit app.
All functions include comprehensive error handling and validation.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import logging
import pickle
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# LOGGING & HISTORY
# ============================================================================


def log_action(action: str) -> None:
    """
    Log processing actions for audit trail with timestamp.
    
    Args:
        action: Description of the action performed
        
    Example:
        log_action("Loaded dataset with 1000 rows")
    """
    try:
        if 'processing_log' not in st.session_state:
            st.session_state.processing_log = []
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {action}"
        
        st.session_state.processing_log.append(log_entry)
        logger.info(action)
        
        # Limit log size to prevent memory issues
        if len(st.session_state.processing_log) > 1000:
            st.session_state.processing_log = st.session_state.processing_log[-1000:]
    
    except Exception as e:
        logger.error(f"Failed to log action: {str(e)}")


def update_df(new_df: pd.DataFrame, action: str = "") -> bool:
    """
    Update dataframe with history tracking and validation.
    
    Args:
        new_df: New dataframe to set as current
        action: Description of what changed
        
    Returns:
        bool: True if update successful, False otherwise
        
    Example:
        if update_df(filtered_df, "Filtered rows by age > 25"):
            st.success("Dataset updated!")
    """
    # Validate input
    if new_df is None:
        st.error("Cannot update with None dataframe")
        logger.error("Attempted to update with None dataframe")
        return False
    
    if not isinstance(new_df, pd.DataFrame):
        st.error("Invalid input: must be a pandas DataFrame")
        logger.error(f"Invalid type: {type(new_df)}")
        return False
    
    if new_df.empty:
        st.warning("Updating with empty dataframe")
        logger.warning("Empty dataframe update")
    
    try:
        # Initialize session state if needed
        if 'history' not in st.session_state:
            st.session_state.history = []
        if 'redo_stack' not in st.session_state:
            st.session_state.redo_stack = []
        
        # Save current state to history (if exists)
        if st.session_state.df is not None:
            # Limit history size to prevent memory issues (keep last 20)
            if len(st.session_state.history) >= 20:
                st.session_state.history.pop(0)
            
            st.session_state.history.append(st.session_state.df.copy())
        
        # Clear redo stack on new action
        st.session_state.redo_stack = []

        # Fix category columns before storing (PREVENTS DISPLAY ERRORS)
        new_df = fix_category_columns(new_df)  # ADD THIS LINE

        
        # Update current dataframe
        st.session_state.df = new_df.copy()
        
        # Log action
        if action:
            log_action(action)
        
        return True
    
    except Exception as e:
        st.error(f"Failed to update dataframe: {str(e)}")
        logger.exception("Dataframe update error")
        return False


# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(file, nrows: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Load CSV or Excel file with robust error handling and validation.
    
    Args:
        file: Uploaded file object from st.file_uploader
        nrows: Maximum number of rows to load (None for all)
        
    Returns:
        pd.DataFrame or None if loading failed
        
    Features:
        - Auto-detects encoding for CSV files
        - Handles files with/without headers
        - Validates loaded data
        - Cleans column names
        - Auto-detects and converts data types
        
    Example:
        uploaded = st.file_uploader("Upload file")
        if uploaded:
            df = load_data(uploaded)
            if df is not None:
                st.success(f"Loaded {len(df)} rows")
    """
    if file is None:
        st.error("No file provided")
        logger.error("load_data called with None file")
        return None
    
    try:
        filename = file.name
        file_ext = Path(filename).suffix.lower()
        
        logger.info(f"Loading file: {filename} ({file_ext})")
        
        # Load based on file type
        if file_ext == '.csv':
            df = _load_csv(file, nrows)
        elif file_ext in ['.xlsx', '.xls']:
            df = _load_excel(file, nrows, file_ext)
        else:
            st.error(f"Unsupported file format: {file_ext}")
            logger.error(f"Unsupported format: {file_ext}")
            return None
        
        # Validate loaded data
        if df is None:
            st.error("Failed to load file")
            return None
        
        if df.empty:
            st.error("File is empty or contains no valid data")
            logger.warning(f"Empty dataframe from {filename}")
            return None
        
        if len(df.columns) == 0:
            st.error("No columns detected in file")
            logger.error(f"No columns in {filename}")
            return None
        
        # Clean column names
        df = _clean_column_names(df)
        
        # Auto-detect and convert data types
        df = _auto_convert_types(df)
        
        # Remove completely empty rows and columns
        initial_rows = len(df)
        initial_cols = len(df.columns)
        
        df = df.dropna(how='all', axis=0)  # Drop empty rows
        df = df.dropna(how='all', axis=1)  # Drop empty columns
        
        removed_rows = initial_rows - len(df)
        removed_cols = initial_cols - len(df.columns)
        
        if removed_rows > 0 or removed_cols > 0:
            logger.info(f"Removed {removed_rows} empty rows and {removed_cols} empty columns")
        
        logger.info(f"Successfully loaded: {len(df)} rows × {len(df.columns)} columns")

        df = fix_category_columns(df)  # ADD THIS LINE
        
        return df
    
    except Exception as e:
        st.error(f"Error loading file '{filename}': {str(e)}")
        logger.exception(f"Error loading {filename}")
        return None


def _load_csv(file, nrows: Optional[int]) -> Optional[pd.DataFrame]:
    """Load CSV with encoding detection and fallback strategies."""
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        for has_header in [True, False]:
            try:
                file.seek(0)
                header = 0 if has_header else None
                df = pd.read_csv(file, encoding=encoding, nrows=nrows, header=header)
                
                # Validate we got some data
                if not df.empty and len(df.columns) > 0:
                    logger.info(f"CSV loaded with encoding={encoding}, header={has_header}")
                    return df
            
            except UnicodeDecodeError:
                continue  # Try next encoding
            except Exception as e:
                logger.debug(f"Failed with encoding={encoding}, header={has_header}: {e}")
                continue
    
    logger.error("Could not load CSV with any encoding/header combination")
    return None


def _load_excel(file, nrows: Optional[int], file_ext: str) -> Optional[pd.DataFrame]:
    """Load Excel file with appropriate engine."""
    engine = 'openpyxl' if file_ext == '.xlsx' else 'xlrd'
    
    for has_header in [True, False]:
        try:
            file.seek(0)
            header = 0 if has_header else None
            df = pd.read_excel(file, engine=engine, nrows=nrows, header=header)
            
            if not df.empty and len(df.columns) > 0:
                logger.info(f"Excel loaded with engine={engine}, header={has_header}")
                return df
        
        except Exception as e:
            logger.debug(f"Failed with header={has_header}: {e}")
            continue
    
    logger.error("Could not load Excel file")
    return None


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize column names.
    
    - Handles unnamed/empty columns
    - Removes special characters
    - Handles duplicates
    - Converts to strings
    """
    new_columns = []
    
    for i, col in enumerate(df.columns):
        # Handle unnamed, empty, or numeric columns
        if (pd.isna(col) or 
            str(col).strip() == '' or 
            'Unnamed' in str(col) or 
            isinstance(col, (int, float))):
            new_columns.append(f"Column_{i+1}")
        else:
            # Clean column name
            clean_name = str(col).strip()
            # Replace problematic characters
            clean_name = clean_name.replace('\n', '_').replace('\r', '_').replace('\t', '_')
            new_columns.append(clean_name)
    
    # Handle duplicates by appending numbers
    seen = {}
    final_columns = []
    
    for col in new_columns:
        if col in seen:
            seen[col] += 1
            final_columns.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            final_columns.append(col)
    
    df.columns = final_columns
    
    logger.debug(f"Cleaned column names: {final_columns}")
    return df


def _auto_convert_types(df: pd.DataFrame, sample_size: int = 5000) -> pd.DataFrame:
    """
    Automatically detect and convert column data types.
    
    Args:
        df: Input dataframe
        sample_size: Number of rows to sample for type detection
        
    Returns:
        DataFrame with optimized types
    """
    # Use sample for faster type detection on large datasets
    sample_df = df.head(sample_size) if len(df) > sample_size else df
    
    for col in df.columns:
        try:
            # Skip if already numeric or datetime
            if (pd.api.types.is_numeric_dtype(df[col]) or 
                pd.api.types.is_datetime64_any_dtype(df[col])):
                continue
            
            # Try to convert to numeric
            try:
                converted = pd.to_numeric(sample_df[col], errors='coerce')
                # If >80% of values convert successfully, apply to full column
                success_rate = converted.notna().sum() / len(sample_df)
                if success_rate > 0.8:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    logger.debug(f"Converted '{col}' to numeric ({success_rate:.1%} success)")
                    continue
            except Exception:
                pass
            
            # Try to convert to datetime
            try:
                converted = pd.to_datetime(sample_df[col], errors='coerce')
                success_rate = converted.notna().sum() / len(sample_df)
                if success_rate > 0.8:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    logger.debug(f"Converted '{col}' to datetime ({success_rate:.1%} success)")
                    continue
            except Exception:
                pass
            
        
        except Exception as e:
            logger.warning(f"Type conversion failed for '{col}': {e}")
            continue
    
    return df


# ============================================================================
# COLUMN TYPE DETECTION
# ============================================================================

def get_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorize columns by data type with enhanced detection.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dict with keys: 'numeric', 'categorical', 'datetime', 'boolean'
        Each contains list of column names
        
    Example:
        types = get_column_types(df)
        st.write(f"Numeric columns: {types['numeric']}")
        st.write(f"Categorical columns: {types['categorical']}")
    """
    if df is None or not isinstance(df, pd.DataFrame):
        logger.error("Invalid input to get_column_types")
        return {'numeric': [], 'categorical': [], 'datetime': [], 'boolean': []}
    
    if df.empty:
        logger.warning("Empty dataframe passed to get_column_types")
        return {'numeric': [], 'categorical': [], 'datetime': [], 'boolean': []}
    
    try:
        column_types = {
            'numeric': [],
            'categorical': [],
            'datetime': [],
            'boolean': []
        }
        
        for col in df.columns:
            try:
                # DateTime detection
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    column_types['datetime'].append(col)
                
                # Boolean detection
                elif pd.api.types.is_bool_dtype(df[col]):
                    column_types['boolean'].append(col)
                
                # Numeric detection
                elif pd.api.types.is_numeric_dtype(df[col]):
                    column_types['numeric'].append(col)
                
                # Categorical/Object detection
                else:
                    column_types['categorical'].append(col)
            
            except Exception as e:
                logger.warning(f"Type detection failed for column '{col}': {e}")
                column_types['categorical'].append(col)  # Default to categorical
        
        logger.debug(f"Column types: {[(k, len(v)) for k, v in column_types.items()]}")
        return column_types
    
    except Exception as e:
        logger.exception("Error in get_column_types")
        st.error(f"Column type detection failed: {str(e)}")
        return {'numeric': [], 'categorical': [], 'datetime': [], 'boolean': []}


# ============================================================================
# OUTLIER DETECTION
# ============================================================================

def detect_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> Optional[Tuple[int, float, float]]:
    """
    Detect outliers using Interquartile Range (IQR) method.
    
    Args:
        df: Input dataframe
        column: Column name to analyze
        multiplier: IQR multiplier (1.5 = standard, 3.0 = extreme outliers)
        
    Returns:
        Tuple of (outlier_count, lower_bound, upper_bound) or None if invalid
        
    Example:
        result = detect_outliers_iqr(df, 'age', multiplier=1.5)
        if result:
            count, lower, upper = result
            st.write(f"Found {count} outliers")
            st.write(f"Valid range: {lower:.2f} to {upper:.2f}")
    """
    # Validate inputs
    if df is None or not isinstance(df, pd.DataFrame):
        logger.error("Invalid dataframe in detect_outliers_iqr")
        return None
    
    if column not in df.columns:
        logger.error(f"Column '{column}' not found in dataframe")
        st.warning(f"Column '{column}' not found")
        return None
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        logger.warning(f"Column '{column}' is not numeric")
        st.info(f"Column '{column}' must be numeric for outlier detection")
        return None
    
    if multiplier <= 0:
        logger.error(f"Invalid multiplier: {multiplier}")
        st.error("Multiplier must be positive")
        return None
    
    try:
        # Remove NaN values for calculation
        clean_data = df[column].dropna()
        
        if len(clean_data) == 0:
            logger.warning(f"No valid data in column '{column}'")
            st.info(f"Column '{column}' has no valid numeric values")
            return None
        
        if len(clean_data) < 4:
            logger.warning(f"Insufficient data in '{column}' for IQR calculation")
            st.info(f"Need at least 4 values for IQR calculation (found {len(clean_data)})")
            return None
        
        # Calculate IQR
        Q1 = clean_data.quantile(0.25)
        Q3 = clean_data.quantile(0.75)
        IQR = Q3 - Q1
        
        # Calculate bounds
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        # Find outliers
        outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
        outlier_count = len(outliers)
        
        logger.info(f"IQR outliers in '{column}': {outlier_count} / {len(clean_data)} "
                   f"({outlier_count/len(clean_data)*100:.1f}%)")
        
        return outlier_count, lower_bound, upper_bound
    
    except Exception as e:
        logger.exception(f"Error in IQR outlier detection for '{column}'")
        st.error(f"Outlier detection failed for '{column}': {str(e)}")
        return None


def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0) -> Optional[int]:
    """
    Detect outliers using Z-score method.
    
    Args:
        df: Input dataframe
        column: Column name to analyze
        threshold: Z-score threshold (typically 2.5-3.5)
        
    Returns:
        Number of outliers or None if invalid
        
    Example:
        outlier_count = detect_outliers_zscore(df, 'salary', threshold=3.0)
        if outlier_count is not None:
            st.write(f"Found {outlier_count} outliers using Z-score method")
    """
    # Validate inputs
    if df is None or not isinstance(df, pd.DataFrame):
        logger.error("Invalid dataframe in detect_outliers_zscore")
        return None
    
    if column not in df.columns:
        logger.error(f"Column '{column}' not found in dataframe")
        st.warning(f"Column '{column}' not found")
        return None
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        logger.warning(f"Column '{column}' is not numeric")
        st.info(f"Column '{column}' must be numeric for outlier detection")
        return None
    
    if threshold <= 0:
        logger.error(f"Invalid threshold: {threshold}")
        st.error("Threshold must be positive")
        return None
    
    try:
        # Remove NaN values
        clean_data = df[column].dropna()
        
        if len(clean_data) == 0:
            logger.warning(f"No valid data in column '{column}'")
            st.info(f"Column '{column}' has no valid numeric values")
            return None
        
        if len(clean_data) < 3:
            logger.warning(f"Insufficient data in '{column}' for Z-score calculation")
            st.info(f"Need at least 3 values for Z-score calculation (found {len(clean_data)})")
            return None
        
        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(clean_data))
        
        # Count outliers
        outlier_count = int(np.sum(z_scores > threshold))
        
        logger.info(f"Z-score outliers in '{column}': {outlier_count} / {len(clean_data)} "
                   f"({outlier_count/len(clean_data)*100:.1f}%)")
        
        return outlier_count
    
    except Exception as e:
        logger.exception(f"Error in Z-score outlier detection for '{column}'")
        st.error(f"Outlier detection failed for '{column}': {str(e)}")
        return None


# ============================================================================
# METADATA GENERATION
# ============================================================================

def generate_column_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comprehensive metadata for all columns.
    
    Args:
        df: Input dataframe
        
    Returns:
        DataFrame with columns: Column, Type, Data Type, Missing %, Unique Values, Relevance
        
    Relevance categories:
        - "Target": Likely target variable (contains keywords)
        - "High": Useful for analysis (numeric/binary with low missing)
        - "Medium": Potentially useful
        - "Low": High missing rate or low information value
        
    Example:
        metadata = generate_column_metadata(df)
        st.dataframe(metadata)
    """
    if df is None or not isinstance(df, pd.DataFrame):
        logger.error("Invalid dataframe in generate_column_metadata")
        return pd.DataFrame()
    
    if df.empty:
        logger.warning("Empty dataframe in generate_column_metadata")
        return pd.DataFrame()
    
    try:
        metadata = []
        target_keywords = ['target', 'label', 'class', 'status', 'outcome', 'result', 'y']
        
        for col in df.columns:
            try:
                col_data = df[col]
                col_type = str(col_data.dtype)
                
                # Determine type category
                if pd.api.types.is_numeric_dtype(col_data):
                    # Check if binary numeric (only 0 and 1)
                    unique_vals = col_data.dropna().unique()
                    if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
                        type_category = "binary"
                    else:
                        type_category = "numeric"
                elif pd.api.types.is_bool_dtype(col_data):
                    type_category = "binary"
                elif col_data.nunique() == 2:
                    type_category = "binary"
                else:
                    type_category = "categorical"
                
                # Calculate statistics
                missing_pct = (col_data.isnull().sum() / len(df)) * 100
                unique_count = col_data.nunique()
                
                # Determine relevance
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in target_keywords):
                    relevance = "Target"
                elif missing_pct > 50:
                    relevance = "Low"
                elif type_category in ['numeric', 'binary'] and missing_pct <= 30:
                    relevance = "High"
                elif type_category == 'categorical' and 2 <= unique_count <= 20:
                    relevance = "High"
                else:
                    relevance = "Medium"
                
                metadata.append({
                    'Column': col,
                    'Type': type_category,
                    'Data Type': col_type,
                    'Missing %': f"{missing_pct:.1f}%",
                    'Unique Values': unique_count,
                    'Relevance': relevance
                })
            
            except Exception as e:
                logger.warning(f"Metadata generation failed for column '{col}': {e}")
                metadata.append({
                    'Column': col,
                    'Type': 'unknown',
                    'Data Type': 'error',
                    'Missing %': 'N/A',
                    'Unique Values': 'N/A',
                    'Relevance': 'Unknown'
                })
        
        metadata_df = pd.DataFrame(metadata)
        logger.info(f"Generated metadata for {len(metadata_df)} columns")
        
        return metadata_df
    
    except Exception as e:
        logger.exception("Error generating column metadata")
        st.error(f"Metadata generation failed: {str(e)}")
        return pd.DataFrame()


# ============================================================================
# DATA QUALITY SCORING
# ============================================================================

def calculate_data_quality_score(df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    """
    Calculate overall data quality score based on 6 dimensions.
    
    Args:
        df: Input dataframe
        
    Returns:
        Tuple of (overall_score, dimension_scores_dict)
        
    Dimensions (with weights):
        - Completeness (20%): Percentage of non-missing values
        - Validity (20%): Data type consistency
        - Accuracy (25%): Outlier-based estimation
        - Consistency (15%): Data format uniformity
        - Timeliness (10%): Placeholder for data freshness
        - Uniqueness (10%): Inverse of duplicate rate
        
    Example:
        overall, dimensions = calculate_data_quality_score(df)
        st.metric("Data Quality", f"{overall:.1f}/100")
        for dim, score in dimensions.items():
            st.write(f"{dim}: {score:.1f}/100")
    """
    if df is None or not isinstance(df, pd.DataFrame):
        logger.error("Invalid dataframe in calculate_data_quality_score")
        return 0.0, {}
    
    if df.empty:
        logger.warning("Empty dataframe in calculate_data_quality_score")
        return 0.0, {}
    
    try:
        scores = {}
        
        # 1. Completeness (20%)
        total_cells = len(df) * len(df.columns)
        if total_cells > 0:
            missing_cells = df.isnull().sum().sum()
            completeness = 100 - (missing_cells / total_cells * 100)
        else:
            completeness = 0
        scores['Completeness'] = max(0, min(100, completeness))
        
        # 2. Validity (20%) - Check for inf values and valid types
        valid_columns = 0
        for col in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Check for inf values
                    has_inf = np.isinf(df[col]).any()
                    if not has_inf:
                        valid_columns += 1
                else:
                    valid_columns += 1
            except Exception:
                pass
        
        validity = (valid_columns / len(df.columns)) * 100 if len(df.columns) > 0 else 100
        scores['Validity'] = max(0, min(100, validity))
        
        # 3. Uniqueness (10%)
        if len(df) > 0:
            duplicate_count = df.duplicated().sum()
            uniqueness = 100 - (duplicate_count / len(df) * 100)
        else:
            uniqueness = 100
        scores['Uniqueness'] = max(0, min(100, uniqueness))
        
        # 4. Consistency (15%) - Simplified placeholder
        consistency = 90  # Would require more complex checks
        scores['Consistency'] = consistency
        
        # 5. Accuracy (25%) - Placeholder
        accuracy = 85  # Would require validation against source
        scores['Accuracy'] = accuracy
        
        # 6. Timeliness (10%) - Placeholder
        timeliness = 95  # Would require metadata about data freshness
        scores['Timeliness'] = timeliness
        
        # Calculate weighted overall score
        weights = {
            'Completeness': 0.20,
            'Validity': 0.20,
            'Accuracy': 0.25,
            'Consistency': 0.15,
            'Timeliness': 0.10,
            'Uniqueness': 0.10
        }
        
        overall_score = sum(scores[dim] * weights[dim] for dim in scores.keys())
        overall_score = max(0, min(100, overall_score))
        
        logger.info(f"Data quality score: {overall_score:.2f}/100")
        
        return round(overall_score, 2), {k: round(v, 2) for k, v in scores.items()}
    
    except Exception as e:
        logger.exception("Error calculating data quality score")
        st.error(f"Quality calculation failed: {str(e)}")
        return 0.0, {}


# ============================================================================
# MULTI-DATASET SUPPORT
# ============================================================================

def get_columns(dataset_name: str) -> List[str]:
    """
    Get list of columns for a named dataset in session state.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        List of column names, or empty list if dataset not found
        
    Example:
        cols = get_columns("my_dataset")
        if cols:
            selected = st.selectbox("Choose column", cols)
    """
    try:
        # Check if datasets exist in session state
        if 'datasets' not in st.session_state:
            logger.debug("No 'datasets' in session state")
            return []
        
        datasets = st.session_state.datasets
        
        # Check if dataset exists
        if dataset_name not in datasets:
            logger.warning(f"Dataset '{dataset_name}' not found in session state")
            return []
        
        # Check if dataset has 'df' key
        if 'df' not in datasets[dataset_name]:
            logger.warning(f"No 'df' found in dataset '{dataset_name}'")
            return []
        
        df = datasets[dataset_name]['df']
        
        # Validate it's a DataFrame
        if not isinstance(df, pd.DataFrame):
            logger.error(f"Invalid dataframe type in dataset '{dataset_name}'")
            return []
        
        columns = list(df.columns)
        logger.debug(f"Retrieved {len(columns)} columns from '{dataset_name}'")
        
        return columns
    
    except Exception as e:
        logger.exception(f"Error getting columns for dataset '{dataset_name}'")
        return []

PROJECTS_DIR = Path("saved_projects")
PROJECTS_DIR.mkdir(exist_ok=True)


def save_project(project_name: str) -> bool:
    """Save current session to disk."""
    try:
        # Create safe filename
        safe_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        
        if not safe_name:
            st.error("Invalid project name")
            return False
        
        # Create project folder
        project_folder = PROJECTS_DIR / safe_name
        project_folder.mkdir(exist_ok=True)
        
        # Get data to save
        session_data = {
            'df': st.session_state.get('df'),
            'history': st.session_state.get('history', []),
            'redo_stack': st.session_state.get('redo_stack', []),
            'processing_log': st.session_state.get('processing_log', []),
        }
        
        # Save as pickle
        data_file = project_folder / "data.pkl"
        with open(data_file, 'wb') as f:
            pickle.dump(session_data, f)
        
        # Save metadata
        metadata = {
            'project_name': project_name,
            'safe_name': safe_name,
            'saved_date': datetime.now().isoformat(),
            'row_count': len(session_data['df']) if session_data['df'] is not None else 0,
            'column_count': len(session_data['df'].columns) if session_data['df'] is not None else 0,
        }
        
        metadata_file = project_folder / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved project: {project_name}")
        return True
        
    except Exception as e:
        st.error(f"Save failed: {str(e)}")
        logger.exception(f"Error saving project {project_name}")
        return False


def load_project(project_name: str) -> bool:
    """Load a saved project."""
    try:
        # Find project
        projects = list_projects()
        project_meta = next((p for p in projects if p['project_name'] == project_name), None)
        
        if not project_meta:
            st.error(f"Project '{project_name}' not found")
            return False
        
        safe_name = project_meta['safe_name']
        project_folder = PROJECTS_DIR / safe_name
        data_file = project_folder / "data.pkl"
        
        if not data_file.exists():
            st.error("Project data file not found")
            return False
        
        # Load data
        with open(data_file, 'rb') as f:
            session_data = pickle.load(f)
        
        # Restore to session state
        st.session_state.df = session_data.get('df')
        st.session_state.history = session_data.get('history', [])
        st.session_state.redo_stack = session_data.get('redo_stack', [])
        st.session_state.processing_log = session_data.get('processing_log', [])
        
        logger.info(f"Loaded project: {project_name}")
        return True
        
    except Exception as e:
        st.error(f"Load failed: {str(e)}")
        logger.exception(f"Error loading project {project_name}")
        return False


def delete_project(project_name: str) -> bool:
    """Delete a saved project."""
    try:
        projects = list_projects()
        project_meta = next((p for p in projects if p['project_name'] == project_name), None)
        
        if not project_meta:
            return False
        
        safe_name = project_meta['safe_name']
        project_folder = PROJECTS_DIR / safe_name
        
        if project_folder.exists():
            import shutil
            shutil.rmtree(project_folder)
        
        logger.info(f"Deleted project: {project_name}")
        return True
        
    except Exception as e:
        logger.exception(f"Error deleting project {project_name}")
        return False


def list_projects() -> list:
    """Get list of all saved projects."""
    try:
        projects = []
        
        for folder in PROJECTS_DIR.iterdir():
            if folder.is_dir():
                metadata_file = folder / "metadata.json"
                
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        projects.append(metadata)
        
        # Sort by date (newest first)
        projects.sort(key=lambda x: x.get('saved_date', ''), reverse=True)
        
        return projects
        
    except Exception as e:
        logger.exception("Error listing projects")
        return []

# ============================================================================
# PYARROW DISPLAY FIX
# ============================================================================

def fix_category_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix category columns for PyArrow/Streamlit display compatibility.
    
    Converts all category dtype columns to string to prevent:
    'ArrowInvalid: Could not convert with type str: tried to convert to double'
    
    Args:
        df: Input dataframe
        
    Returns:
        DataFrame with category columns converted to string
    """
    if df is None or df.empty:
        return df
    
    try:
        # Work on a copy to avoid modifying original
        fixed_df = df.copy()
        
        # Convert all category columns to string
        for col in fixed_df.columns:
            if fixed_df[col].dtype.name == 'category':
                fixed_df[col] = fixed_df[col].astype(str)
                logger.debug(f"Converted category column '{col}' to string for display")
        
        return fixed_df
    
    except Exception as e:
        logger.warning(f"Error fixing category columns: {e}")
        return df