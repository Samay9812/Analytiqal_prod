"""
Analytics Assistant Configuration - UPDATED VERSION
Now includes data-aware intents for automatic issue detection
"""

from typing import Dict, List, Tuple

# ============================================================================
# INTENT KEYWORDS - UPDATED with data-aware queries
# ============================================================================

INTENT_KEYWORDS = {
    # DATA QUALITY QUERIES (NEW!)
    'analyze_data': [
        'analyze', 'check data', 'what should i do', 'what to do',
        'help me', 'guide me', 'where to start', 'data quality',
        'check quality', 'inspect data', 'review data'
    ],
    
    'clean_data': [
        'clean', 'clean data', 'clean my data', 'fix data',
        'prepare data', 'fix issues', 'improve quality'
    ],
    
    'check_issues': [
        'issues', 'problems', 'what\'s wrong', 'find issues',
        'detect issues', 'data problems', 'check for problems'
    ],
    
    # DATA CLEANING (existing)
    'remove_nulls': [
        'null', 'missing', 'nan', 'empty', 'blank', 'remove missing',
        'fill missing', 'handle missing', 'missing values', 'missing data'
    ],
    
    'remove_duplicates': [
        'duplicate', 'duplicates', 'repeated', 'duplicate rows',
        'remove duplicates', 'drop duplicates', 'unique rows'
    ],
    
    'remove_outliers': [
        'outlier', 'outliers', 'extreme values', 'anomaly', 'anomalies',
        'remove outliers', 'handle outliers', 'outlier detection'
    ],
    
    'remove_constant': [
        'constant column', 'constant columns', 'same value', 'no variation',
        'remove constant', 'drop constant'
    ],
    
    'standardize_text': [
        'lowercase', 'uppercase', 'title case', 'standardize',
        'clean text', 'normalize text', 'trim whitespace'
    ],
    
    # TRANSFORMATIONS
    'filter_rows': [
        'filter', 'where', 'rows where', 'select rows', 'subset',
        'filter by', 'keep rows', 'remove rows'
    ],
    
    'select_columns': [
        'select column', 'keep column', 'choose column', 'pick column',
        'drop column', 'remove column', 'columns only'
    ],
    
    'sort_data': [
        'sort', 'order', 'sort by', 'order by', 'arrange',
        'ascending', 'descending'
    ],
    
    'rename_column': [
        'rename', 'rename column', 'change name', 'column name'
    ],
    
    'create_column': [
        'create column', 'new column', 'add column', 'calculated column',
        'compute column', 'derive column'
    ],
    
    'pivot': [
        'pivot', 'pivot table', 'wide format', 'reshape wide'
    ],
    
    'melt': [
        'melt', 'unpivot', 'long format', 'reshape long'
    ],
    
    # FEATURE ENGINEERING
    'datetime_features': [
        'extract', 'datetime', 'date', 'year', 'month', 'day',
        'hour', 'minute', 'day of week', 'extract date', 'parse date'
    ],
    
    'math_features': [
        'multiply', 'divide', 'add', 'subtract', 'power',
        'ratio', 'product', 'sum', 'difference', 'polynomial'
    ],
    
    'aggregation': [
        'group by', 'aggregate', 'mean by', 'sum by', 'count by',
        'average by', 'groupby', 'group'
    ],
    
    'interaction': [
        'interaction', 'combine', 'cross', 'multiply columns'
    ],
    
    'transformation': [
        'log', 'sqrt', 'normalize', 'standardize', 'scale',
        'transform', 'log transform', 'square root'
    ],
    
    # VISUALIZATIONS
    'chart': [
        'chart', 'plot', 'graph', 'visualize', 'visualization',
        'show', 'display'
    ],
    
    'histogram': [
        'histogram', 'distribution', 'frequency'
    ],
    
    'scatter': [
        'scatter', 'scatter plot', 'relationship', 'correlation plot'
    ],
    
    'bar_chart': [
        'bar chart', 'bar plot', 'bar graph', 'bars'
    ],
    
    'line_chart': [
        'line chart', 'line plot', 'time series', 'trend'
    ],
    
    'box_plot': [
        'box plot', 'boxplot', 'box and whisker'
    ],
    
    # EXPORT
    'export': [
        'export', 'download', 'save', 'save as', 'export to'
    ]
}


# ============================================================================
# WORKFLOW TEMPLATES - UPDATED
# ============================================================================

WORKFLOW_TEMPLATES = {
    'auto_clean': [
        {'page': '🔍 Data Profiling', 'action': 'Review data quality'},
        {'page': '🧹 Data Cleaning', 'action': 'Handle missing values'},
        {'page': '🧹 Data Cleaning', 'action': 'Remove duplicates'},
        {'page': '🧹 Data Cleaning', 'action': 'Treat outliers'}
    ],
    
    'clean_and_visualize': [
        {'page': '🧹 Data Cleaning', 'action': 'Remove missing values'},
        {'page': '🧹 Data Cleaning', 'action': 'Remove duplicates'},
        {'page': '📊 Visualizations', 'action': 'Create visualization'}
    ],
    
    'prepare_for_analysis': [
        {'page': '🧹 Data Cleaning', 'action': 'Handle missing values'},
        {'page': '🧹 Data Cleaning', 'action': 'Remove outliers'},
        {'page': '🔄 Transformations', 'action': 'Filter and select columns'},
        {'page': '📈 Feature Engineering', 'action': 'Create features'}
    ],
    
    'time_series_analysis': [
        {'page': '🔄 Transformations', 'action': 'Convert to datetime'},
        {'page': '📈 Feature Engineering', 'action': 'Extract datetime features'},
        {'page': '📊 Visualizations', 'action': 'Time series plot'}
    ],
    
    'feature_creation': [
        {'page': '📈 Feature Engineering', 'action': 'Mathematical features'},
        {'page': '📈 Feature Engineering', 'action': 'Aggregation features'},
        {'page': '🔍 Data Profiling', 'action': 'Review new features'}
    ]
}


# ============================================================================
# PAGE CAPABILITIES
# ============================================================================

PAGE_CAPABILITIES = {
    '🏠 Dataset': {
        'description': 'Upload and preview data',
        'actions': [
            'Upload CSV/Excel file',
            'View dataset preview',
            'Check data types',
            'See basic statistics'
        ]
    },
    
    '🧹 Data Cleaning': {
        'description': 'Clean and improve data quality',
        'actions': [
            'Handle missing values (fill, drop)',
            'Remove duplicate records',
            'Detect and treat outliers',
            'Remove constant columns',
            'Standardize text columns',
            'Remove high correlation features'
        ]
    },
    
    '🔄 Transformations': {
        'description': 'Transform and reshape data',
        'actions': [
            'Filter rows by conditions',
            'Select specific columns',
            'Sort and reorder data',
            'Sample data',
            'Rename columns',
            'Create calculated columns',
            'Pivot/Melt data'
        ]
    },
    
    '🔍 Data Profiling': {
        'description': 'Analyze and understand your data',
        'actions': [
            'View statistical summary',
            'Analyze individual columns',
            'Explore distributions',
            'Check correlations',
            'Get automated insights'
        ]
    },
    
    '📈 Feature Engineering': {
        'description': 'Create new features',
        'actions': [
            'Extract datetime features',
            'Create mathematical features',
            'Generate aggregation features',
            'Build interaction features',
            'Apply transformations (log, normalize, etc.)'
        ]
    },
    
    '📊 Visualizations': {
        'description': 'Create charts and visualizations',
        'actions': [
            'Univariate charts (histogram, box plot, etc.)',
            'Bivariate charts (scatter, line, etc.)',
            'Multivariate analysis (heatmap, 3D)',
            'Time series plots',
            'Export charts'
        ]
    },
    
    '💾 Export & Reports': {
        'description': 'Export data and reports',
        'actions': [
            'Download cleaned data',
            'Export to CSV/Excel',
            'Generate profiling report',
            'Save visualization'
        ]
    }
}


# ============================================================================
# EXAMPLE QUERIES - UPDATED with data-aware examples
# ============================================================================

EXAMPLE_QUERIES = [
    # Data-aware queries (NEW!)
    "What should I do to clean my data?",
    "Check my data quality and suggest improvements",
    "What issues does my data have?",
    "Help me prepare this dataset for analysis",
    
    # Specific queries
    "Remove missing values and duplicates",
    "Create a profit column from revenue and cost",
    "Filter data where age > 25 and create a bar chart",
    "Extract year and month from date column",
    "Remove outliers from salary and show distribution",
    "Group by category and calculate average price",
    "Create interaction between price and quantity",
    "Normalize all numeric columns",
    "Show correlation between numeric variables",
    "Convert to long format and visualize"
]


# ============================================================================
# COLUMN TYPE REQUIREMENTS
# ============================================================================

COLUMN_TYPE_REQUIREMENTS = {
    'remove_outliers': ['numeric'],
    'histogram': ['numeric'],
    'scatter': ['numeric'],
    'normalize': ['numeric'],
    'log_transform': ['numeric'],
    'group_by': ['categorical'],
    'bar_chart': ['categorical'],
    'datetime_features': ['datetime'],
    'time_series': ['datetime']
}


# ============================================================================
# ERROR MESSAGES
# ============================================================================

ERROR_MESSAGES = {
    'no_dataset': "⚠️ **No dataset loaded**\n\nPlease upload a dataset from the **🏠 Dataset** page first.",
    
    'column_not_found': "⚠️ **Column '{column}' not found**\n\nAvailable columns: {available}",
    
    'wrong_type': "⚠️ **Column '{column}' is {current_type} but needs to be {required_type}**\n\nConsider converting it first in the Transformations page.",
    
    'no_numeric': "⚠️ **No numeric columns found**\n\nThis operation requires numeric data.",
    
    'no_categorical': "⚠️ **No categorical columns found**\n\nThis operation requires categorical data.",
    
    'no_datetime': "⚠️ **No datetime columns found**\n\nPlease convert a column to datetime first in the Transformations page.",
    
    'unclear_intent': "🤔 **I'm not sure what you want to do**\n\nTry being more specific or use one of the example queries below.",
    
    'too_complex': "⚠️ **This request is too complex**\n\nTry breaking it down into smaller steps or use multiple queries."
}


# ============================================================================
# OPERATION PRIORITY - UPDATED
# ============================================================================

OPERATION_PRIORITY = {
    # Data quality checks first
    'analyze_data': 0,
    'check_issues': 0,
    
    # Then cleaning
    'remove_nulls': 1,
    'remove_duplicates': 2,
    'remove_outliers': 3,
    'remove_constant': 4,
    'clean_data': 5,
    
    # Then transformations
    'filter_rows': 6,
    'select_columns': 7,
    'create_column': 8,
    
    # Then feature engineering
    'datetime_features': 9,
    'math_features': 10,
    'aggregation': 11,
    'transformation': 12,
    
    # Finally visualization
    'chart': 13,
    'export': 14
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_page_for_operation(operation: str) -> str:
    """Map operation to appropriate page."""
    
    page_mapping = {
        # Data quality
        'analyze_data': '🔍 Data Profiling',
        'check_issues': '🔍 Data Profiling',
        'clean_data': '🧹 Data Cleaning',
        
        # Cleaning
        'remove_nulls': '🧹 Data Cleaning',
        'remove_duplicates': '🧹 Data Cleaning',
        'remove_outliers': '🧹 Data Cleaning',
        'remove_constant': '🧹 Data Cleaning',
        'standardize_text': '🧹 Data Cleaning',
        
        # Transformations
        'filter_rows': '🔄 Transformations',
        'select_columns': '🔄 Transformations',
        'sort_data': '🔄 Transformations',
        'rename_column': '🔄 Transformations',
        'create_column': '🔄 Transformations',
        'pivot': '🔄 Transformations',
        'melt': '🔄 Transformations',
        
        # Feature engineering
        'datetime_features': '📈 Feature Engineering',
        'math_features': '📈 Feature Engineering',
        'aggregation': '📈 Feature Engineering',
        'interaction': '📈 Feature Engineering',
        'transformation': '📈 Feature Engineering',
        
        # Visualization
        'chart': '📊 Visualizations',
        'histogram': '📊 Visualizations',
        'scatter': '📊 Visualizations',
        'bar_chart': '📊 Visualizations',
        'line_chart': '📊 Visualizations',
        'box_plot': '📊 Visualizations',
        
        # Export
        'export': '💾 Export & Reports'
    }
    
    return page_mapping.get(operation, '🏠 Dataset')


def get_friendly_action_name(operation: str) -> str:
    """Convert operation code to friendly name."""
    
    friendly_names = {
        # Data quality
        'analyze_data': 'Analyze Data Quality',
        'check_issues': 'Check for Issues',
        'clean_data': 'Clean Data',
        
        # Cleaning
        'remove_nulls': 'Handle Missing Values',
        'remove_duplicates': 'Remove Duplicate Records',
        'remove_outliers': 'Detect and Treat Outliers',
        'remove_constant': 'Remove Constant Columns',
        'standardize_text': 'Standardize Text',
        
        # Transformations
        'filter_rows': 'Filter Rows',
        'select_columns': 'Select Columns',
        'sort_data': 'Sort Data',
        'rename_column': 'Rename Column',
        'create_column': 'Create Calculated Column',
        'pivot': 'Pivot Table',
        'melt': 'Melt (Unpivot)',
        
        # Feature engineering
        'datetime_features': 'Extract DateTime Features',
        'math_features': 'Create Mathematical Features',
        'aggregation': 'Group and Aggregate',
        'interaction': 'Create Interaction Features',
        'transformation': 'Apply Transformation',
        
        # Visualization
        'chart': 'Create Visualization',
        'histogram': 'Create Histogram',
        'scatter': 'Create Scatter Plot',
        'bar_chart': 'Create Bar Chart',
        'line_chart': 'Create Line Plot',
        'box_plot': 'Create Box Plot',
        
        # Export
        'export': 'Export Data'
    }
    
    return friendly_names.get(operation, operation.replace('_', ' ').title())