"""
Analytics Assistant Engine - UPDATED VERSION
Now includes data-aware analysis using profiler
"""

import pandas as pd
import re
from typing import Dict, List, Tuple, Optional, Any
from assistant_config import (
    INTENT_KEYWORDS,
    WORKFLOW_TEMPLATES,
    PAGE_CAPABILITIES,
    COLUMN_TYPE_REQUIREMENTS,
    ERROR_MESSAGES,
    OPERATION_PRIORITY,
    get_page_for_operation,
    get_friendly_action_name
)

# Import the profiler (NEW!)
try:
    from assistant_profiler import get_dataset_statistics, DataProfiler
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False
    print("Warning: assistant_profiler not found. Data-aware features disabled.")


# ============================================================================
# INTENT PARSER
# ============================================================================

class IntentParser:
    """Parse user text to extract intents and entities."""
    
    def __init__(self):
        self.intent_keywords = INTENT_KEYWORDS
    
    def parse(self, user_text: str) -> Dict[str, Any]:
        """
        Parse user input to extract intents and mentioned columns.
        
        Args:
            user_text: Raw user input
            
        Returns:
            {
                'intents': List of detected operations,
                'columns': List of mentioned column names,
                'confidence': Confidence score (0-1),
                'is_data_aware': Whether query asks for automated analysis
            }
        """
        
        if not user_text or not user_text.strip():
            return {
                'intents': [],
                'columns': [],
                'confidence': 0.0,
                'is_data_aware': False
            }
        
        text = user_text.lower().strip()
        detected_intents = []
        
        # Check if this is a data-aware query (NEW!)
        data_aware_triggers = ['what should', 'what to do', 'analyze', 'check data',
                              'find issues', 'help me', 'guide me', 'clean data',
                              'data quality', 'what\'s wrong', 'problems']
        
        is_data_aware = any(trigger in text for trigger in data_aware_triggers)
        
        # Detect intents based on keywords
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    if intent not in detected_intents:
                        detected_intents.append(intent)
                    break
        
        # Extract potential column names (words in quotes or backticks)
        columns = []
        
        # Pattern 1: Text in quotes
        quoted = re.findall(r'["\']([^"\']+)["\']', user_text)
        columns.extend(quoted)
        
        # Pattern 2: Text in backticks
        backticked = re.findall(r'`([^`]+)`', user_text)
        columns.extend(backticked)
        
        # Pattern 3: Common column indicators
        column_indicators = [
            r'column\s+(\w+)',
            r'from\s+(\w+)',
            r'(\w+)\s+column',
            r'in\s+(\w+)',
            r'by\s+(\w+)'
        ]
        
        for pattern in column_indicators:
            matches = re.findall(pattern, text)
            columns.extend(matches)
        
        # Remove duplicates and common words
        common_words = {'the', 'and', 'or', 'from', 'to', 'in', 'by', 'with', 'column', 'columns'}
        columns = list(set([col for col in columns if col.lower() not in common_words]))
        
        # Calculate confidence
        confidence = min(1.0, len(detected_intents) * 0.3 + (0.2 if columns else 0))
        
        return {
            'intents': detected_intents,
            'columns': columns,
            'confidence': confidence,
            'is_data_aware': is_data_aware
        }


# ============================================================================
# COLUMN VALIDATOR
# ============================================================================

class ColumnValidator:
    """Validate columns against dataset schema."""
    
    def __init__(self, df: pd.DataFrame, col_types: Dict[str, List[str]]):
        self.df = df
        self.col_types = col_types
        self.all_columns = list(df.columns) if df is not None else []
    
    def validate_column(self, column_name: str) -> Tuple[bool, str]:
        """
        Check if column exists and return its type.
        
        Returns:
            (is_valid, column_type)
        """
        
        if self.df is None:
            return False, "no_dataset"
        
        # Case-insensitive search
        matching_cols = [col for col in self.all_columns if col.lower() == column_name.lower()]
        
        if not matching_cols:
            # Fuzzy match
            similar = [col for col in self.all_columns if column_name.lower() in col.lower()]
            if similar:
                return False, f"not_found_suggestion:{similar[0]}"
            return False, "not_found"
        
        actual_col = matching_cols[0]
        
        # Determine type
        if actual_col in self.col_types.get('numeric', []):
            return True, 'numeric'
        elif actual_col in self.col_types.get('categorical', []):
            return True, 'categorical'
        elif actual_col in self.col_types.get('datetime', []):
            return True, 'datetime'
        else:
            return True, 'unknown'
    
    def check_type_requirement(self, column_name: str, required_type: str) -> bool:
        """Check if column meets type requirement."""
        
        is_valid, col_type = self.validate_column(column_name)
        
        if not is_valid:
            return False
        
        if required_type == 'numeric':
            return col_type == 'numeric'
        elif required_type == 'categorical':
            return col_type == 'categorical'
        elif required_type == 'datetime':
            return col_type == 'datetime'
        
        return True
    
    def get_available_columns_by_type(self, column_type: str) -> List[str]:
        """Get list of columns of specific type."""
        return self.col_types.get(column_type, [])


# ============================================================================
# WORKFLOW GENERATOR
# ============================================================================

class WorkflowGenerator:
    """Generate step-by-step workflows from intents."""
    
    def __init__(self):
        self.operation_priority = OPERATION_PRIORITY
    
    def generate(self, intents: List[str], columns: List[str], col_validator: ColumnValidator) -> List[Dict]:
        """
        Generate ordered workflow from detected intents.
        
        Returns:
            List of steps
        """
        
        if not intents:
            return []
        
        workflow = []
        
        # Sort intents by priority
        sorted_intents = sorted(
            intents,
            key=lambda x: self.operation_priority.get(x, 999)
        )
        
        for intent in sorted_intents:
            step = {
                'page': get_page_for_operation(intent),
                'action': get_friendly_action_name(intent),
                'operation': intent,
                'priority': self.operation_priority.get(intent, 999)
            }
            
            # Add column-specific details if applicable
            details = self._get_step_details(intent, columns, col_validator)
            if details:
                step['details'] = details
            
            # Add guidance
            guidance = self._get_step_guidance(intent, columns, col_validator)
            if guidance:
                step['guidance'] = guidance
            
            workflow.append(step)
        
        # Add step numbers
        for i, step in enumerate(workflow, 1):
            step['step'] = i
        
        return workflow
    
    def _get_step_details(self, intent: str, columns: List[str], validator: ColumnValidator) -> str:
        """Generate specific details for a step."""
        
        details = []
        
        # Column-specific operations
        if intent in ['remove_outliers', 'normalize', 'log_transform'] and columns:
            valid_cols = []
            for col in columns:
                is_valid, col_type = validator.validate_column(col)
                if is_valid and col_type == 'numeric':
                    valid_cols.append(col)
            
            if valid_cols:
                details.append(f"Column(s): {', '.join(valid_cols)}")
        
        elif intent in ['bar_chart', 'group_by'] and columns:
            valid_cols = []
            for col in columns:
                is_valid, col_type = validator.validate_column(col)
                if is_valid and col_type == 'categorical':
                    valid_cols.append(col)
            
            if valid_cols:
                details.append(f"Column(s): {', '.join(valid_cols)}")
        
        elif intent == 'datetime_features' and columns:
            valid_cols = []
            for col in columns:
                is_valid, col_type = validator.validate_column(col)
                if is_valid:
                    valid_cols.append(col)
            
            if valid_cols:
                details.append(f"Date column: {valid_cols[0]}")
        
        elif intent == 'create_column' and columns:
            details.append(f"Suggested columns: {', '.join(columns)}")
        
        return ' | '.join(details) if details else None
    
    def _get_step_guidance(self, intent: str, columns: List[str], validator: ColumnValidator) -> str:
        """Generate helpful guidance for a step."""
        
        guidance_map = {
            'remove_nulls': 'Choose treatment method based on column type and % missing',
            'remove_outliers': 'Recommended: IQR method for most cases',
            'create_column': 'Use eval() for mathematical expressions',
            'datetime_features': 'Common: Year, Month, Day, Day of Week, Is Weekend',
            'aggregation': 'Group by categorical, aggregate numeric columns',
            'transformation': 'Log transform for right-skewed data',
            'chart': 'Choose chart type based on variable types',
            'pivot': 'Wide format good for reporting',
            'melt': 'Long format good for visualization',
            'analyze_data': 'Review statistics to identify issues',
            'clean_data': 'Start with missing values and duplicates'
        }
        
        return guidance_map.get(intent)


# ============================================================================
# ASSISTANT ENGINE (Main Class) - UPDATED
# ============================================================================

class AnalyticsAssistant:
    """Main assistant engine that coordinates all components."""
    
    def __init__(self, df: Optional[pd.DataFrame] = None, col_types: Optional[Dict] = None):
        self.df = df
        self.col_types = col_types or {'numeric': [], 'categorical': [], 'datetime': []}
        
        self.parser = IntentParser()
        self.validator = ColumnValidator(df, self.col_types) if df is not None else None
        self.generator = WorkflowGenerator()
        
        # NEW: Initialize profiler if available
        self.stats = None
        self.profiler = None
        if PROFILER_AVAILABLE and df is not None:
            self.stats = get_dataset_statistics(df, self.col_types)
            self.profiler = DataProfiler(self.stats)
    
    def update_dataset(self, df: pd.DataFrame, col_types: Dict):
        """Update dataset and column types."""
        self.df = df
        self.col_types = col_types
        self.validator = ColumnValidator(df, col_types)
        
        # NEW: Update profiler
        if PROFILER_AVAILABLE:
            self.stats = get_dataset_statistics(df, col_types)
            self.profiler = DataProfiler(self.stats)
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Main method to process user query and generate workflow.
        UPDATED with data-aware analysis
        
        Args:
            user_query: User's natural language request
            
        Returns:
            {
                'success': bool,
                'workflow': List of steps,
                'warnings': List of warnings,
                'suggestions': List of suggestions,
                'recommendations': List of data-aware recommendations (NEW!),
                'data_summary': Quality summary (NEW!),
                'error': Optional error message
            }
        """
        
        # Check if dataset exists
        if self.df is None:
            return {
                'success': False,
                'workflow': [],
                'warnings': [],
                'suggestions': [],
                'recommendations': [],
                'error': ERROR_MESSAGES['no_dataset']
            }
        
        # Parse intent
        parsed = self.parser.parse(user_query)
        
        # NEW: Handle data-aware queries
        if parsed['is_data_aware'] and PROFILER_AVAILABLE and self.profiler:
            return self._process_data_aware_query(user_query, parsed)
        
        # Regular query processing
        if not parsed['intents']:
            return {
                'success': False,
                'workflow': [],
                'warnings': [],
                'suggestions': self._get_query_suggestions(),
                'recommendations': [],
                'error': ERROR_MESSAGES['unclear_intent']
            }
        
        # Validate columns
        warnings = []
        validated_columns = []
        
        for col in parsed['columns']:
            is_valid, col_info = self.validator.validate_column(col)
            
            if is_valid:
                validated_columns.append(col)
            else:
                if 'suggestion' in col_info:
                    suggested = col_info.split(':')[1]
                    warnings.append(f"Column '{col}' not found. Did you mean '{suggested}'?")
                else:
                    available = ', '.join(self.validator.all_columns[:5])
                    warnings.append(
                        ERROR_MESSAGES['column_not_found'].format(
                            column=col,
                            available=available + ('...' if len(self.validator.all_columns) > 5 else '')
                        )
                    )
        
        # Generate workflow
        workflow = self.generator.generate(
            parsed['intents'],
            validated_columns,
            self.validator
        )
        
        # Add suggestions
        suggestions = self._get_contextual_suggestions(parsed['intents'], validated_columns)
        
        return {
            'success': True,
            'workflow': workflow,
            'warnings': warnings,
            'suggestions': suggestions,
            'recommendations': [],
            'confidence': parsed['confidence']
        }
    
    def _process_data_aware_query(self, user_query: str, parsed: Dict) -> Dict[str, Any]:
        """
        NEW: Process data-aware queries that analyze the dataset.
        
        Examples:
        - "What should I do?"
        - "Check my data quality"
        - "Find issues in my data"
        """
        
        # Generate recommendations from profiler
        summary = self.profiler.get_summary()
        recommendations = summary['recommendations']
        
        # Priority mapping for recommendations
        priority_map = {
            'critical': 0,
            'high': 1,
            'medium': 2,
            'low': 3
        }
        
        # Convert recommendations to workflow steps
        workflow = []
        for i, rec in enumerate(recommendations[:10], 1):  # Top 10 recommendations
            workflow.append({
                'step': i,
                'page': rec['page'],
                'action': rec['tab'] if 'tab' in rec else rec['action'],
                'operation': rec['issue'],
                'details': rec['details'],
                'guidance': rec['action'],
                'priority': priority_map.get(rec['priority'], 999),  # ← FIXED!
                'severity': rec.get('icon', '🟡')
            })
        
        # Generate suggestions
        suggestions = []
        if summary['quality_score'] < 70:
            suggestions.append("💡 Focus on critical and high priority issues first")
        if summary['by_priority']['critical'] > 0:
            suggestions.append("⚠️ Address critical issues before proceeding with analysis")
        if summary['by_priority']['low'] > 5:
            suggestions.append("🟢 Low priority items are optional improvements")
        
        return {
            'success': True,
            'workflow': workflow,
            'warnings': [],
            'suggestions': suggestions,
            'recommendations': recommendations,
            'data_summary': summary,
            'is_data_aware': True
        }    
    def _get_query_suggestions(self) -> List[str]:
        """Get example queries to help user."""
        from assistant_config import EXAMPLE_QUERIES
        return EXAMPLE_QUERIES[:5]
    
    def _get_contextual_suggestions(self, intents: List[str], columns: List[str]) -> List[str]:
        """Generate contextual suggestions based on intents."""
        
        suggestions = []
        
        # Suggest related operations
        if 'remove_nulls' in intents and 'remove_duplicates' not in intents:
            suggestions.append("💡 Also consider removing duplicates")
        
        if 'create_column' in intents and 'chart' not in intents:
            suggestions.append("💡 You might want to visualize the new column")
        
        if 'datetime_features' in intents and 'chart' not in intents:
            suggestions.append("💡 Consider creating a time series visualization")
        
        if 'remove_outliers' in intents and 'chart' not in intents:
            suggestions.append("💡 Visualize the distribution after outlier removal")
        
        # Suggest checking results
        if any(intent in intents for intent in ['remove_nulls', 'remove_outliers', 'transformation']):
            suggestions.append("💡 Check Data Profiling page to verify results")
        
        return suggestions
    
    def get_page_help(self, page_name: str) -> Dict[str, Any]:
        """Get help information for a specific page."""
        return PAGE_CAPABILITIES.get(page_name, {})
    
    def suggest_workflow(self, goal: str) -> List[Dict]:
        """Suggest a workflow template based on high-level goal."""
        
        goal = goal.lower()
        
        if 'clean' in goal and 'visualize' in goal:
            return WORKFLOW_TEMPLATES['clean_and_visualize']
        
        elif 'prepare' in goal or 'analysis' in goal:
            return WORKFLOW_TEMPLATES['prepare_for_analysis']
        
        elif 'time series' in goal or 'date' in goal:
            return WORKFLOW_TEMPLATES['time_series_analysis']
        
        elif 'feature' in goal or 'engineer' in goal:
            return WORKFLOW_TEMPLATES['feature_creation']
        
        return []


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_workflow_for_display(workflow: List[Dict]) -> str:
    """Format workflow as readable text."""
    
    if not workflow:
        return "No steps generated."
    
    output = []
    output.append("📋 **Recommended Workflow:**\n")
    
    for step in workflow:
        step_num = step['step']
        page = step['page']
        action = step['action']
        
        output.append(f"{step_num}️⃣ **Go to {page}**")
        output.append(f"   → {action}")
        
        if 'details' in step:
            output.append(f"   ℹ️ {step['details']}")
        
        if 'guidance' in step:
            output.append(f"   💡 {step['guidance']}")
        
        output.append("")
    
    return "\n".join(output)


def get_available_operations(df: pd.DataFrame, col_types: Dict) -> Dict[str, List[str]]:
    """Get list of available operations based on dataset."""
    
    operations = {
        'cleaning': ['Remove Missing Values', 'Remove Duplicates'],
        'transformation': ['Filter Rows', 'Select Columns', 'Sort Data'],
        'feature_engineering': [],
        'visualization': []
    }
    
    # Add operations based on column types
    if col_types['numeric']:
        operations['cleaning'].append('Remove Outliers')
        operations['transformation'].append('Create Calculated Column')
        operations['feature_engineering'].extend(['Mathematical Features', 'Transformations'])
        operations['visualization'].extend(['Histogram', 'Box Plot', 'Scatter Plot'])
    
    if col_types['categorical']:
        operations['transformation'].append('Group and Aggregate')
        operations['feature_engineering'].append('Interaction Features')
        operations['visualization'].extend(['Bar Chart', 'Pie Chart'])
    
    if col_types['datetime']:
        operations['feature_engineering'].append('Extract DateTime Features')
        operations['visualization'].append('Time Series Plot')
    
    return operations