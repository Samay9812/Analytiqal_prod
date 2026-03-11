"""
Analytics Assistant - Data Profiler
Analyzes dataset statistics and generates smart recommendations
WITHOUT accessing raw data - only uses metadata and summary stats
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


# ============================================================================
# STATISTICS EXTRACTOR
# ============================================================================

def get_dataset_statistics(df: pd.DataFrame, col_types: Dict) -> Dict[str, Any]:
    """
    Extract comprehensive statistics from dataset.
    This is what the assistant sees - NOT the raw data!
    
    Args:
        df: The dataframe
        col_types: Dict from get_column_types()
        
    Returns:
        Comprehensive statistics dictionary
    """
    
    stats = {
        'overview': {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / (1024**2),
            'numeric_count': len(col_types.get('numeric', [])),
            'categorical_count': len(col_types.get('categorical', [])),
            'datetime_count': len(col_types.get('datetime', []))
        },
        
        'quality': {
            'missing_total': int(df.isnull().sum().sum()),
            'missing_pct': float((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100),
            'duplicates': int(df.duplicated().sum()),
            'duplicate_pct': float((df.duplicated().sum() / len(df)) * 100) if len(df) > 0 else 0,
            'constant_columns': [col for col in df.columns if df[col].nunique() <= 1]
        },
        
        'columns': {}
    }
    
    # Per-column statistics
    for col in df.columns:
        col_stats = {
            'type': 'datetime' if col in col_types.get('datetime', []) 
                    else 'numeric' if col in col_types.get('numeric', []) 
                    else 'categorical',
            'missing': int(df[col].isnull().sum()),
            'missing_pct': float((df[col].isnull().sum() / len(df)) * 100) if len(df) > 0 else 0,
            'unique': int(df[col].nunique()),
            'unique_pct': float((df[col].nunique() / len(df)) * 100) if len(df) > 0 else 0
        }
        
        # Numeric column stats
        if col in col_types.get('numeric', []):
            try:
                col_stats.update({
                    'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else 0,
                    'median': float(df[col].median()) if not pd.isna(df[col].median()) else 0,
                    'std': float(df[col].std()) if not pd.isna(df[col].std()) else 0,
                    'min': float(df[col].min()) if not pd.isna(df[col].min()) else 0,
                    'max': float(df[col].max()) if not pd.isna(df[col].max()) else 0,
                    'skewness': float(df[col].skew()) if not pd.isna(df[col].skew()) else 0,
                    'kurtosis': float(df[col].kurtosis()) if not pd.isna(df[col].kurtosis()) else 0,
                    'zeros': int((df[col] == 0).sum()),
                    'zeros_pct': float(((df[col] == 0).sum() / len(df)) * 100) if len(df) > 0 else 0
                })
                
                # Detect outliers using IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
                
                col_stats['outliers_iqr'] = int(len(outliers))
                col_stats['outliers_pct'] = float((len(outliers) / len(df)) * 100) if len(df) > 0 else 0
                col_stats['iqr_bounds'] = {
                    'lower': float(Q1 - 1.5*IQR),
                    'upper': float(Q3 + 1.5*IQR)
                }
            except:
                pass
        
        # Categorical column stats
        elif col in col_types.get('categorical', []):
            try:
                mode_val = df[col].mode()
                col_stats['mode'] = str(mode_val[0]) if len(mode_val) > 0 else 'N/A'
                col_stats['mode_frequency'] = int(df[col].value_counts().iloc[0]) if len(df[col].value_counts()) > 0 else 0
                col_stats['mode_pct'] = float((col_stats['mode_frequency'] / len(df)) * 100) if len(df) > 0 else 0
                col_stats['cardinality'] = 'high' if col_stats['unique_pct'] > 90 else 'medium' if col_stats['unique_pct'] > 50 else 'low'
            except:
                pass
        
        stats['columns'][col] = col_stats
    
    return stats


# ============================================================================
# SMART RECOMMENDATION ENGINE
# ============================================================================

class DataProfiler:
    """Analyzes dataset statistics and generates smart recommendations."""
    
    def __init__(self, stats: Dict):
        self.stats = stats
    
    def generate_recommendations(self) -> List[Dict]:
        """
        Generate prioritized recommendations based on dataset statistics.
        
        Returns:
            List of recommendations sorted by priority
        """
        
        recommendations = []
        
        # 1. Check for missing values
        recommendations.extend(self._check_missing_values())
        
        # 2. Check for duplicates
        recommendations.extend(self._check_duplicates())
        
        # 3. Check for outliers
        recommendations.extend(self._check_outliers())
        
        # 4. Check for skewed distributions
        recommendations.extend(self._check_skewness())
        
        # 5. Check for constant columns
        recommendations.extend(self._check_constant_columns())
        
        # 6. Check for high cardinality
        recommendations.extend(self._check_cardinality())
        
        # 7. Check for zeros
        recommendations.extend(self._check_zeros())
        
        # Sort by priority
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 999))
        
        return recommendations
    
    def _check_missing_values(self) -> List[Dict]:
        """Check for missing values in columns."""
        
        recommendations = []
        
        for col, col_stats in self.stats['columns'].items():
            missing_pct = col_stats.get('missing_pct', 0)
            
            if missing_pct > 50:
                recommendations.append({
                    'priority': 'critical',
                    'category': 'Data Quality',
                    'issue': 'Very High Missing Values',
                    'column': col,
                    'details': f"{col_stats['missing']} missing ({missing_pct:.1f}%)",
                    'severity': f'{missing_pct:.1f}%',
                    'impact': 'May need to drop this column',
                    'action': 'Consider dropping column or investigate why so much data is missing',
                    'page': '🧹 Data Cleaning',
                    'tab': 'Missing Values',
                    'icon': '🔴'
                })
            
            elif missing_pct > 20:
                recommendations.append({
                    'priority': 'high',
                    'category': 'Data Quality',
                    'issue': 'High Missing Values',
                    'column': col,
                    'details': f"{col_stats['missing']} missing ({missing_pct:.1f}%)",
                    'severity': f'{missing_pct:.1f}%',
                    'impact': 'May affect analysis quality',
                    'action': 'Use advanced imputation (grouped mean/median) or drop rows',
                    'page': '🧹 Data Cleaning',
                    'tab': 'Missing Values',
                    'icon': '🟠'
                })
            
            elif missing_pct > 5:
                col_type = col_stats.get('type', 'unknown')
                method = 'median' if col_type == 'numeric' else 'mode'
                
                recommendations.append({
                    'priority': 'medium',
                    'category': 'Data Quality',
                    'issue': 'Missing Values Detected',
                    'column': col,
                    'details': f"{col_stats['missing']} missing ({missing_pct:.1f}%)",
                    'severity': f'{missing_pct:.1f}%',
                    'impact': 'Minor impact on analysis',
                    'action': f'Fill with {method} or drop rows',
                    'page': '🧹 Data Cleaning',
                    'tab': 'Missing Values',
                    'icon': '🟡'
                })
        
        return recommendations
    
    def _check_duplicates(self) -> List[Dict]:
        """Check for duplicate rows."""
        
        recommendations = []
        
        dup_count = self.stats['quality'].get('duplicates', 0)
        dup_pct = self.stats['quality'].get('duplicate_pct', 0)
        
        if dup_count > 0:
            if dup_pct > 10:
                priority = 'high'
                icon = '🔴'
            elif dup_pct > 5:
                priority = 'medium'
                icon = '🟠'
            else:
                priority = 'low'
                icon = '🟢'
            
            recommendations.append({
                'priority': priority,
                'category': 'Data Quality',
                'issue': 'Duplicate Rows Found',
                'column': None,
                'details': f"{dup_count} duplicate rows ({dup_pct:.1f}%)",
                'severity': f'{dup_pct:.1f}%',
                'impact': 'Inflates counts and skews analysis',
                'action': 'Remove duplicates (keep first occurrence recommended)',
                'page': '🧹 Data Cleaning',
                'tab': 'Duplicates',
                'icon': icon
            })
        
        return recommendations
    
    def _check_outliers(self) -> List[Dict]:
        """Check for outliers in numeric columns."""
        
        recommendations = []
        
        for col, col_stats in self.stats['columns'].items():
            if col_stats.get('type') == 'numeric':
                outliers = col_stats.get('outliers_iqr', 0)
                outliers_pct = col_stats.get('outliers_pct', 0)
                
                if outliers > 0 and outliers_pct > 1:
                    if outliers_pct > 15:
                        priority = 'high'
                        icon = '🔴'
                        impact = 'Severely affects statistics and models'
                    elif outliers_pct > 5:
                        priority = 'medium'
                        icon = '🟠'
                        impact = 'May distort analysis'
                    else:
                        priority = 'low'
                        icon = '🟡'
                        impact = 'Minor impact'
                    
                    bounds = col_stats.get('iqr_bounds', {})
                    
                    recommendations.append({
                        'priority': priority,
                        'category': 'Data Quality',
                        'issue': 'Outliers Detected',
                        'column': col,
                        'details': f"{outliers} outliers ({outliers_pct:.1f}%) detected using IQR method",
                        'severity': f'{outliers_pct:.1f}%',
                        'impact': impact,
                        'action': f'Use IQR method (bounds: [{bounds.get("lower", 0):.1f}, {bounds.get("upper", 0):.1f}])',
                        'page': '🧹 Data Cleaning',
                        'tab': 'Outliers',
                        'icon': icon
                    })
        
        return recommendations
    
    def _check_skewness(self) -> List[Dict]:
        """Check for highly skewed distributions."""
        
        recommendations = []
        
        for col, col_stats in self.stats['columns'].items():
            if col_stats.get('type') == 'numeric':
                skewness = abs(col_stats.get('skewness', 0))
                
                if skewness > 2:
                    recommendations.append({
                        'priority': 'low',
                        'category': 'Feature Engineering',
                        'issue': 'Highly Skewed Distribution',
                        'column': col,
                        'details': f"Skewness = {col_stats.get('skewness', 0):.2f}",
                        'severity': f'{skewness:.1f}',
                        'impact': 'May affect model performance',
                        'action': 'Consider log transform or Box-Cox transformation',
                        'page': '📈 Feature Engineering',
                        'tab': 'Transformations',
                        'icon': '🟢'
                    })
                
                elif skewness > 1:
                    recommendations.append({
                        'priority': 'low',
                        'category': 'Feature Engineering',
                        'issue': 'Moderately Skewed Distribution',
                        'column': col,
                        'details': f"Skewness = {col_stats.get('skewness', 0):.2f}",
                        'severity': f'{skewness:.1f}',
                        'impact': 'May need transformation for some models',
                        'action': 'Consider sqrt or log transformation',
                        'page': '📈 Feature Engineering',
                        'tab': 'Transformations',
                        'icon': '🟢'
                    })
        
        return recommendations
    
    def _check_constant_columns(self) -> List[Dict]:
        """Check for constant columns (all values same)."""
        
        recommendations = []
        
        constant_cols = self.stats['quality'].get('constant_columns', [])
        
        if constant_cols:
            recommendations.append({
                'priority': 'medium',
                'category': 'Data Quality',
                'issue': 'Constant Columns',
                'column': ', '.join(constant_cols),
                'details': f"{len(constant_cols)} columns with no variation",
                'severity': f'{len(constant_cols)} columns',
                'impact': 'Provides no information for analysis',
                'action': 'Remove these columns (no variation = no value)',
                'page': '🧹 Data Cleaning',
                'tab': 'General Cleaning',
                'icon': '🟡'
            })
        
        return recommendations
    
    def _check_cardinality(self) -> List[Dict]:
        """Check for high cardinality categorical columns."""
        
        recommendations = []
        
        for col, col_stats in self.stats['columns'].items():
            if col_stats.get('type') == 'categorical':
                cardinality = col_stats.get('cardinality', 'low')
                unique_pct = col_stats.get('unique_pct', 0)
                
                if cardinality == 'high':
                    recommendations.append({
                        'priority': 'low',
                        'category': 'Feature Engineering',
                        'issue': 'High Cardinality Column',
                        'column': col,
                        'details': f"{col_stats['unique']} unique values ({unique_pct:.1f}%)",
                        'severity': f'{unique_pct:.1f}% unique',
                        'impact': 'May be ID column or require special encoding',
                        'action': 'Consider dropping if ID, or use target encoding',
                        'page': '🔄 Transformations',
                        'tab': 'Select Columns',
                        'icon': '🟢'
                    })
        
        return recommendations
    
    def _check_zeros(self) -> List[Dict]:
        """Check for high percentage of zeros."""
        
        recommendations = []
        
        for col, col_stats in self.stats['columns'].items():
            if col_stats.get('type') == 'numeric':
                zeros_pct = col_stats.get('zeros_pct', 0)
                
                if zeros_pct > 50:
                    recommendations.append({
                        'priority': 'low',
                        'category': 'Data Quality',
                        'issue': 'High Percentage of Zeros',
                        'column': col,
                        'details': f"{col_stats.get('zeros', 0)} zeros ({zeros_pct:.1f}%)",
                        'severity': f'{zeros_pct:.1f}%',
                        'impact': 'May indicate sparse data or measurement issue',
                        'action': 'Investigate if zeros are valid or missing values',
                        'page': '🔍 Data Profiling',
                        'tab': 'Column Analysis',
                        'icon': '🟢'
                    })
        
        return recommendations
    
    def get_summary(self) -> Dict:
        """Get overall data quality summary."""
        
        recommendations = self.generate_recommendations()
        
        # Count by priority
        priority_counts = {
            'critical': sum(1 for r in recommendations if r['priority'] == 'critical'),
            'high': sum(1 for r in recommendations if r['priority'] == 'high'),
            'medium': sum(1 for r in recommendations if r['priority'] == 'medium'),
            'low': sum(1 for r in recommendations if r['priority'] == 'low')
        }
        
        # Calculate overall quality score (0-100)
        total_issues = len(recommendations)
        critical_weight = priority_counts['critical'] * 25
        high_weight = priority_counts['high'] * 15
        medium_weight = priority_counts['medium'] * 10
        low_weight = priority_counts['low'] * 5
        
        total_penalty = critical_weight + high_weight + medium_weight + low_weight
        quality_score = max(0, 100 - total_penalty)
        
        return {
            'total_issues': total_issues,
            'by_priority': priority_counts,
            'quality_score': quality_score,
            'quality_grade': 'A' if quality_score >= 90 else 'B' if quality_score >= 80 else 'C' if quality_score >= 70 else 'D' if quality_score >= 60 else 'F',
            'recommendations': recommendations
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_recommendations_for_display(recommendations: List[Dict]) -> str:
    """Format recommendations as readable text."""
    
    if not recommendations:
        return "✅ **No issues found! Your data looks clean.**"
    
    output = []
    
    # Group by priority
    priorities = {
        'critical': [],
        'high': [],
        'medium': [],
        'low': []
    }
    
    for rec in recommendations:
        priorities[rec['priority']].append(rec)
    
    # Display by priority
    if priorities['critical']:
        output.append("## 🔴 CRITICAL PRIORITY\n")
        for rec in priorities['critical']:
            output.append(f"**{rec['issue']}**")
            if rec['column']:
                output.append(f"Column: `{rec['column']}`")
            output.append(f"└─ {rec['details']}")
            output.append(f"└─ Action: {rec['action']}")
            output.append(f"└─ Go to: {rec['page']} → {rec['tab']}")
            output.append("")
    
    if priorities['high']:
        output.append("## 🟠 HIGH PRIORITY\n")
        for rec in priorities['high']:
            output.append(f"**{rec['issue']}**")
            if rec['column']:
                output.append(f"Column: `{rec['column']}`")
            output.append(f"└─ {rec['details']}")
            output.append(f"└─ Action: {rec['action']}")
            output.append(f"└─ Go to: {rec['page']} → {rec['tab']}")
            output.append("")
    
    if priorities['medium']:
        output.append("## 🟡 MEDIUM PRIORITY\n")
        for rec in priorities['medium']:
            output.append(f"**{rec['issue']}**")
            if rec['column']:
                output.append(f"Column: `{rec['column']}`")
            output.append(f"└─ {rec['details']}")
            output.append(f"└─ Action: {rec['action']}")
            output.append("")
    
    if priorities['low']:
        output.append("## 🟢 LOW PRIORITY (Optional)\n")
        for rec in priorities['low']:
            output.append(f"**{rec['issue']}**")
            if rec['column']:
                output.append(f"Column: `{rec['column']}`")
            output.append(f"└─ {rec['details']}")
            output.append("")
    
    return "\n".join(output)