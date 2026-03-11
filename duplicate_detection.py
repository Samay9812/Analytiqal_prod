"""
Multi-Field Duplicate Detection Module
Intelligent entity resolution with confidence scoring
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from itertools import combinations
from difflib import SequenceMatcher
import re


class DuplicateDetector:
    """
    Smart duplicate detection using multiple fields and confidence scoring
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = []
    
    def detect_duplicates(
        self,
        primary_field: str,
        supporting_fields: List[str],
        fuzzy_threshold: float = 0.85,
        confidence_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Detect duplicates using multi-field matching
        
        Args:
            primary_field: Main field to check (usually name or email)
            supporting_fields: Additional fields to verify match
            fuzzy_threshold: Similarity threshold for primary field (0-1)
            confidence_threshold: Minimum confidence to report (0-1)
        
        Returns:
            List of potential duplicate pairs with confidence scores
        """
        self.results = []
        
        # Get unique values from primary field for initial grouping
        primary_groups = self._group_by_fuzzy_match(primary_field, fuzzy_threshold)
        
        # For each group, check supporting fields
        for group in primary_groups:
            if len(group) < 2:
                continue
            
            # Check all pairs within group
            for idx1, idx2 in combinations(group, 2):
                record1 = self.df.iloc[idx1]
                record2 = self.df.iloc[idx2]
                
                # Calculate confidence score
                match_info = self._calculate_match_confidence(
                    record1, record2, primary_field, supporting_fields
                )
                
                if match_info['confidence'] >= confidence_threshold:
                    match_info['index_1'] = idx1
                    match_info['index_2'] = idx2
                    match_info['record_1'] = record1.to_dict()
                    match_info['record_2'] = record2.to_dict()
                    self.results.append(match_info)
        
        # Sort by confidence (highest first)
        self.results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return self.results
    
    def _group_by_fuzzy_match(self, field: str, threshold: float) -> List[List[int]]:
        """
        Group records by fuzzy matching on a single field
        Returns list of groups (each group is list of row indices)
        """
        groups = []
        processed = set()
        
        values = self.df[field].fillna('').astype(str)
        
        for idx1, val1 in enumerate(values):
            if idx1 in processed or not val1.strip():
                continue
            
            group = [idx1]
            processed.add(idx1)
            
            # Find similar values
            for idx2, val2 in enumerate(values):
                if idx2 <= idx1 or idx2 in processed or not val2.strip():
                    continue
                
                similarity = self._string_similarity(val1, val2)
                if similarity >= threshold:
                    group.append(idx2)
                    processed.add(idx2)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def _calculate_match_confidence(
        self,
        record1: pd.Series,
        record2: pd.Series,
        primary_field: str,
        supporting_fields: List[str]
    ) -> Dict:
        """
        Calculate confidence that two records are the same entity
        """
        matches = {}
        weights = {}
        
        # Primary field (30% weight)
        primary_sim = self._string_similarity(
            str(record1[primary_field]),
            str(record2[primary_field])
        )
        matches[primary_field] = primary_sim
        weights[primary_field] = 0.3
        
        # Supporting fields (70% weight total, divided equally)
        field_weight = 0.7 / len(supporting_fields) if supporting_fields else 0
        
        for field in supporting_fields:
            if field not in record1.index or field not in record2.index:
                continue
            
            val1 = record1[field]
            val2 = record2[field]
            
            # Skip if both are missing
            if pd.isna(val1) and pd.isna(val2):
                matches[field] = None
                weights[field] = 0
                continue
            
            # Penalize if only one is missing
            if pd.isna(val1) or pd.isna(val2):
                matches[field] = 0.0
                weights[field] = field_weight * 0.5  # Reduced weight for missing data
                continue
            
            # Check field type and compare accordingly
            if self._is_numeric_field(val1, val2):
                match_score = 1.0 if self._numeric_match(val1, val2) else 0.0
            elif self._is_email_field(field):
                match_score = self._email_similarity(str(val1), str(val2))
            elif self._is_phone_field(field):
                match_score = self._phone_similarity(str(val1), str(val2))
            else:
                match_score = self._string_similarity(str(val1), str(val2))
            
            matches[field] = match_score
            weights[field] = field_weight
        
        # Calculate weighted confidence score
        total_weight = sum(weights.values())
        if total_weight == 0:
            confidence = primary_sim
        else:
            confidence = sum(
                matches[field] * weights[field] 
                for field in matches 
                if matches[field] is not None
            ) / total_weight
        
        return {
            'confidence': confidence,
            'matches': matches,
            'weights': weights
        }
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using SequenceMatcher"""
        str1 = str(str1).lower().strip()
        str2 = str(str2).lower().strip()
        
        if not str1 or not str2:
            return 0.0
        
        return SequenceMatcher(None, str1, str2).ratio()
    
    def _is_numeric_field(self, val1, val2) -> bool:
        """Check if both values are numeric"""
        try:
            float(val1)
            float(val2)
            return True
        except:
            return False
    
    def _numeric_match(self, val1, val2, tolerance: float = 0.01) -> bool:
        """Check if numeric values match within tolerance"""
        try:
            num1 = float(val1)
            num2 = float(val2)
            return abs(num1 - num2) <= tolerance * max(abs(num1), abs(num2))
        except:
            return False
    
    def _is_email_field(self, field_name: str) -> bool:
        """Check if field is likely an email field"""
        email_keywords = ['email', 'e-mail', 'mail']
        return any(kw in field_name.lower() for kw in email_keywords)
    
    def _email_similarity(self, email1: str, email2: str) -> float:
        """
        Compare emails with special handling for common typos
        """
        email1 = email1.lower().strip()
        email2 = email2.lower().strip()
        
        # Exact match
        if email1 == email2:
            return 1.0
        
        # Parse email parts
        parts1 = email1.split('@')
        parts2 = email2.split('@')
        
        if len(parts1) != 2 or len(parts2) != 2:
            return self._string_similarity(email1, email2)
        
        username1, domain1 = parts1
        username2, domain2 = parts2
        
        # Username similarity
        username_sim = self._string_similarity(username1, username2)
        
        # Domain similarity with common typo detection
        domain_sim = self._domain_similarity(domain1, domain2)
        
        # Weighted average (username 60%, domain 40%)
        return username_sim * 0.6 + domain_sim * 0.4
    
    def _domain_similarity(self, domain1: str, domain2: str) -> float:
        """
        Compare email domains with common typo detection
        """
        # Common email providers and their typos
        common_domains = {
            'gmail.com': ['gamil.com', 'gmai.com', 'gmial.com', 'gmail.co'],
            'yahoo.com': ['yaho.com', 'yahooo.com', 'yahoo.co'],
            'hotmail.com': ['hotmial.com', 'hotmai.com', 'hotmail.co'],
            'outlook.com': ['outlok.com', 'outlook.co'],
        }
        
        # Check if one is a known typo of the other
        for correct, typos in common_domains.items():
            if (domain1 == correct and domain2 in typos) or \
               (domain2 == correct and domain1 in typos):
                return 0.95  # High similarity for known typos
        
        # Otherwise use string similarity
        return self._string_similarity(domain1, domain2)
    
    def _is_phone_field(self, field_name: str) -> bool:
        """Check if field is likely a phone field"""
        phone_keywords = ['phone', 'tel', 'mobile', 'cell']
        return any(kw in field_name.lower() for kw in phone_keywords)
    
    def _phone_similarity(self, phone1: str, phone2: str) -> float:
        """
        Compare phone numbers after normalizing
        """
        # Extract digits only
        digits1 = re.sub(r'\D', '', str(phone1))
        digits2 = re.sub(r'\D', '', str(phone2))
        
        if not digits1 or not digits2:
            return 0.0
        
        # Exact match on digits
        if digits1 == digits2:
            return 1.0
        
        # Check last 10 digits (for country codes)
        if len(digits1) >= 10 and len(digits2) >= 10:
            if digits1[-10:] == digits2[-10:]:
                return 0.95
        
        # Fuzzy match on digits
        return self._string_similarity(digits1, digits2)


def render_duplicate_detection_page():
    """
    Render the duplicate detection UI
    Should be called from a dedicated page or section
    """
    st.header("🔍 Smart Duplicate Detection")
    st.markdown("Detect duplicates using multiple fields with intelligent confidence scoring")
    
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ Please upload data first")
        return
    
    df = st.session_state.df
    
    # Configuration section
    with st.expander("⚙️ Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Primary Identifier**")
            primary_field = st.selectbox(
                "Main field to check for duplicates",
                options=df.columns.tolist(),
                help="Usually name, email, or product ID"
            )
            
            fuzzy_threshold = st.slider(
                "Primary field similarity threshold",
                min_value=0.5,
                max_value=1.0,
                value=0.85,
                step=0.05,
                help="How similar values need to be (0.85 = 85% similar)"
            )
        
        with col2:
            st.markdown("**Supporting Fields**")
            available_fields = [col for col in df.columns if col != primary_field]
            supporting_fields = st.multiselect(
                "Fields to verify matches",
                options=available_fields,
                default=available_fields[:3] if len(available_fields) >= 3 else available_fields,
                help="Additional fields like email, phone, age to confirm duplicates"
            )
            
            confidence_threshold = st.slider(
                "Minimum confidence to report",
                min_value=0.3,
                max_value=1.0,
                value=0.6,
                step=0.05,
                help="Only show matches with at least this confidence"
            )
    
    # Detect button
    if st.button("🔍 Detect Duplicates", type="primary", use_container_width=True):
        with st.spinner("Analyzing for duplicates..."):
            detector = DuplicateDetector(df)
            results = detector.detect_duplicates(
                primary_field=primary_field,
                supporting_fields=supporting_fields,
                fuzzy_threshold=fuzzy_threshold,
                confidence_threshold=confidence_threshold
            )
            
            st.session_state.duplicate_results = results
    
    # Display results
    if 'duplicate_results' in st.session_state and st.session_state.duplicate_results:
        results = st.session_state.duplicate_results
        
        st.success(f"✅ Found {len(results)} potential duplicate pairs")
        
        # Categorize by confidence
        high_conf = [r for r in results if r['confidence'] >= 0.85]
        medium_conf = [r for r in results if 0.6 <= r['confidence'] < 0.85]
        low_conf = [r for r in results if r['confidence'] < 0.6]
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("High Confidence (≥85%)", len(high_conf))
        col2.metric("Medium Confidence (60-85%)", len(medium_conf))
        col3.metric("Low Confidence (<60%)", len(low_conf))
        
        # Tabs for each confidence level
        tab1, tab2, tab3 = st.tabs(["🔴 High Confidence", "🟡 Medium Confidence", "⚪ Low Confidence"])
        
        with tab1:
            if high_conf:
                render_duplicate_pairs(high_conf, primary_field, supporting_fields)
            else:
                st.info("No high confidence duplicates found")
        
        with tab2:
            if medium_conf:
                render_duplicate_pairs(medium_conf, primary_field, supporting_fields)
            else:
                st.info("No medium confidence duplicates found")
        
        with tab3:
            if low_conf:
                render_duplicate_pairs(low_conf, primary_field, supporting_fields)
            else:
                st.info("No low confidence duplicates found")
    
    elif 'duplicate_results' in st.session_state:
        st.info("ℹ️ No duplicates found with current settings. Try lowering the confidence threshold.")


def render_duplicate_pairs(results: List[Dict], primary_field: str, supporting_fields: List[str]):
    """
    Render duplicate pairs with merge options
    """
    if 'selected_merges' not in st.session_state:
        st.session_state.selected_merges = set()
    
    for i, result in enumerate(results):
        confidence = result['confidence']
        record1 = result['record_1']
        record2 = result['record_2']
        idx1 = result['index_1']
        idx2 = result['index_2']
        
        # Create unique key for this pair
        pair_key = f"{idx1}_{idx2}"
        
        with st.container():
            # Header with confidence
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                confidence_color = "🟢" if confidence >= 0.85 else "🟡" if confidence >= 0.6 else "⚪"
                st.markdown(f"### {confidence_color} Match #{i+1}: {confidence*100:.1f}% Confidence")
            
            with col2:
                merge_selected = st.checkbox(
                    "Select to merge",
                    key=f"merge_{pair_key}",
                    value=pair_key in st.session_state.selected_merges
                )
                if merge_selected:
                    st.session_state.selected_merges.add(pair_key)
                else:
                    st.session_state.selected_merges.discard(pair_key)
            
            with col3:
                if st.button("👁️ Details", key=f"details_{pair_key}"):
                    st.session_state[f"show_details_{pair_key}"] = \
                        not st.session_state.get(f"show_details_{pair_key}", False)
            
            # Show records side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Record A** (Row {})".format(idx1))
                display_dict1 = {primary_field: record1[primary_field]}
                for field in supporting_fields:
                    if field in record1:
                        display_dict1[field] = record1[field]
                st.json(display_dict1, expanded=False)
            
            with col2:
                st.markdown("**Record B** (Row {})".format(idx2))
                display_dict2 = {primary_field: record2[primary_field]}
                for field in supporting_fields:
                    if field in record2:
                        display_dict2[field] = record2[field]
                st.json(display_dict2, expanded=False)
            
            # Show matching details if expanded
            if st.session_state.get(f"show_details_{pair_key}", False):
                st.markdown("**Match Analysis:**")
                matches = result['matches']
                
                match_data = []
                for field, score in matches.items():
                    if score is None:
                        match_status = "❓ Both missing"
                        score_display = "N/A"
                    elif score == 0.0:
                        match_status = "❌ No match"
                        score_display = "0%"
                    elif score >= 0.95:
                        match_status = "✅ Perfect match"
                        score_display = f"{score*100:.0f}%"
                    elif score >= 0.85:
                        match_status = "✅ Strong match"
                        score_display = f"{score*100:.0f}%"
                    elif score >= 0.6:
                        match_status = "⚠️ Partial match"
                        score_display = f"{score*100:.0f}%"
                    else:
                        match_status = "❌ Weak match"
                        score_display = f"{score*100:.0f}%"
                    
                    match_data.append({
                        'Field': field,
                        'Match Score': score_display,
                        'Status': match_status
                    })
                
                st.dataframe(
                    pd.DataFrame(match_data),
                    use_container_width=True,
                    hide_index=True
                )
            
            st.divider()
    
    # Merge selected pairs button
    if st.session_state.selected_merges:
        st.markdown(f"**Selected {len(st.session_state.selected_merges)} pairs for merging**")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔀 Merge Selected", type="primary", use_container_width=True):
                merge_duplicate_pairs(results)
        with col2:
            if st.button("Clear Selection", use_container_width=True):
                st.session_state.selected_merges = set()
                st.rerun()


def merge_duplicate_pairs(results: List[Dict]):
    """
    Merge selected duplicate pairs
    """
    if 'df' not in st.session_state or not st.session_state.selected_merges:
        return
    
    df = st.session_state.df.copy()
    rows_to_drop = set()
    
    for pair_key in st.session_state.selected_merges:
        idx1, idx2 = map(int, pair_key.split('_'))
        
        # Find the result for this pair
        result = next((r for r in results if r['index_1'] == idx1 and r['index_2'] == idx2), None)
        if not result:
            continue
        
        # Keep the first record, drop the second
        # (In production, you might want to let user choose which to keep or merge values)
        rows_to_drop.add(idx2)
    
    # Drop duplicate rows
    df_cleaned = df.drop(index=list(rows_to_drop)).reset_index(drop=True)
    
    # Update session state
    from utils_robust import update_df, log_action
    
    log_action(
        "Merge Duplicates",
        f"Merged {len(rows_to_drop)} duplicate records"
    )
    update_df(df_cleaned, "Merged duplicate records")
    
    # Clear selection
    st.session_state.selected_merges = set()
    st.session_state.pop('duplicate_results', None)
    
    st.success(f"✅ Merged {len(rows_to_drop)} duplicate records!")
    st.rerun()


# Export main function
__all__ = [
    'DuplicateDetector',
    'render_duplicate_detection_page'
]