"""
ANALYTIQAL — Dataset Manager v8.0
Structured Analytics Workflow Engine

Workflow:
  Load → Schema → Import → Row Integrity → [Checkpoint] →
  Reshape → Column Restructuring → Audit → Export

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │  DATA LAYER (never modified by UI code)                 │
  │  • Immutable raw layer  — frozen at import              │
  │  • HistoryManager       — undo/redo snapshot stack      │
  │  • OperationLog         — append-only audit trail       │
  │  • QuarantineManager    — failed-row partition           │
  │  • CheckpointSnapshot   — post-row-integrity baseline   │
  ├─────────────────────────────────────────────────────────┤
  │  STATE LAYER                                            │
  │  • WorkflowState   — stage completion flags             │
  │  • WorkflowFlags   — one-time ops + UI preferences      │
  ├─────────────────────────────────────────────────────────┤
  │  CONFIG LAYER                                           │
  │  • STAGE_CONFIG    — declarative per-stage metadata     │
  │  • DESIGN (D)      — visual design token system         │
  ├─────────────────────────────────────────────────────────┤
  │  UI LAYER                                               │
  │  • StageRenderer contract: _stage_header → controls     │
  │  • Global utility bar: undo/redo, lock, export          │
  │  • Clickable progress rail with stage navigation        │
  │  • _msg() graduated errors — never raw tracebacks       │
  └─────────────────────────────────────────────────────────┘

  Failure propagation: HARD_STOP / SOFT_FAIL / WARN

Author: Analytiqal Team
Version: 8.0.0
"""

import io
import re
import uuid
import hashlib
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

MAX_FILE_SIZE_MB   = 1024
MAX_HISTORY_STATES = 50
WARN_FILE_SIZE_MB  = 500
PIVOT_SAMPLE_SIZE  = 5000
MAX_PIVOT_CELLS    = 2_000_000


BRAND_PRIMARY  = "#667eea"
BRAND_GRADIENT = "135deg, #667eea 0%, #764ba2 100%"

DEFAULT_MISSING_VALUES = ["-", "NA", "N/A", "null", "NULL", ""]

# ─── Design tokens ─────────────────────────────────────────────────────────
D = {
    "c_brand":        "#667eea",
    "c_brand_dark":   "#5a6fd8",
    "c_success":      "#27ae60",
    "c_warning":      "#f39c12",
    "c_danger":       "#e74c3c",
    "c_neutral":      "#6c757d",
    "c_surface":      "#f8f9ff",
    "c_border":       "#e8eaf0",
    "c_text_primary": "#1a1a2e",
    "c_text_muted":   "#6c757d",
    "t_xs":  "0.72rem",
    "t_sm":  "0.83rem",
    "t_base":"0.92rem",
    "t_lg":  "1.1rem",
    "t_xl":  "1.4rem",
    "sp_xs": "4px",
    "sp_sm": "8px",
    "sp_md": "16px",
    "sp_lg": "24px",
    "sp_xl": "40px",
}

# ─── Graduated message renderer ────────────────────────────────────────────
_MSG_CFG = {
    "block":   {"icon": "⛔", "bg": "#fef0f0", "border": "#e74c3c", "title": "Cannot continue"},
    "warn":    {"icon": "⚠",  "bg": "#fff8e1", "border": "#f39c12", "title": "Heads up"},
    "info":    {"icon": "ℹ",  "bg": "#f0f4ff", "border": "#667eea", "title": ""},
    "success": {"icon": "✓",  "bg": "#f0faf4", "border": "#27ae60", "title": ""},
}

def _msg(level: str, user_text: str, detail: str = ""):
    """Render a styled, graduated message. Never exposes raw tracebacks."""
    cfg = _MSG_CFG.get(level, _MSG_CFG["info"])
    title_part = f"<b>{cfg['title']} — </b>" if cfg["title"] else ""
    st.markdown(
        f"<div style='background:{cfg['bg']};border-left:4px solid {cfg['border']};"
        f"border-radius:6px;padding:10px 14px;margin:6px 0;font-size:{D['t_sm']};'>"
        f"{cfg['icon']} {title_part}{user_text}</div>",
        unsafe_allow_html=True,
    )
    if detail:
        with st.expander("Technical detail", expanded=False):
            st.code(detail, language=None)

def _card(subtitle: str):
    """Render the 'What this step does' explanation card."""
    st.markdown(
        f"<div style='background:{D['c_surface']};border:1px solid {D['c_border']};"
        f"border-radius:8px;padding:10px 16px;margin-bottom:{D['sp_md']};'>"
        f"<span style='font-size:{D['t_xs']};color:{D['c_text_muted']};text-transform:uppercase;"
        f"letter-spacing:0.06em;font-weight:600;'>What this step does</span><br>"
        f"<span style='font-size:{D['t_base']};color:{D['c_text_primary']};'>{subtitle}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

def _metric_strip(metrics: list):
    """Render a consistent horizontal metric strip. metrics = [(label,value,delta),...]"""
    cols = st.columns(len(metrics))
    for col, (label, value, delta) in zip(cols, metrics):
        with col:
            st.markdown(
                f"<div style='background:{D['c_surface']};border:1px solid {D['c_border']};"
                f"border-radius:8px;padding:12px;text-align:center;'>"
                f"<div style='font-size:{D['t_xs']};color:{D['c_text_muted']};text-transform:uppercase;"
                f"letter-spacing:0.06em;font-weight:600;margin-bottom:4px;'>{label}</div>"
                f"<div style='font-size:{D['t_xl']};font-weight:700;color:{D['c_text_primary']};'>{value}</div>"
                + (f"<div style='font-size:{D['t_xs']};color:{D['c_text_muted']};'>{delta}</div>" if delta else "")
                + "</div>",
                unsafe_allow_html=True,
            )


# ─── Semantic spacer ───────────────────────────────────────────────────────
def _space(size: str = "lg"):
    """Intentional, named vertical whitespace. size: xs|sm|md|lg|xl"""
    val = D.get(f"sp_{size}", D["sp_lg"])
    st.markdown(
        f"<div style='margin-top:{val};'></div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# WORKFLOW FLAGS  — centralised one-time operation state
# ═══════════════════════════════════════════════════════════════════════════

class WorkflowFlags:
    _KEY = "wf_flags"

    _DEFAULTS: Dict[str, Any] = {
        "norm_missing":   False,
        "dataset_locked": False,
    }

    @classmethod
    def _store(cls) -> Dict:
        if cls._KEY not in st.session_state:
            st.session_state[cls._KEY] = dict(cls._DEFAULTS)
        return st.session_state[cls._KEY]

    @classmethod
    def is_set(cls, key: str) -> bool:
        return bool(cls._store().get(key, False))

    @classmethod
    def set(cls, key: str, value: Any = True):
        cls._store()[key] = value

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        return cls._store().get(key, default)

    @classmethod
    def clear(cls, key: str):
        cls._store()[key] = cls._DEFAULTS.get(key, False)

    @classmethod
    def reset_all(cls):
        st.session_state[cls._KEY] = dict(cls._DEFAULTS)

    @classmethod
    def as_dict(cls) -> Dict:
        return dict(cls._store())


# ═══════════════════════════════════════════════════════════════════════════
# STAGE CONFIG REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

STAGE_CONFIG: Dict[str, Dict] = {
    "loaded": {
        "step":     1,
        "icon":     "📥",
        "title":    "Load",
        "subtitle": "Upload your dataset. It's validated before anything is loaded into the pipeline.",
        "skippable": False,
        "has_undo":  False,
    },
    "schema_done": {
        "step":     2,
        "icon":     "🏷️",
        "title":    "Schema",
        "subtitle": "Clean up column names — trim spaces, fix casing, and remove special characters.",
        "skippable": True,
        "has_undo":  False,
    },
    "imported": {
        "step":     3,
        "icon":     "👀",
        "title":    "Import",
        "subtitle": "Choose which columns to bring in. You can always go back if you need to adjust.",
        "skippable": False,
        "has_undo":  False,
    },
    "row_integrity": {
        "step":     4,
        "icon":     "🧹",
        "title":    "Row Integrity",
        "subtitle": "Fix row-level issues — remove duplicates, empty rows, and normalise missing value tokens. A checkpoint is taken when you continue.",
        "skippable": True,
        "has_undo":  True,
    },
    "reshaped": {
        "step":     5,
        "icon":     "🔄",
        "title":    "Reshape",
        "subtitle": "Change how your data is oriented — wide to long, or long to wide. Skip if your data is already in the right shape.",
        "skippable": True,
        "has_undo":  True,
    },
    "col_restructured": {
        "step":     6,
        "icon":     "📋",
        "title":    "Columns",
        "subtitle": "Fine-tune your columns — fix data types, rename, split, or combine them.",
        "skippable": True,
        "has_undo":  True,
    },
    "audited": {
        "step":     7,
        "icon":     "🔎",
        "title":    "Audit",
        "subtitle": "Review what changed since the checkpoint. Resolve any quarantined rows before exporting.",
        "skippable": False,
        "has_undo":  False,
    },
}


def _stage_header(stage_key: str):
    cfg      = STAGE_CONFIG.get(stage_key, {})
    step     = cfg.get("step", "")
    icon     = cfg.get("icon", "")
    title    = cfg.get("title", stage_key)
    subtitle = cfg.get("subtitle", "")

    _section_header(f"{icon} Step {step} — {title}")
    if subtitle:
        _card(subtitle)


# ─── V1 Workflow stages ────────────────────────────────────────────────────
WORKFLOW_STAGES = [
    "loaded",
    "schema_done",
    "imported",
    "row_integrity",
    "reshaped",
    "col_restructured",
    "audited",
]

STAGE_LABELS = {
    "loaded":           "Load",
    "schema_done":      "Schema",
    "imported":         "Import",
    "row_integrity":    "Row Ops",
    "reshaped":         "Reshape",
    "col_restructured": "Columns",
    "audited":          "Audit",
}

# ─── Failure classes ───────────────────────────────────────────────────────
HARD_STOP = "hard_stop"
SOFT_FAIL = "soft_fail"
WARN_ONLY = "warn"

# ─── Operation type constants ──────────────────────────────────────────────
OP_IMPORT         = "import"
OP_SCHEMA         = "schema"
OP_DEDUP          = "dedup"
OP_DROP_NULL      = "drop_null"
OP_NORM_MISSING   = "normalize_missing"
OP_DTYPE          = "dtype_change"
OP_DROP_COL       = "drop_column"
OP_RENAME         = "rename"
OP_SPLIT          = "split"
OP_MERGE          = "merge"
OP_ROW_FILTER     = "row_filter"
OP_PIVOT          = "pivot"
OP_MELT           = "melt"
OP_MEASURE_RENAME = "measure_rename"
OP_UNDO           = "undo"


# ═══════════════════════════════════════════════════════════════════════════
# OPERATION LOG
# ═══════════════════════════════════════════════════════════════════════════

class OperationLog:

    @staticmethod
    def _log() -> List[Dict]:
        if "op_log" not in st.session_state:
            st.session_state.op_log = []
        return st.session_state.op_log

    @staticmethod
    def append(
        phase: str,
        operation: str,
        parameters: Dict,
        rows_before: int,
        rows_after: int,
        cols_before: int,
        cols_after: int,
        status: str = "success",
        quarantine_count: int = 0,
        error_message: str = None,
    ) -> str:
        entry = {
            "id":               str(uuid.uuid4())[:8],
            "timestamp":        datetime.now().strftime("%H:%M:%S"),
            "phase":            phase,
            "operation":        operation,
            "parameters":       parameters,
            "rows_before":      rows_before,
            "rows_after":       rows_after,
            "cols_before":      cols_before,
            "cols_after":       cols_after,
            "row_delta":        rows_after - rows_before,
            "col_delta":        cols_after - cols_before,
            "quarantine_count": quarantine_count,
            "status":           status,
            "error_message":    error_message,
        }
        OperationLog._log().append(entry)
        return entry["id"]

    @staticmethod
    def get_all() -> List[Dict]:
        return OperationLog._log()

    @staticmethod
    def mark_undone(entry_id: str):
        for e in OperationLog._log():
            if e["id"] == entry_id:
                e["status"] = "undone"
                return

    @staticmethod
    def clear():
        st.session_state.op_log = []


# ═══════════════════════════════════════════════════════════════════════════
# QUARANTINE MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class QuarantineManager:

    @staticmethod
    def _store() -> List[Dict]:
        if "quarantine" not in st.session_state:
            st.session_state.quarantine = []
        return st.session_state.quarantine

    @staticmethod
    def add(rows: pd.DataFrame, failure_stage: str,
            failure_reason: str, failed_column: str = None):
        if rows.empty:
            return
        for idx, row in rows.iterrows():
            QuarantineManager._store().append({
                "id":              str(uuid.uuid4())[:8],
                "original_row_id": int(idx),
                "failure_stage":   failure_stage,
                "failure_reason":  failure_reason,
                "failed_column":   failed_column,
                "row_data":        row.to_dict(),
                "resolution":      "pending",
                "timestamp":       datetime.now().strftime("%H:%M:%S"),
            })

    @staticmethod
    def get_pending() -> List[Dict]:
        return [r for r in QuarantineManager._store() if r["resolution"] == "pending"]

    @staticmethod
    def count_pending() -> int:
        return len(QuarantineManager.get_pending())

    @staticmethod
    def resolve(ids: List[str], resolution: str):
        for r in QuarantineManager._store():
            if r["id"] in ids:
                r["resolution"] = resolution

    @staticmethod
    def export_blocked() -> bool:
        return QuarantineManager.count_pending() > 0

    @staticmethod
    def as_dataframe() -> pd.DataFrame:
        rows = QuarantineManager.get_pending()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([{
            "id":           r["id"],
            "stage":        r["failure_stage"],
            "reason":       r["failure_reason"],
            "column":       r["failed_column"] or "—",
            "original_row": r["original_row_id"],
            "timestamp":    r["timestamp"],
        } for r in rows])

    @staticmethod
    def clear():
        st.session_state.quarantine = []


# ═══════════════════════════════════════════════════════════════════════════
# CHECKPOINT SNAPSHOT
# ═══════════════════════════════════════════════════════════════════════════

class CheckpointSnapshot:

    @staticmethod
    def take(df: pd.DataFrame, checkpoint_type: str = "post_row_integrity"):
        col_schema = [
            {
                "name":         col,
                "dtype":        str(df[col].dtype),
                "null_pct":     round(df[col].isna().mean() * 100, 2),
                "unique_count": int(df[col].nunique()),
            }
            for col in df.columns
        ]
        raw_hash = hashlib.sha256(
            pd.util.hash_pandas_object(df, index=True).values.tobytes()
        ).hexdigest()[:16]

        st.session_state.checkpoint = {
            "checkpoint_type":  checkpoint_type,
            "row_count":        len(df),
            "col_count":        len(df.columns),
            "null_pct":         round(df.isnull().mean().mean() * 100, 2),
            "memory_bytes":     int(df.memory_usage(deep=True).sum()),
            "data_hash":        raw_hash,
            "col_schema":       col_schema,
            "created_at":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "quarantine_count": QuarantineManager.count_pending(),
        }

    @staticmethod
    def get() -> Optional[Dict]:
        return st.session_state.get("checkpoint")

    @staticmethod
    def exists() -> bool:
        return "checkpoint" in st.session_state


# ═══════════════════════════════════════════════════════════════════════════
# HISTORY / UNDO-REDO
# ═══════════════════════════════════════════════════════════════════════════

class StateSnapshot:
    def __init__(self, df: pd.DataFrame, description: str,
                 op_type: str, op_log_id: str = None):
        self.state_id    = str(uuid.uuid4())[:8]
        self.timestamp   = datetime.now().strftime("%H:%M:%S")
        self.df          = df.copy()
        self.description = description
        self.op_type     = op_type
        self.op_log_id   = op_log_id
        self.shape       = df.shape


class HistoryManager:

    @staticmethod
    def _stack() -> List[StateSnapshot]:
        if "history_stack" not in st.session_state:
            st.session_state.history_stack = []
        return st.session_state.history_stack

    @staticmethod
    def _idx() -> int:
        return st.session_state.get("history_index", -1)

    @staticmethod
    def push(df: pd.DataFrame, description: str,
             op_type: str, op_log_id: str = None):
        stack = HistoryManager._stack()
        idx   = HistoryManager._idx()
        st.session_state.history_stack = stack[:idx + 1]
        snap = StateSnapshot(df, description, op_type, op_log_id)
        st.session_state.history_stack.append(snap)
        if len(st.session_state.history_stack) > MAX_HISTORY_STATES:
            st.session_state.history_stack.pop(0)
        else:
            st.session_state.history_index = idx + 1
        st.session_state.df = df.copy()

    @staticmethod
    def can_undo() -> bool:
        return HistoryManager._idx() > 0

    @staticmethod
    def can_redo() -> bool:
        return HistoryManager._idx() < len(HistoryManager._stack()) - 1

    @staticmethod
    def undo() -> Optional[str]:
        if not HistoryManager.can_undo():
            return None
        idx  = HistoryManager._idx() - 1
        snap = HistoryManager._stack()[idx]
        st.session_state.history_index = idx
        st.session_state.df = snap.df.copy()
        if snap.op_log_id:
            OperationLog.mark_undone(snap.op_log_id)
        return snap.description

    @staticmethod
    def redo() -> Optional[str]:
        if not HistoryManager.can_redo():
            return None
        idx  = HistoryManager._idx() + 1
        snap = HistoryManager._stack()[idx]
        st.session_state.history_index = idx
        st.session_state.df = snap.df.copy()
        return snap.description

    @staticmethod
    def clear():
        st.session_state.history_stack = []
        st.session_state.history_index = -1


# ═══════════════════════════════════════════════════════════════════════════
# TRANSFORMATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class TransformResult:
    def __init__(self, df: pd.DataFrame, quarantine: pd.DataFrame = None,
                 warnings: List[str] = None, failure_class: str = None,
                 error: str = None):
        self.df            = df
        self.quarantine    = quarantine if quarantine is not None else pd.DataFrame()
        self.warnings      = warnings or []
        self.failure_class = failure_class
        self.error         = error
        self.rows_out      = len(df)
        self.cols_out      = len(df.columns)


class TransformationEngine:

    @staticmethod
    def remove_duplicates(df: pd.DataFrame, subset: List[str] = None,
                          keep: str = "first") -> TransformResult:
        try:
            new_df = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
            removed = len(df) - len(new_df)
            return TransformResult(new_df, warnings=[f"{removed:,} duplicate rows removed"])
        except Exception as e:
            return TransformResult(df, error=str(e), failure_class=HARD_STOP)

    @staticmethod
    def normalize_missing(df: pd.DataFrame, tokens: List[str]) -> TransformResult:
        try:
            new_df = df.copy()
            for col in new_df.select_dtypes(include="object").columns:
                new_df[col] = new_df[col].replace(tokens, np.nan)
            return TransformResult(new_df)
        except Exception as e:
            return TransformResult(df, error=str(e), failure_class=HARD_STOP)

    @staticmethod
    def drop_all_null_rows(df: pd.DataFrame) -> TransformResult:
        try:
            new_df = df[~df.isnull().all(axis=1)].reset_index(drop=True)
            return TransformResult(
                new_df,
                warnings=[f"{len(df) - len(new_df):,} all-null rows removed"]
            )
        except Exception as e:
            return TransformResult(df, error=str(e), failure_class=HARD_STOP)

    @staticmethod
    def correct_dtypes(df: pd.DataFrame,
                       dtype_map: Dict[str, str]) -> TransformResult:
        new_df         = df.copy()
        all_quarantine = pd.DataFrame()
        warnings       = []

        for col, target in dtype_map.items():
            if col not in new_df.columns:
                warnings.append(f"Column '{col}' not found — skipped")
                continue
            try:
                if target == "datetime64":
                    converted = pd.to_datetime(new_df[col], errors="coerce")
                    failed_mask = converted.isna() & new_df[col].notna()
                elif target in ("int64", "float64"):
                    converted = pd.to_numeric(new_df[col], errors="coerce")
                    failed_mask = converted.isna() & new_df[col].notna()
                elif target == "bool":
                    bool_map = {"True": True, "False": False,
                                "1": True, "0": False,
                                True: True, False: False,
                                1: True, 0: False}
                    converted   = new_df[col].map(bool_map)
                    failed_mask = converted.isna() & new_df[col].notna()
                else:
                    new_df[col] = new_df[col].astype(target)
                    continue

                n_failed = int(failed_mask.sum())
                if n_failed > 0:
                    bad_rows = df[failed_mask].copy()
                    bad_rows["__failure_reason__"] = f"Cannot convert '{col}' to {target}"
                    all_quarantine = pd.concat([all_quarantine, bad_rows])
                    new_df = new_df[~failed_mask].copy()
                    warnings.append(
                        f"'{col}': {n_failed} row(s) could not convert → quarantined"
                    )

                if target == "datetime64":
                    new_df[col] = pd.to_datetime(new_df[col], errors="coerce")
                elif target == "int64":
                    new_df[col] = (
                        pd.to_numeric(new_df[col], errors="coerce")
                        .fillna(0).astype("int64")
                    )
                elif target == "float64":
                    new_df[col] = pd.to_numeric(new_df[col], errors="coerce")
                elif target == "bool":
                    new_df[col] = new_df[col].map(bool_map)

            except Exception as e:
                return TransformResult(
                    df,
                    error=f"Hard stop on '{col}': {e}",
                    failure_class=HARD_STOP,
                )

        fc = SOFT_FAIL if not all_quarantine.empty else None
        return TransformResult(new_df, quarantine=all_quarantine,
                               warnings=warnings, failure_class=fc)

    @staticmethod
    def drop_columns(df: pd.DataFrame, columns: List[str]) -> TransformResult:
        try:
            missing = [c for c in columns if c not in df.columns]
            if missing:
                return TransformResult(
                    df, error=f"Columns not found: {missing}",
                    failure_class=HARD_STOP
                )
            return TransformResult(df.drop(columns=columns))
        except Exception as e:
            return TransformResult(df, error=str(e), failure_class=HARD_STOP)

    @staticmethod
    def rename_columns(df: pd.DataFrame,
                       rename_map: Dict[str, str]) -> TransformResult:
        try:
            new_names = [rename_map.get(c, c) for c in df.columns]
            if len(new_names) != len(set(new_names)):
                return TransformResult(
                    df,
                    error="Rename would create duplicate column names",
                    failure_class=HARD_STOP,
                )
            return TransformResult(df.rename(columns=rename_map))
        except Exception as e:
            return TransformResult(df, error=str(e), failure_class=HARD_STOP)

    @staticmethod
    def split_column(df: pd.DataFrame, col: str, delimiter: str,
                     n_parts: int, new_names: List[str]) -> TransformResult:
        try:
            new_df = df.copy()
            parts  = new_df[col].astype(str).str.split(
                delimiter, n=n_parts - 1, expand=True
            )
            for i, name in enumerate(new_names):
                new_df[name] = parts[i] if i in parts.columns else None
            return TransformResult(new_df)
        except Exception as e:
            return TransformResult(df, error=str(e), failure_class=HARD_STOP)

    @staticmethod
    def merge_columns(df: pd.DataFrame, cols: List[str],
                      new_name: str, sep: str = " ") -> TransformResult:
        try:
            existing = set(df.columns)
            if new_name in existing and new_name not in cols:
                return TransformResult(
                    df,
                    error=f"Column '{new_name}' already exists",
                    failure_class=HARD_STOP,
                )
            new_df           = df.copy()
            new_df[new_name] = new_df[cols].astype(str).agg(sep.join, axis=1)
            return TransformResult(new_df)
        except Exception as e:
            return TransformResult(df, error=str(e), failure_class=HARD_STOP)

    @staticmethod
    def pivot_table(
        df: pd.DataFrame,
        index_cols: List[str],
        column_cols: List[str],
        value_cols: List[str],
        aggfunc: str,
        fill_val_str: str = "",
    ) -> TransformResult:
        try:
            work = df.copy()
            for vc in value_cols:
                work[vc] = pd.to_numeric(work[vc], errors="coerce")

            result = pd.pivot_table(
                work,
                index=index_cols,
                columns=column_cols,
                values=value_cols,
                aggfunc=aggfunc,
            ).reset_index()

            if fill_val_str.strip() != "":
                try:
                    fill_val = float(fill_val_str)
                except ValueError:
                    fill_val = fill_val_str
                result = result.fillna(fill_val)

            if isinstance(result.columns, pd.MultiIndex):
                result.columns = [
                    "_".join(str(x) for x in col if str(x)).strip("_")
                    for col in result.columns
                ]

            return TransformResult(result)
        except Exception as e:
            return TransformResult(df, error=str(e), failure_class=HARD_STOP)

    @staticmethod
    def melt(
        df: pd.DataFrame,
        id_vars: List[str],
        value_vars: List[str],
        var_name: str,
        value_name: str,
        split_after: bool = False,
        split_sep: str = "_",
        split_names: List[str] = None,
    ) -> TransformResult:
        try:
            result = pd.melt(
                df,
                id_vars=id_vars or None,
                value_vars=value_vars,
                var_name=var_name,
                value_name=value_name,
            )
            if split_after and split_names:
                split_df = result[var_name].str.split(split_sep, expand=True)
                for i, name in enumerate(split_names):
                    if i in split_df.columns:
                        result[name] = split_df[i]
            return TransformResult(result)
        except Exception as e:
            return TransformResult(df, error=str(e), failure_class=HARD_STOP)

    @staticmethod
    def row_filter(df: pd.DataFrame, col: str, op: str, val: Any,
                   keep_matching: bool = False) -> TransformResult:
        try:
            mask   = _build_row_mask(df, col, op, val)
            keep   = mask if keep_matching else ~mask
            new_df = df[keep].reset_index(drop=True)
            return TransformResult(
                new_df,
                warnings=[f"{int((~keep).sum()):,} rows removed"]
            )
        except Exception as e:
            return TransformResult(df, error=str(e), failure_class=HARD_STOP)


# ═══════════════════════════════════════════════════════════════════════════
# FILE VALIDATION GATE
# ═══════════════════════════════════════════════════════════════════════════

class ValidationResult:
    def __init__(self):
        self.checks: List[Dict] = []
        self.status = "PASS"

    def add(self, name: str, status: str, message: str):
        self.checks.append({"name": name, "status": status, "message": message})
        if status == "BLOCK":
            self.status = "BLOCK"
        elif status == "WARN" and self.status == "PASS":
            self.status = "WARN"


def validate_file(file_bytes: bytes, filename: str, fmt: str) -> ValidationResult:
    result = ValidationResult()

    if not file_bytes:
        result.add("Empty file", "BLOCK", "File is empty")
        return result

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        result.add("File size", "BLOCK",
                   f"{size_mb:.0f} MB exceeds {MAX_FILE_SIZE_MB} MB limit")
    elif size_mb > WARN_FILE_SIZE_MB:
        result.add("File size", "WARN", f"{size_mb:.0f} MB — large file, may be slow")
    else:
        result.add("File size", "PASS", f"{size_mb:.1f} MB — OK")

    if fmt in ("xlsx", "xls"):
        try:
            pd.ExcelFile(io.BytesIO(file_bytes))
            result.add("File integrity", "PASS", "Valid Excel archive")
        except Exception as e:
            result.add("File integrity", "BLOCK", f"Corrupt Excel file: {e}")
    elif fmt == "csv":
        try:
            sample = file_bytes[:4096].decode("utf-8", errors="replace")
            lines  = [l for l in sample.split("\n") if l.strip()]
            if len(lines) < 2:
                result.add("Row check", "WARN", "Fewer than 2 rows in sample")
            else:
                counts = [len(l.split(",")) for l in lines[:10]]
                if max(counts) - min(counts) > 2:
                    result.add("Row check", "WARN",
                               "Uneven column counts — check delimiter")
                else:
                    result.add("Row check", "PASS", "Consistent column counts")
        except Exception as e:
            result.add("CSV check", "WARN", str(e))
    elif fmt == "json":
        try:
            import json
            json.loads(file_bytes[:2048])
            result.add("JSON validity", "PASS", "Valid JSON")
        except Exception:
            try:
                pd.read_json(io.BytesIO(file_bytes[:4096]))
                result.add("JSON validity", "PASS", "Readable JSON")
            except Exception as e:
                result.add("JSON validity", "BLOCK", f"Invalid JSON: {e}")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def get_file_format(filename: str) -> str:
    return filename.lower().rsplit(".", 1)[-1]


def load_sheet_preview(file_bytes: bytes, fmt: str, sheet: str,
                        skip_rows: int = 0, header: int = 0,
                        index_col=None):
    """Return (preview_df_5rows, total_rows, total_cols)."""
    df = load_dataframe(file_bytes, fmt, sheet, skip_rows,
                        header=header, index_col=index_col)
    return df.head(5), len(df), len(df.columns)


def load_dataframe(file_bytes: bytes, fmt: str, sheet: str = None,
                   skip_rows: int = 0, missing: List[str] = None,
                   nrows: int = None, header: int = 0,
                   index_col=None) -> pd.DataFrame:
    if missing is None:
        missing = DEFAULT_MISSING_VALUES
    buf = io.BytesIO(file_bytes)
    kw  = dict(na_values=missing, skiprows=skip_rows, header=header)
    if fmt in ("xlsx", "xls"):
        df = pd.read_excel(buf, sheet_name=sheet or 0, nrows=nrows,
                           index_col=index_col, **kw)
    elif fmt == "csv":
        df = pd.read_csv(buf, nrows=nrows, index_col=index_col, **kw)
    elif fmt == "json":
        df = pd.read_json(buf)
        return df.head(nrows) if nrows else df
    elif fmt == "parquet":
        df = pd.read_parquet(buf)
        return df.head(nrows) if nrows else df
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    # Reset index so it becomes a regular column if index_col was used
    if index_col is not None:
        df = df.reset_index()
    return df


def detect_sheets(file_bytes: bytes, fmt: str) -> List[str]:
    if fmt in ("xlsx", "xls"):
        try:
            return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names
        except Exception:
            pass
    return []


def _build_row_mask(df: pd.DataFrame, col: str, op: str, val: Any) -> pd.Series:
    s = df[col]
    if op == "=":            return s == val
    if op == "!=":           return s != val
    if op == ">":            return s > val
    if op == "<":            return s < val
    if op == ">=":           return s >= val
    if op == "<=":           return s <= val
    if op == "contains":     return s.astype(str).str.contains(str(val), case=False, na=False)
    if op == "not contains": return ~s.astype(str).str.contains(str(val), case=False, na=False)
    if op == "is null":      return s.isna()
    if op == "is not null":  return s.notna()
    return pd.Series([False] * len(df), index=df.index)


def detect_type_suggestions(df: pd.DataFrame) -> Dict[str, str]:
    suggestions = {}
    for col in df.select_dtypes(include="object").columns:
        s = df[col].dropna().astype(str).head(200)
        if not len(s):
            continue
        if s.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}", na=False).mean() > 0.7:
            suggestions[col] = "datetime64"
        elif s.str.match(r"^-?\d+\.?\d*$", na=False).mean() > 0.7:
            suggestions[col] = "float64"
        elif df[col].nunique() / max(len(df), 1) < 0.05:
            suggestions[col] = "category"
    return suggestions


def get_numeric_like_cols(df: pd.DataFrame) -> List[str]:
    result = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            result.append(col)
        elif df[col].dtype == object:
            coerced  = pd.to_numeric(df[col], errors="coerce")
            non_null = df[col].notna().sum()
            if non_null > 0 and coerced.notna().sum() / non_null >= 0.5:
                result.append(col)
    return result


def _apply_transform(
    result: TransformResult,
    phase: str,
    operation: str,
    parameters: Dict,
    df_before: pd.DataFrame,
    description: str,
    op_type: str,
) -> bool:
    if result.failure_class == HARD_STOP:
        _msg("block", result.error or "An error occurred that prevented this operation.")
        OperationLog.append(
            phase, operation, parameters,
            len(df_before), len(df_before),
            len(df_before.columns), len(df_before.columns),
            status="hard_fail", error_message=result.error,
        )
        return False

    for w in result.warnings:
        st.warning(f"⚠️ {w}")

    q_count = 0
    if not result.quarantine.empty:
        QuarantineManager.add(
            result.quarantine,
            failure_stage=operation,
            failure_reason=result.warnings[0] if result.warnings else "Conversion failed",
        )
        q_count = len(result.quarantine)
        st.warning(f"⚠️ {q_count} row(s) quarantined — resolve before export")

    log_id = OperationLog.append(
        phase, operation, parameters,
        len(df_before), result.rows_out,
        len(df_before.columns), result.cols_out,
        status=result.failure_class or "success",
        quarantine_count=q_count,
    )
    HistoryManager.push(result.df, description, op_type, log_id)
    return True


# ═══════════════════════════════════════════════════════════════════════════
# WORKFLOW STATE
# ═══════════════════════════════════════════════════════════════════════════

class WorkflowState:

    @staticmethod
    def init():
        if "workflow" not in st.session_state:
            st.session_state.workflow = {s: False for s in WORKFLOW_STAGES}

    @staticmethod
    def done(stage: str):
        st.session_state.workflow[stage] = True

    @staticmethod
    def is_done(stage: str) -> bool:
        return st.session_state.workflow.get(stage, False)

    @staticmethod
    def go_back():
        for stage in reversed(WORKFLOW_STAGES):
            if st.session_state.workflow.get(stage, False):
                st.session_state.workflow[stage] = False
                return stage

    @staticmethod
    def reset():
        st.session_state.workflow = {s: False for s in WORKFLOW_STAGES}

    @staticmethod
    def current_index() -> int:
        for i, stage in enumerate(WORKFLOW_STAGES):
            if not st.session_state.workflow.get(stage, False):
                return i
        return len(WORKFLOW_STAGES)


# ═══════════════════════════════════════════════════════════════════════════
# SHARED UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════

def _progress_bar():
    current = WorkflowState.current_index()
    stages  = list(STAGE_LABELS.items())

    col_ratios = []
    for i in range(len(stages)):
        col_ratios.append(3)
        if i < len(stages) - 1:
            col_ratios.append(1)
    cols = st.columns(col_ratios)

    for i, (key, label) in enumerate(stages):
        done   = WorkflowState.is_done(key)
        active = (i == current)
        col    = cols[i * 2]

        if done:
            bg, symbol, lc, fw = D["c_success"], "✓", D["c_success"], "700"
        elif active:
            bg, symbol, lc, fw = D["c_brand"], "●", D["c_text_primary"], "700"
        else:
            bg, symbol, lc, fw = D["c_border"], "○", D["c_text_muted"], "400"

        with col:
            st.markdown(
                f"<div style='text-align:center;'>"
                f"<div style='width:26px;height:26px;border-radius:50%;background:{bg};"
                f"color:white;font-size:0.72rem;display:flex;align-items:center;"
                f"justify-content:center;margin:0 auto;font-weight:700;'>{symbol}</div>"
                f"<div style='font-size:0.67rem;color:{lc};margin-top:3px;"
                f"font-weight:{fw};white-space:nowrap;overflow:hidden;"
                f"text-overflow:ellipsis;'>{label}</div></div>",
                unsafe_allow_html=True,
            )
            if done:
                if st.button(
                    "Go", key=f"nav_{key}",
                    help=f"Return to {label}",
                    use_container_width=True,
                ):
                    _navigate_to_stage(key)

        if i < len(stages) - 1:
            bar_c = D["c_success"] if done else D["c_border"]
            cols[i * 2 + 1].markdown(
                f"<div style='height:2px;background:{bar_c};"
                f"margin-top:13px;border-radius:2px;'></div>",
                unsafe_allow_html=True,
            )

    st.markdown(f"<div style='margin-bottom:{D['sp_md']};'></div>",
                unsafe_allow_html=True)


def _navigate_to_stage(target_stage: str):
    idx = WORKFLOW_STAGES.index(target_stage)
    for stage in WORKFLOW_STAGES[idx:]:
        st.session_state.workflow[stage] = False
    st.rerun()


def _back_button(key: str):
    if st.button("← Back", key=f"back_{key}", help="Return to previous step"):
        WorkflowState.go_back()
        st.rerun()


def _undo_redo(key: str):
    c1, c2, _ = st.columns([1, 1, 5])
    with c1:
        if st.button("↩ Undo", key=f"undo_{key}",
                     disabled=not HistoryManager.can_undo(),
                     use_container_width=True):
            desc = HistoryManager.undo()
            st.toast(f"Undone: {desc}")
            st.rerun()
    with c2:
        if st.button("↪ Redo", key=f"redo_{key}",
                     disabled=not HistoryManager.can_redo(),
                     use_container_width=True):
            desc = HistoryManager.redo()
            st.toast(f"Redone: {desc}")
            st.rerun()


def _section_header(label: str):
    st.markdown(
        f"<div style='margin-bottom:{D['sp_md']};'>"
        f"<h3 style='color:{D['c_text_primary']};margin:0 0 6px 0;"
        f"font-size:{D['t_lg']};font-weight:700;letter-spacing:-0.01em;'>{label}</h3>"
        f"<div style='height:2px;background:linear-gradient({BRAND_GRADIENT});"
        f"border-radius:2px;width:48px;'></div></div>",
        unsafe_allow_html=True,
    )


def _quarantine_badge():
    n = QuarantineManager.count_pending()
    if n > 0:
        st.markdown(
            f"<div style='background:{D['c_danger']}18;border:1px solid {D['c_danger']}44;"
            f"color:{D['c_danger']};padding:5px 12px;border-radius:6px;"
            f"font-size:{D['t_sm']};display:inline-block;margin-bottom:8px;font-weight:600;'>"
            f"⚠ {n} quarantined row(s) — resolve in Audit before export</div>",
            unsafe_allow_html=True,
        )


def _full_reset():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()


def _render_global_controls():
    df = st.session_state.get("df")
    if df is None:
        return

    can_undo = HistoryManager.can_undo()
    can_redo = HistoryManager.can_redo()
    q_count  = QuarantineManager.count_pending()
    locked   = WorkflowFlags.is_set("dataset_locked")

    q_html = (
        f"&nbsp;·&nbsp;<span style='color:{D['c_danger']};font-weight:700;'>"
        f"⚠ {q_count} quarantined</span>"
        if q_count > 0 else ""
    )
    lock_html = (
        f"&nbsp;·&nbsp;<span style='color:{D['c_danger']};font-weight:600;"
        f"font-size:{D['t_xs']};'>🔒 LOCKED</span>"
        if locked else ""
    )
    st.markdown(
        f"<div style='background:{D['c_surface']};border:1px solid {D['c_border']};"
        f"border-radius:8px;padding:7px 14px;margin-bottom:{D['sp_sm']};'>"
        f"<span style='font-size:{D['t_sm']};color:{D['c_text_muted']};'>"
        f"📊 <b>{len(df):,}</b> rows &nbsp;·&nbsp; <b>{len(df.columns)}</b> cols"
        f"{q_html}{lock_html}</span></div>",
        unsafe_allow_html=True,
    )

    u1, u2, _ = st.columns([1, 1, 7])
    with u1:
        if st.button("↩ Undo", disabled=(not can_undo or locked),
                     use_container_width=True, key="gc_undo"):
            desc = HistoryManager.undo()
            if desc:
                st.toast(f"Undone: {desc}")
            st.rerun()
    with u2:
        if st.button("↪ Redo", disabled=(not can_redo or locked),
                     use_container_width=True, key="gc_redo"):
            desc = HistoryManager.redo()
            if desc:
                st.toast(f"Redone: {desc}")
            st.rerun()

    with st.expander("⚙ Utilities", expanded=False):
        st.markdown(
            f"<p style='font-size:{D['t_xs']};color:{D['c_text_muted']};"
            f"margin:0 0 8px 0;text-transform:uppercase;letter-spacing:0.06em;"
            f"font-weight:600;'>File & Workflow</p>",
            unsafe_allow_html=True,
        )
        ua, ub, uc = st.columns(3)
        with ua:
            if st.button("🔄 Load new file", use_container_width=True,
                         key="util_reload",
                         help="Discard current dataset and start fresh"):
                _full_reset()
        with ub:
            st.download_button(
                "💾 Export CSV",
                df.to_csv(index=False),
                "dataset_current.csv",
                "text/csv",
                use_container_width=True,
                key="util_export",
                help="Download current dataset as CSV",
            )
        with uc:
            lock_label = "🔒 Unlock" if locked else "🔓 Lock edits"
            if st.button(lock_label, use_container_width=True, key="util_lock",
                         help="Lock prevents any further transformations"):
                WorkflowFlags.set("dataset_locked", not locked)
                st.rerun()

        st.markdown(
            f"<div style='height:1px;background:{D['c_border']};"
            f"margin:10px 0;'></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<p style='font-size:{D['t_xs']};color:{D['c_text_muted']};"
            f"margin:0 0 8px 0;text-transform:uppercase;letter-spacing:0.06em;"
            f"font-weight:600;'>One-time Operations</p>",
            unsafe_allow_html=True,
        )

        if not WorkflowFlags.is_set("norm_missing"):
            ot1, ot2 = st.columns([3, 1])
            with ot1:
                st.caption(
                    "**Normalise missing values** — replace tokens like "
                    '"-", "NA", "N/A", "null" with standard null.'
                )
            with ot2:
                if st.button("Run", key="ot_norm", use_container_width=True,
                             disabled=locked):
                    tokens = ["-", "NA", "N/A", "null", "NULL", ""]
                    r = TransformationEngine.normalize_missing(df, tokens)
                    if _apply_transform(
                        r, "utility", OP_NORM_MISSING,
                        {"tokens": tokens}, df,
                        "Normalise missing values (utility)", OP_NORM_MISSING,
                    ):
                        WorkflowFlags.set("norm_missing")
                        st.toast("✅ Missing values normalised")
                        st.rerun()
        else:
            st.caption("✅ Missing value normalisation already applied.")

        st.markdown(
            f"<div style='height:1px;background:{D['c_border']};"
            f"margin:10px 0;'></div>",
            unsafe_allow_html=True,
        )
        st.caption("**Reset workflow** — returns to step 1. Your raw data is preserved in session.")
        if st.button("⚠ Reset workflow", key="util_reset",
                     use_container_width=True,
                     help="Resets all stage flags but keeps the loaded data"):
            WorkflowState.reset()
            WorkflowFlags.reset_all()
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1 — LOAD + FILE VALIDATION GATE
# ═══════════════════════════════════════════════════════════════════════════

def render_load_stage():
    _stage_header("loaded")

    uploaded = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls", "json", "parquet"],
        key="file_uploader",
    )
    if not uploaded:
        return

    fmt       = get_file_format(uploaded.name)
    raw_bytes = uploaded.getvalue()

    # ── Validation gate ────────────────────────────────────────────────────
    val = validate_file(raw_bytes, uploaded.name, fmt)
    with st.expander("🔍 Validation results",
                     expanded=(val.status != "PASS")):
        for c in val.checks:
            icon = {"PASS": "✅", "WARN": "⚠️", "BLOCK": "❌"}.get(c["status"], "•")
            st.markdown(f"{icon} **{c['name']}** — {c['message']}")

    if val.status == "BLOCK":
        st.error("❌ File blocked. Fix the issues above and re-upload.")
        return
    if val.status == "WARN":
        if not st.checkbox("I've reviewed the warnings — proceed anyway",
                           key="override_warn"):
            return

    # ── Load options ────────────────────────────────────────────────────────
    sheets = detect_sheets(raw_bytes, fmt)
    with st.expander("⚙️ Options", expanded=bool(sheets)):
        skip = st.number_input("Skip first N rows", 0, 100, 0, key="skip_rows")

        # ── Header row option ──────────────────────────────────────────────
        use_header = st.checkbox(
            "First row is header (column names)",
            value=True,
            key="opt_header",
            help="Uncheck if your file has no header row — columns will be auto-named 0, 1, 2…",
        )
        header_val = 0 if use_header else None

        # ── Index / row label column ───────────────────────────────────────
        use_index = st.checkbox(
            "Treat a column as the row index (convert to regular column)",
            value=False,
            key="opt_use_index",
            help="Use this when the file's first column is an ID or row label stored as the index",
        )
        index_col_val = None
        if use_index:
            index_col_input = st.text_input(
                "Column position (0 = first) or column name",
                value="0",
                key="opt_index_col",
                help="Enter a number like 0 for the first column, or type the exact column name",
            )
            try:
                index_col_val = int(index_col_input)
            except ValueError:
                index_col_val = index_col_input.strip() if index_col_input.strip() else None

        if sheets:
                st.markdown("**Select sheet**")
                
                # Create tabs for each sheet
                tabs = st.tabs(sheets)
                
                # Iterate through tabs
                for i, tab in enumerate(tabs):
                        with tab:
                                st.session_state["sel_sheet"] = sheets[i]
                                try:
                                        _preview, _nr, _nc = load_sheet_preview(
                                                raw_bytes, fmt, sheets[i],
                                                skip_rows=int(skip),
                                                header=header_val,
                                                index_col=index_col_val,
                                        )
                                        st.caption(f"📏 {_nr:,} rows × {_nc} columns")
                                        st.dataframe(_preview, use_container_width=True,
                                                     height=180, hide_index=True)
                                except Exception as _e:
                                        st.caption(f"Preview unavailable: {_e}")

        sheet = st.session_state.get("sel_sheet") if sheets else None

        st.markdown("**Missing value tokens**")
        mc1, mc2, mc3 = st.columns(3)
        with mc1: u_dash = st.checkbox('"-" as null',   value=True, key="m_dash")
        with mc2: u_na   = st.checkbox('"NA/N/A"',      value=True, key="m_na")
        with mc3: u_null = st.checkbox('"null/NULL"',   value=True, key="m_null")
        custom = st.text_input("Custom null token", key="m_custom")

    missing = []
    if u_dash:  missing.append("-")
    if u_na:    missing += ["NA", "N/A"]
    if u_null:  missing += ["null", "NULL"]
    if custom:  missing.append(custom)

    if st.button("Load Preview →", type="primary", use_container_width=True):
        with st.spinner("Reading…"):
            try:
                prev_df = load_dataframe(
                    raw_bytes, fmt, sheet, skip, missing,
                    nrows=10, header=header_val, index_col=index_col_val,
                )
                full_df = load_dataframe(
                    raw_bytes, fmt, sheet, skip, missing,
                    header=header_val, index_col=index_col_val,
                )
            except Exception as e:
                _msg("block", "Could not read the file. Check the format and try again.", detail=str(e))
                return
        st.session_state.update({
            "raw_bytes":  raw_bytes,
            "raw_name":   uploaded.name,
            "raw_fmt":    fmt,
            "preview_df": prev_df,
            "full_df":    full_df,
        })
        WorkflowState.done("loaded")
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2 — SCHEMA STANDARDISATION
# ═══════════════════════════════════════════════════════════════════════════

def render_schema_stage():
    if "full_df" not in st.session_state:
        st.warning("Session data lost — please re-upload your file.")
        WorkflowState.reset()
        st.rerun()
        return
#    _back_button("schema")
    _stage_header("schema_done")

    full_df = st.session_state.full_df
    cols    = full_df.columns.tolist()

    issues = {}
    for col in cols:
        flags = []
        if col != col.strip():           flags.append("whitespace")
        if "  " in col:                  flags.append("double spaces")
        for ch in r'\/!@#$%^&*(){}[]|<>?`~"\'':
            if ch in col:
                flags.append("special chars")
                break
        if flags:
            issues[col] = flags

    if issues:
        st.warning(f"⚠️ {len(issues)} column name(s) have issues")
        with st.expander("Details", expanded=False):
            for col, flags in issues.items():
                st.caption(f"• `{col}` — {', '.join(set(flags))}")
    else:
        st.success("✅ No column naming issues detected")

    c1, c2, c3 = st.columns(3)
    with c1: do_strip = st.checkbox("Trim whitespace",       value=True,  key="sc_strip")
    with c2: do_lower = st.checkbox("Lowercase",             value=False, key="sc_lower")
    with c3: do_snake = st.checkbox("snake_case (spaces→_)", value=True,  key="sc_snake")

    def apply_rules(col: str) -> str:
        if do_strip: col = col.strip()
        if do_lower: col = col.lower()
        if do_snake: col = col.replace(" ", "_").replace("-", "_")
        return col

    proposed = [apply_rules(c) for c in cols]
    dupes    = len(proposed) != len(set(proposed))

    if dupes:
        st.error("❌ Would create duplicate column names — adjust rules")
    else:
        changes = [(o, n) for o, n in zip(cols, proposed) if o != n]
        if changes:
            with st.expander(f"Preview {len(changes)} rename(s)"):
                for old, new in changes:
                    st.caption(f"• `{old}` → `{new}`")

    ca, cb = st.columns(2)
    with ca:
        if st.button("Apply & Continue →", type="primary",
                     use_container_width=True, disabled=dupes):
            rmap = {o: n for o, n in zip(cols, proposed) if o != n}
            if rmap:
                st.session_state.full_df = st.session_state.full_df.rename(columns=rmap)
            WorkflowState.done("schema_done")
            st.rerun()
    with cb:
        if st.button("Skip →", use_container_width=True, key="skip_schema"):
            WorkflowState.done("schema_done")
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3 — PREVIEW & COLUMN SELECTION → IMPORT
# ═══════════════════════════════════════════════════════════════════════════

def render_import_stage():
    if "full_df" not in st.session_state or "preview_df" not in st.session_state:
        st.warning("Session data lost — please re-upload your file.")
        WorkflowState.reset()
        st.rerun()
        return
#    _back_button("import")
    _stage_header("imported")

    full_df    = st.session_state.full_df
    preview_df = st.session_state.preview_df

    _metric_strip([
        ("Rows",    f"{len(full_df):,}",                                              ""),
        ("Columns", str(len(full_df.columns)),                                        ""),
        ("Size",    f"{full_df.memory_usage(deep=True).sum()/(1024**2):.1f} MB",      ""),
    ])

    st.dataframe(preview_df, use_container_width=True, height=220, hide_index=True)

    st.markdown("**Select columns to import**")
    r1, r2, r3 = st.columns([4, 1, 1])
    with r1:
        search = st.text_input("Search", key="col_search",
                               label_visibility="collapsed",
                               placeholder="Search columns…")
    with r2:
        if st.button("All", use_container_width=True, key="sel_all"):
            st.session_state.sel_cols = full_df.columns.tolist()
            st.rerun()
    with r3:
        if st.button("None", use_container_width=True, key="sel_none"):
            st.session_state.sel_cols = []
            st.rerun()

    filtered = (
        [c for c in full_df.columns if search.lower() in c.lower()]
        if search else full_df.columns.tolist()
    )
    selected = st.multiselect(
        "cols", filtered,
        default=st.session_state.get("sel_cols", filtered),
        label_visibility="collapsed",
        key="col_selector",
    )
    st.session_state.sel_cols = selected

    if selected:
        st.caption(f"{len(selected)} of {len(full_df.columns)} columns selected")
    else:
        st.warning("No columns selected")

    if st.button("✅ Import →", type="primary", use_container_width=True,
                 disabled=not selected):
        imported = full_df[selected].copy()
        log_id = OperationLog.append(
            "import", OP_IMPORT,
            {"columns": selected, "n_cols": len(selected)},
            0, len(imported), 0, len(imported.columns),
        )
        HistoryManager.push(imported, f"Import: {len(selected)} cols", OP_IMPORT, log_id)
        st.session_state.raw_df = imported.copy()
        for k in ("preview_df", "full_df", "sel_cols"):
            st.session_state.pop(k, None)
        WorkflowState.done("imported")
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# STRUCTURAL OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════

def render_structural_overview():
    df = st.session_state.get("df")
    if df is None:
        return

    st.markdown(f"<div style='height:{D['sp_xl']};'></div>", unsafe_allow_html=True)
    _section_header("📊 Dataset Overview")

    _metric_strip([
        ("Rows",    f"{len(df):,}",                                              ""),
        ("Columns", str(len(df.columns)),                                        ""),
        ("Missing", f"{df.isnull().mean().mean()*100:.1f}%",                     ""),
        ("Dupes",   f"{df.duplicated().sum():,}",                                ""),
        ("Size",    f"{df.memory_usage(deep=True).sum()/(1024**2):.1f} MB",      ""),
    ])

    tab_data, tab_profile, tab_log = st.tabs(["Data", "Profile", "Log"])

    with tab_data:
        pa, pb = st.columns([3, 1])
        with pa: n = st.slider("Rows", 5, min(200, len(df)), 20, 5, key="ov_n")
        with pb: view = st.selectbox("View", ["First", "Last", "Sample"], key="ov_view")
        disp = (df.head(n) if view == "First"
                else df.tail(n) if view == "Last"
                else df.sample(min(n, len(df)), random_state=42))
        st.dataframe(disp, use_container_width=True, height=300, hide_index=True)

    with tab_profile:
        rows = []
        for col in df.columns:
            s = df[col]
            rows.append({
                "Column":   col,
                "Type":     str(s.dtype),
                "Non-null": int(s.notna().sum()),
                "Null %":   f"{s.isna().mean() * 100:.1f}%",
                "Unique":   int(s.nunique()),
                "Sample":   str(s.dropna().iloc[0]) if s.notna().any() else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab_log:
        log = OperationLog.get_all()
        if not log:
            st.caption("No operations logged yet")
        else:
            ldf = pd.DataFrame([{
                "time":      e["timestamp"],
                "operation": e["operation"],
                "rows Δ":    f"{e['row_delta']:+,}",
                "cols Δ":    f"{e['col_delta']:+,}",
                "quarantine": e["quarantine_count"],
                "status":    e["status"],
            } for e in reversed(log)])
            st.dataframe(ldf, use_container_width=True, hide_index=True, height=240)

        _space("sm")
        with st.expander("🔧 Workflow flags (debug)", expanded=False):
            flags    = WorkflowFlags.as_dict()
            workflow = st.session_state.get("workflow", {})
            combined = {
                **{f"stage.{k}": ("✓ done" if v else "○ pending")
                   for k, v in workflow.items()},
                **{f"flag.{k}": v for k, v in flags.items()},
            }
            st.dataframe(
                pd.DataFrame(list(combined.items()), columns=["Key", "Value"]),
                use_container_width=True, hide_index=True, height=220,
            )


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4 — ROW INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════

def render_row_integrity_stage():
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("No dataset loaded.")
        return

    df = st.session_state.df

    if "dataset_locked" not in st.session_state:
        st.session_state.dataset_locked = False

    is_locked = bool(st.session_state.dataset_locked)

    _stage_header("row_integrity")

    try:
        n_dupes = int(df.duplicated().sum()) if not df.empty else 0
    except Exception:
        n_dupes = 0

    try:
        n_allnull = int(df.isnull().all(axis=1).sum()) if not df.empty else 0
    except Exception:
        n_allnull = 0

    total_rows = int(len(df)) if df is not None else 0

    _metric_strip([
        ("Duplicate rows", f"{n_dupes:,}", ""),
        ("All-null rows",  f"{n_allnull:,}", ""),
        ("Total rows",     f"{total_rows:,}", ""),
    ])

    _space("sm")

    # ── Quick actions ──────────────────────────────────────────────────────
    qc1, qc2 = st.columns(2)
    with qc1:
        if st.button(
            f"Remove {n_dupes:,} duplicates",
            key="qa_dup",
            disabled=(n_dupes == 0) or is_locked,
            use_container_width=True,
        ):
            r = TransformationEngine.remove_duplicates(df)
            if _apply_transform(r, "row_integrity", OP_DEDUP, {}, df,
                                f"Remove {n_dupes:,} duplicates", OP_DEDUP):
                st.rerun()

    with qc2:
        if st.button(
            f"Remove {n_allnull:,} all-null rows",
            key="qa_null",
            disabled=(n_allnull == 0) or is_locked,
            use_container_width=True,
        ):
            r = TransformationEngine.drop_all_null_rows(df)
            if _apply_transform(r, "row_integrity", OP_DROP_NULL, {}, df,
                                f"Remove {n_allnull:,} all-null rows", OP_DROP_NULL):
                st.rerun()

    _space("sm")

    # ── Row operations tabs ────────────────────────────────────────────────
    filter_tab, total_tab, index_tab = st.tabs([
        "🔍 Row Filter",
        "🔎 Detect Total / Summary Rows & Columns",
        "🔢 Remove by Index",
    ])

    # ── Tab 1: Condition-based row filter ──────────────────────────────────
    with filter_tab:
        st.caption("Filter rows based on a column condition.")
        _space("xs")

        col_name = st.selectbox("Column", df.columns.tolist(), key="rf_col")
        is_num   = pd.api.types.is_numeric_dtype(df[col_name])
        ops      = (
            ["=", "!=", ">", "<", ">=", "<=", "is null", "is not null"]
            if is_num else
            ["=", "!=", "contains", "not contains", "is null", "is not null"]
        )
        rc1, rc2 = st.columns(2)
        with rc1:
            op = st.selectbox("Operator", ops, key="rf_op")
        with rc2:
            null_op = op in ("is null", "is not null")
            if null_op:
                val = None
                st.text_input("Value", value="(not needed)", disabled=True, key="rf_val_d")
            elif is_num:
                val = st.number_input("Value", key="rf_val_n")
            else:
                val = st.text_input("Value", key="rf_val_s")

        # st.radio → st.tabs for action
        rm_tab, keep_tab = st.tabs(["🗑 Remove matching rows", "✅ Keep only matching rows"])
        with rm_tab:
            mode = "Remove matching"
        with keep_tab:
            mode = "Keep only matching"
        # Resolve which tab is active via session state
        _rf_mode = st.session_state.get("rf_mode_tab", "Remove matching")
        with rm_tab:
            if st.button("Use: Remove matching", key="rf_mode_rm",
                         use_container_width=True,
                         type="primary" if _rf_mode == "Remove matching" else "secondary"):
                st.session_state["rf_mode_tab"] = "Remove matching"
                st.rerun()
        with keep_tab:
            if st.button("Use: Keep only matching", key="rf_mode_keep",
                         use_container_width=True,
                         type="primary" if _rf_mode == "Keep only matching" else "secondary"):
                st.session_state["rf_mode_tab"] = "Keep only matching"
                st.rerun()
        mode     = st.session_state.get("rf_mode_tab", "Remove matching")
        has_val  = null_op or (val is not None and val != "")

        if has_val:
            try:
                mask      = _build_row_mask(df, col_name, op, val)
                keep_mask = mask if "Keep" in mode else ~mask
                removed   = int((~keep_mask).sum())
                remaining = len(df) - removed
                st.caption(f"→ {removed:,} rows affected, {remaining:,} remain")
                fa, fb = st.columns(2)
                with fa:
                    if st.button("👁 Preview", key="rf_prev", use_container_width=True):
                        st.dataframe(df[~keep_mask].head(30), height=200,
                                     use_container_width=True)
                with fb:
                    if st.button("Apply filter", type="primary", key="rf_apply",
                                 use_container_width=True, disabled=remaining == 0):
                        r = TransformationEngine.row_filter(
                            df, col_name, op, val,
                            keep_matching="Keep" in mode
                        )
                        if _apply_transform(
                            r, "row_integrity", OP_ROW_FILTER,
                            {"col": col_name, "op": op, "mode": mode},
                            df, f"Filter: {col_name} {op} '{val}'", OP_ROW_FILTER,
                        ):
                            st.rerun()
            except Exception as e:
                _msg("warn", "Could not build the filter condition. Check your column and value combination.", detail=str(e))

    # ── Tab 2: Total / summary detection ──────────────────────────────────
    with total_tab:
        _suspect_cols_check = [col for col in df.columns if _is_total_column(str(col))]
        _text_cols_check    = df.select_dtypes(include="object").columns.tolist()
        _suspect_rows_check = (
            df[_text_cols_check].apply(lambda c: c.map(_is_total_row_value)).any(axis=1).any()
            if _text_cols_check else False
        )
        _has_issues = bool(_suspect_cols_check) or _suspect_rows_check

        if _has_issues:
            _msg("warn", "Suspected total/summary columns or rows detected — review below.")
        else:
            _msg("success", "No total or summary patterns detected in this dataset.")

        st.caption(
            "Scans column names and row values for total/summary patterns "
            "(e.g. 'Grand Total', 'total_sales', 'Subtotal', 'ROW TOTAL')."
        )
        _space("xs")

        tc_tab, tr_tab = st.tabs(["📋 Column detection", "🔢 Row detection"])

        with tc_tab:
            suspect_cols = [col for col in df.columns if _is_total_column(str(col))]
            if not suspect_cols:
                _msg("success", "No total / summary columns detected.")
            else:
                _msg("warn", f"{len(suspect_cols)} suspected total / summary column(s) found.")
                preview_rows = []
                for col in suspect_cols:
                    sample = df[col].dropna().head(3).tolist()
                    preview_rows.append({
                        "Column":        str(col),
                        "Dtype":         str(df[col].dtype),
                        "Sample values": ", ".join(str(v) for v in sample) if sample else "—",
                    })
                st.dataframe(
                    preview_rows,
                    use_container_width=True,
                    hide_index=True,
                    height=min(200, 40 + len(suspect_cols) * 36),
                )
                col_left, col_right = st.columns([3, 1])
                with col_left:
                    cols_to_remove = st.multiselect(
                        "Select columns to remove",
                        options=suspect_cols,
                        default=suspect_cols,
                        key="total_cols_multiselect",
                    )
                with col_right:
                    st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
                    if st.button(
                        f"🗑️ Remove {len(cols_to_remove)} column(s)",
                        key="remove_total_cols",
                        disabled=len(cols_to_remove) == 0,
                        type="primary",
                        use_container_width=True,
                    ):
                        new_df = df.drop(columns=cols_to_remove)
                        log_id = OperationLog.append(
                            "row_integrity", OP_DROP_COL,
                            {"type": "total_col", "removed": cols_to_remove},
                            len(df), len(new_df),
                            len(df.columns), len(new_df.columns),
                        )
                        HistoryManager.push(new_df, f"Remove {len(cols_to_remove)} total column(s)", OP_DROP_COL, log_id)
                        st.toast(f"✅ Removed {len(cols_to_remove)} total/summary column(s)")
                        st.rerun()

        with tr_tab:
            text_cols = df.select_dtypes(include="object").columns.tolist()
            if not text_cols:
                _msg("info", "No text columns found to scan for total labels.")
            else:
                scan_cols = st.multiselect(
                    "Columns to scan for total labels",
                    options=text_cols,
                    default=text_cols,
                    key="total_row_scan_cols",
                )
                if scan_cols:
                    mask        = df[scan_cols].apply(lambda c: c.map(_is_total_row_value)).any(axis=1)
                    suspect_idx = df.index[mask].tolist()

                    if not suspect_idx:
                        _msg("success", "No total / summary rows detected.")
                    else:
                        _msg("warn", f"{len(suspect_idx)} suspected total / summary row(s) found.")
                        triggered_map = {
                            idx: [c for c in scan_cols if _is_total_row_value(df.loc[idx, c])]
                            for idx in suspect_idx
                        }
                        preview_rows = []
                        for idx in suspect_idx:
                            triggered = triggered_map[idx]
                            preview_rows.append({
                                "Row index":        idx,
                                "Triggered column": ", ".join(triggered),
                                "Value":            str(df.loc[idx, triggered[0]]) if triggered else "—",
                            })
                        st.dataframe(
                            preview_rows,
                            use_container_width=True,
                            hide_index=True,
                            height=min(200, 40 + len(suspect_idx) * 36),
                        )
                        tr_left, tr_right = st.columns([3, 1])
                        with tr_left:
                            rows_to_remove = st.multiselect(
                                "Select rows to remove",
                                options=suspect_idx,
                                default=suspect_idx,
                                key="total_rows_multiselect",
                                format_func=lambda i: (
                                    f"Row {i} — {triggered_map.get(i, [''])[0]}: "
                                    f"'{df.loc[i, triggered_map[i][0]]}'"
                                    if triggered_map.get(i) else f"Row {i}"
                                ),
                            )
                        with tr_right:
                            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
                            if st.button(
                                f"🗑️ Remove {len(rows_to_remove)} row(s)",
                                key="remove_total_rows",
                                disabled=len(rows_to_remove) == 0,
                                type="primary",
                                use_container_width=True,
                            ):
                                new_df = df.drop(index=rows_to_remove).reset_index(drop=True)
                                log_id = OperationLog.append(
                                    "row_integrity", OP_ROW_FILTER,
                                    {"type": "total_row_text", "removed": rows_to_remove},
                                    len(df), len(new_df),
                                    len(df.columns), len(new_df.columns),
                                )
                                HistoryManager.push(new_df, f"Remove {len(rows_to_remove)} total/summary row(s)", OP_ROW_FILTER, log_id)
                                st.toast(f"✅ Removed {len(rows_to_remove)} total/summary row(s)")
                                st.rerun()

    # ── Tab 3: Index-based row removal ─────────────────────────────────────
    with index_tab:
        st.caption(
            "Remove specific rows by their index numbers. "
            "Use comma-separated values and/or ranges (e.g. 0, 5, 10-15)."
        )
        _space("xs")

        idx_input = st.text_input(
            "Row indices",
            key="idx_remove_input",
            placeholder="e.g. 0, 3, 7-10, 25",
        )

        # st.radio → st.tabs
        rm_idx_tab, keep_idx_tab = st.tabs(["🗑 Remove these rows", "✅ Keep only these rows"])
        with rm_idx_tab:
            if st.button("Use: Remove these rows", key="idx_mode_rm",
                         use_container_width=True,
                         type="primary" if st.session_state.get("idx_mode_val", "Remove") == "Remove" else "secondary"):
                st.session_state["idx_mode_val"] = "Remove"
                st.rerun()
        with keep_idx_tab:
            if st.button("Use: Keep only these rows", key="idx_mode_keep",
                         use_container_width=True,
                         type="primary" if st.session_state.get("idx_mode_val", "Remove") == "Keep" else "secondary"):
                st.session_state["idx_mode_val"] = "Keep"
                st.rerun()
        idx_mode = st.session_state.get("idx_mode_val", "Remove")

        if idx_input:
            try:
                indices = set()
                for part in idx_input.split(","):
                    part = part.strip()
                    if "-" in part:
                        a, b = part.split("-", 1)
                        indices.update(range(int(a.strip()), int(b.strip()) + 1))
                    elif part:
                        indices.add(int(part))
                valid_idx = sorted(i for i in indices if 0 <= i < len(df))
                invalid   = indices - set(valid_idx)
                if invalid:
                    _msg("warn", f"Out of range (ignored): {sorted(invalid)}")
                st.caption(
                    f"→ {len(valid_idx)} row(s) selected  "
                    f"({'removing' if idx_mode == 'Remove' else 'keeping'} them)"
                )
                if valid_idx:
                    st.dataframe(df.iloc[valid_idx], height=180,
                                 use_container_width=True, hide_index=False)
                    if st.button("Apply", type="primary", key="idx_apply",
                                 use_container_width=True):
                        if idx_mode == "Remove":
                            new_df = df.drop(index=valid_idx).reset_index(drop=True)
                            desc   = f"Remove {len(valid_idx)} rows by index"
                        else:
                            new_df = df.iloc[valid_idx].reset_index(drop=True)
                            desc   = f"Keep {len(valid_idx)} rows by index"
                        log_id = OperationLog.append(
                            "row_integrity", OP_ROW_FILTER,
                            {"indices": valid_idx, "mode": idx_mode},
                            len(df), len(new_df),
                            len(df.columns), len(new_df.columns),
                        )
                        HistoryManager.push(new_df, desc, OP_ROW_FILTER, log_id)
                        st.rerun()
            except Exception as e:
                _msg("warn", "Could not parse the row indices. Use comma-separated numbers or ranges like 0, 3, 7-10.", detail=str(e))

    # ── Footer: checkpoint + continue ─────────────────────────────────────
    _space("lg")
    st.markdown(
        f"<div style='background:{D['c_surface']};border:1px solid {D['c_border']};"
        f"border-radius:8px;padding:10px 14px;margin-bottom:{D['sp_md']};'>"
        f"<span style='font-size:{D['t_sm']};color:{D['c_text_muted']};'>"
        f"ℹ A checkpoint snapshot will be taken automatically when you continue.</span></div>",
        unsafe_allow_html=True,
    )
    cc, cs = st.columns(2)
    with cc:
        if st.button("✅ Continue — take checkpoint →", type="primary",
                     use_container_width=True, key="rows_next"):
            CheckpointSnapshot.take(st.session_state.df)
            st.toast("✅ Checkpoint taken")
            WorkflowState.done("row_integrity")
            st.rerun()
    with cs:
        if st.button("Skip →", use_container_width=True, key="rows_skip"):
            CheckpointSnapshot.take(st.session_state.df)
            WorkflowState.done("row_integrity")
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 5 — COLUMN RESTRUCTURING
# ═══════════════════════════════════════════════════════════════════════════

def render_column_restructuring_stage():
    df = st.session_state.df
    _stage_header("col_restructured")

    locked = WorkflowFlags.is_set("dataset_locked")
    if locked:
        _msg("warn", "Dataset is locked. Unlock in the Utilities panel to make changes.")
        return

    # st.radio → st.tabs
    dt_tab, ren_tab, split_tab, merge_tab, drop_tab, replace_tab = st.tabs([
        "📐 Data Types",
        "✏️ Rename",
        "✂️ Split",
        "🔗 Merge",
        "🗑 Drop Columns",
        "🔄 Replace Values",
    ])

    with dt_tab:
        _render_dtype_correction(df)

    with ren_tab:
        _render_rename(df)

    with split_tab:
        _render_split(df)

    with merge_tab:
        _render_merge(df)

    with drop_tab:
        _render_drop_columns(df)

    with replace_tab:
        _render_replace_values(df)

    # ── Footer ─────────────────────────────────────────────────────────────
    _space("lg")
    cc, cs = st.columns(2)
    with cc:
        if st.button("✅ Continue →", type="primary",
                     use_container_width=True, key="cols_next"):
            WorkflowState.done("col_restructured")
            st.rerun()
    with cs:
        if st.button("Skip →", use_container_width=True, key="cols_skip"):
            WorkflowState.done("col_restructured")
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# COLUMN SUB-RENDERERS  (unchanged except _render_replace_values)
# ═══════════════════════════════════════════════════════════════════════════

def _render_dtype_correction(df: pd.DataFrame):
    suggestions = detect_type_suggestions(df)
    if suggestions:
        st.caption(f"ℹ️ {len(suggestions)} column(s) have suggested type changes (marked ← rec)")

    if "pending_types" not in st.session_state:
        st.session_state.pending_types = {}

    TYPE_OPTS = ["(no change)", "int64", "float64", "object",
                 "category", "datetime64", "bool"]

    hc = st.columns([2, 1.5, 1, 0.8, 0.8, 2])
    for c, lbl in zip(hc, ["Column", "Current Type", "Non-null",
                            "Unique", "Null%", "Change to"]):
        c.markdown(f"**{lbl}**")

    for col in df.columns:
        s     = df[col]
        cur   = str(s.dtype)
        sug   = suggestions.get(col)
        rc    = st.columns([2, 1.5, 1, 0.8, 0.8, 2])
        rc[0].write(col)
        rc[1].write(cur)
        rc[2].write(f"{s.notna().sum():,}")
        rc[3].write(f"{s.nunique():,}")
        rc[4].write(f"{s.isna().mean()*100:.0f}%")
        with rc[5]:
            opts = [f"{t} ← rec" if (t == sug) else t for t in TYPE_OPTS]
            default_idx = (TYPE_OPTS.index(sug) if sug and sug in TYPE_OPTS else 0)
            chosen = st.selectbox("", opts, index=default_idx,
                                  key=f"dt_{col}", label_visibility="collapsed")
            actual = chosen.replace(" ← rec", "")
            if actual != "(no change)" and actual != cur:
                st.session_state.pending_types[col] = actual
            elif col in st.session_state.pending_types:
                del st.session_state.pending_types[col]

    pending = st.session_state.pending_types
    st.markdown("---")
    dc1, dc2 = st.columns(2)
    with dc1:
        n_pending = len(pending)
        label = f"Apply {n_pending} change(s)" if n_pending else "No changes selected"
        if st.button(label, type="primary", use_container_width=True,
                     key="apply_dtypes", disabled=n_pending == 0):
            r = TransformationEngine.correct_dtypes(df, pending.copy())
            st.session_state.pending_types = {}
            if _apply_transform(r, "col_restructuring", OP_DTYPE,
                                {"n": n_pending}, df,
                                f"Types: {n_pending} col(s)", OP_DTYPE):
                st.rerun()
    with dc2:
        if st.button("Clear all", use_container_width=True, key="clear_dtypes",
                     disabled=len(pending) == 0):
            st.session_state.pending_types = {}
            st.rerun()


def _render_drop_columns(df: pd.DataFrame):
    const_cols = [c for c in df.columns if df[c].nunique() <= 1]
    high_null  = [c for c in df.columns if df[c].isna().mean() > 0.9]
    if const_cols:
        _msg("info", f"Constant columns (safe to drop): {', '.join(const_cols)}")
    if high_null:
        _msg("warn", f"Columns with >90% nulls: {', '.join(high_null)}")

    to_drop = st.multiselect("Columns to drop", df.columns.tolist(), key="drop_sel")
    if not to_drop:
        return
    st.caption(f"→ {len(df.columns) - len(to_drop)} columns remaining after drop")
    if st.button("Drop selected", type="primary", use_container_width=True, key="apply_drop"):
        r = TransformationEngine.drop_columns(df, to_drop)
        if _apply_transform(r, "col_restructuring", OP_DROP_COL,
                            {"columns": to_drop}, df,
                            f"Drop {len(to_drop)} col(s)", OP_DROP_COL):
            st.rerun()


def _render_rename(df: pd.DataFrame):
    sel = st.multiselect("Columns to rename", df.columns.tolist(), key="ren_sel")
    if not sel:
        return
    rmap = {}
    for col in sel:
        new = st.text_input(f"`{col}` →", value=col, key=f"rn_{col}")
        if new and new != col:
            rmap[col] = new
    if rmap and st.button("Apply renames", type="primary",
                          use_container_width=True, key="apply_ren"):
        r = TransformationEngine.rename_columns(df, rmap)
        if _apply_transform(r, "col_restructuring", OP_RENAME, {"renames": rmap},
                            df, f"Rename {len(rmap)} col(s)", OP_RENAME):
            st.rerun()


def _render_split(df: pd.DataFrame):
    col   = st.selectbox("Column to split", df.columns.tolist(), key="split_col")
    delim = st.text_input("Delimiter", value=",", key="split_delim")
    n     = int(st.number_input("Number of parts", 2, 10, 2, key="split_n"))

    names = []
    nc = st.columns(min(n, 4))
    for i in range(n):
        with nc[i % len(nc)]:
            names.append(st.text_input(f"Part {i+1}", value=f"{col}_{i+1}",
                                       key=f"sn_{i}"))
    if delim:
        sample = df[col].astype(str).head(5).str.split(delim, n=n - 1, expand=True)
        st.caption("Sample:")
        st.dataframe(sample, height=130, use_container_width=True, hide_index=True)

    if st.button("Apply split", type="primary", use_container_width=True, key="apply_split"):
        r = TransformationEngine.split_column(df, col, delim, n, names)
        if _apply_transform(r, "col_restructuring", OP_SPLIT,
                            {"col": col, "delim": delim}, df,
                            f"Split '{col}'", OP_SPLIT):
            st.rerun()


def _render_merge(df: pd.DataFrame):
    cols = st.multiselect("Columns to merge (min 2)", df.columns.tolist(), key="merge_sel")
    if len(cols) < 2:
        return
    sep  = st.text_input("Separator", value=" ", key="merge_sep")
    name = st.text_input("New column name",
                         value="_".join(c[:6] for c in cols[:2]),
                         key="merge_name")
    if name and cols:
        sample = df[cols].astype(str).head(5).agg(sep.join, axis=1)
        st.caption("Sample:")
        st.dataframe(sample.rename(name).to_frame(), height=130,
                     use_container_width=True, hide_index=True)
    if st.button("Apply merge", type="primary",
                 use_container_width=True, key="apply_merge"):
        r = TransformationEngine.merge_columns(df, cols, name, sep)
        if _apply_transform(r, "col_restructuring", OP_MERGE,
                            {"cols": cols, "new": name}, df,
                            f"Merge → '{name}'", OP_MERGE):
            st.rerun()


def _render_replace_values(df: pd.DataFrame):
    st.caption("Replace values within a column — single substitution or bulk mapping.")

    col = st.selectbox("Column to replace values in", df.columns.tolist(), key="rv_col")
    if col is None:
        return

    s      = df[col]
    is_num = pd.api.types.is_numeric_dtype(s)
    is_text = s.dtype == object

    single_tab, bulk_tab = st.tabs(["🔁 Single Replace", "📋 Bulk Mapping"])

    # ── Tab 1: Single replace ──────────────────────────────────────────────
    with single_tab:
        # st.radio → st.tabs for scope
        whole_tab, substr_tab, regex_tab = st.tabs([
            "Whole cell value",
            "Substring / partial",
            "Regex pattern",
        ])

        with whole_tab:
            if st.button("Use: Whole cell", key="rv_scope_whole", use_container_width=True,
                         type="primary" if st.session_state.get("rv_scope_val", "Whole cell value") == "Whole cell value" else "secondary"):
                st.session_state["rv_scope_val"] = "Whole cell value"
                st.rerun()
            st.caption("Only replaces if the entire cell matches exactly.")

        with substr_tab:
            if st.button("Use: Substring", key="rv_scope_substr", use_container_width=True,
                         type="primary" if st.session_state.get("rv_scope_val", "Whole cell value") == "Substring / partial" else "secondary"):
                st.session_state["rv_scope_val"] = "Substring / partial"
                st.rerun()
            st.caption("Replaces any occurrence of the text within the cell.")

        with regex_tab:
            if st.button("Use: Regex", key="rv_scope_regex", use_container_width=True,
                         type="primary" if st.session_state.get("rv_scope_val", "Whole cell value") == "Regex pattern" else "secondary"):
                st.session_state["rv_scope_val"] = "Regex pattern"
                st.rerun()
            st.caption("Full regex substitution.")

        scope = st.session_state.get("rv_scope_val", "Whole cell value")
        if not is_text and scope != "Whole cell value":
            _msg("info", "Substring and Regex modes are only available for text columns.")
            scope = "Whole cell value"

        r1, r2 = st.columns(2)
        with r1:
            find_val = st.text_input(
                "Find",
                key="rv_find",
                placeholder=(
                    "e.g. -999" if is_num
                    else "e.g. (?i)n/?a" if scope == "Regex pattern"
                    else "e.g. _"
                ),
            )
        with r2:
            replace_val = st.text_input(
                "Replace with",
                key="rv_replace",
                placeholder="Leave blank to replace with null",
            )

        case_sensitive = True
        if is_text and scope in ("Substring / partial", "Regex pattern"):
            case_sensitive = st.checkbox("Case sensitive", value=False, key="rv_case")

        if not find_val and find_val != "0":
            st.caption("Enter a find value to preview matches.")
        else:
            try:
                flags = 0 if case_sensitive else re.IGNORECASE

                if scope == "Whole cell value":
                    if is_num:
                        try:    numeric_find = float(find_val); mask = s == numeric_find
                        except: mask = s.astype(str) == find_val
                    else:
                        mask = s.astype(str) == find_val

                    def _apply_single(series, find, rep):
                        if is_num:
                            try:    return series.replace(float(find), rep)
                            except: return series.astype(str).replace(find, rep)
                        return series.astype(str).replace(find, rep if not pd.isna(rep) else np.nan)

                elif scope == "Substring / partial":
                    mask = s.astype(str).str.contains(re.escape(find_val), flags=flags, na=False)

                    def _apply_single(series, find, rep):
                        rep_str = "" if pd.isna(rep) else str(rep)
                        return series.astype(str).str.replace(re.escape(find), rep_str, flags=flags, regex=True)

                elif scope == "Regex pattern":
                    mask = s.astype(str).str.contains(find_val, flags=flags, regex=True, na=False)

                    def _apply_single(series, find, rep):
                        rep_str = "" if pd.isna(rep) else str(rep)
                        return series.astype(str).str.replace(find, rep_str, flags=flags, regex=True)

                n_matches = int(mask.sum())

            except re.error as e:
                _msg("warn", "Invalid regex pattern.", detail=str(e))
                n_matches = 0
                mask = pd.Series([False] * len(df), index=df.index)
            except Exception as e:
                _msg("warn", "Could not build the match condition.", detail=str(e))
                n_matches = 0
                mask = pd.Series([False] * len(df), index=df.index)

            replace_with = np.nan if replace_val.strip() == "" else replace_val

            if n_matches == 0:
                _msg("info", f"No matches found in `{col}`.")
            else:
                st.markdown(f"**{n_matches:,} matching cell(s)** in `{col}`")
                matched_df   = df[mask][[col]].head(20).copy()
                after_series = _apply_single(matched_df[col], find_val, replace_with)
                preview_df   = pd.DataFrame({
                    "Row":    matched_df.index,
                    "Before": matched_df[col].astype(str),
                    "After":  after_series.astype(str),
                }).reset_index(drop=True)
                st.dataframe(preview_df, use_container_width=True, hide_index=True,
                             height=min(300, 40 + len(preview_df) * 36))
                if n_matches > 20:
                    st.caption(f"Showing first 20 of {n_matches:,} matches.")

                st.markdown("---")
                ra, rb = st.columns(2)
                with ra:
                    if st.button(f"✅ Apply to {n_matches:,} cell(s)", type="primary",
                                 use_container_width=True, key="rv_apply"):
                        new_df      = df.copy()
                        new_df[col] = _apply_single(new_df[col], find_val, replace_with)
                        if _apply_transform(
                            TransformResult(new_df), "col_restructuring", "replace",
                            {"col": col, "find": find_val,
                             "replace": str(replace_with), "scope": scope},
                            df,
                            f"Replace in '{col}': '{find_val}' → '{replace_with}' ({scope})",
                            OP_RENAME,
                        ):
                            st.rerun()
                with rb:
                    if st.button("Clear", use_container_width=True, key="rv_clear"):
                        for k in ("rv_find", "rv_replace"):
                            st.session_state.pop(k, None)
                        st.rerun()

    # ── Tab 2: Bulk mapping ────────────────────────────────────────────────
    with bulk_tab:
        st.caption(
            "Define multiple find → replace pairs at once. "
            "e.g. jan→01, feb→02, mar→03 or Y→True, N→False."
        )

        unique_vals = s.dropna().astype(str).unique().tolist()
        unique_vals.sort()

        # st.radio → st.tabs for seed mode
        manual_tab, seed_tab = st.tabs(["✏️ Manual entry", "🌱 Seed from column values"])

        with manual_tab:
            st.caption("Add find → replace pairs manually below.")

        with seed_tab:
            st.caption(f"`{col}` has **{len(unique_vals)}** unique value(s).")
            if len(unique_vals) > 200:
                _msg("warn", "More than 200 unique values — showing first 200.")
                unique_vals_display = unique_vals[:200]
            else:
                unique_vals_display = unique_vals

            seed_selection = st.multiselect(
                "Select values to create mappings for",
                options=unique_vals_display,
                default=[],
                key="bm_seed_select",
            )
            if st.button("Seed mapping rows ↓", key="bm_seed_btn", use_container_width=True):
                if "bm_pairs" not in st.session_state:
                    st.session_state.bm_pairs = []
                existing_finds = {p["find"] for p in st.session_state.bm_pairs}
                for v in seed_selection:
                    if v not in existing_finds:
                        st.session_state.bm_pairs.append({"find": v, "replace": ""})
                st.session_state.bm_pairs = [
                    p for p in st.session_state.bm_pairs if p["find"] != ""
                ]
                st.rerun()

        # ── Mapping table (shared below both tabs) ─────────────────────────
        if "bm_pairs" not in st.session_state:
            st.session_state.bm_pairs = [{"find": "", "replace": ""}]

        _space("sm")
        st.markdown("**Mapping table**")
        hc = st.columns([3, 3, 1])
        hc[0].markdown("**Find (exact)**")
        hc[1].markdown("**Replace with**")
        hc[2].markdown("**Del**")

        pairs     = st.session_state.bm_pairs
        to_delete = []

        for i, pair in enumerate(pairs):
            pc = st.columns([3, 3, 1])
            with pc[0]:
                pairs[i]["find"] = st.text_input(
                    f"find_{i}", value=pair["find"],
                    label_visibility="collapsed",
                    key=f"bm_find_{i}",
                    placeholder="e.g. jan",
                )
            with pc[1]:
                pairs[i]["replace"] = st.text_input(
                    f"replace_{i}", value=pair["replace"],
                    label_visibility="collapsed",
                    key=f"bm_replace_{i}",
                    placeholder="e.g. 01",
                )
            with pc[2]:
                if st.button("✕", key=f"bm_del_{i}", use_container_width=True):
                    to_delete.append(i)

        if to_delete:
            st.session_state.bm_pairs = [p for i, p in enumerate(pairs) if i not in to_delete]
            st.rerun()

        ba, bb = st.columns([1, 4])
        with ba:
            if st.button("＋ Add row", key="bm_add", use_container_width=True):
                st.session_state.bm_pairs.append({"find": "", "replace": ""})
                st.rerun()
        with bb:
            if st.button("🗑 Clear all rows", key="bm_clear_all", use_container_width=True):
                st.session_state.bm_pairs = [{"find": "", "replace": ""}]
                st.rerun()

        # ── Validate & preview ─────────────────────────────────────────────
        valid_pairs = [p for p in pairs if p["find"].strip() != ""]
        if not valid_pairs:
            st.caption("Add at least one find value to preview.")
        else:
            find_keys = [p["find"] for p in valid_pairs]
            dupes     = [k for k in find_keys if find_keys.count(k) > 1]
            if dupes:
                _msg("block", f"Duplicate find values: {list(set(dupes))} — each must be unique.")
            else:
                mapping = {
                    p["find"]: (np.nan if p["replace"].strip() == "" else p["replace"])
                    for p in valid_pairs
                }
                affected_mask = s.astype(str).isin(mapping.keys())
                n_affected    = int(affected_mask.sum())

                st.markdown("---")
                st.markdown(f"**Preview** — {n_affected:,} cell(s) will change")

                if n_affected == 0:
                    _msg("info", "None of the find values were found in this column.")
                else:
                    preview_sample = df[affected_mask][[col]].head(20).copy()
                    preview_sample["After"] = preview_sample[col].astype(str).map(
                        lambda v: str(mapping[v]) if v in mapping else v
                    )
                    preview_sample.columns = ["Before", "After"]
                    preview_sample.insert(0, "Row", preview_sample.index)
                    st.dataframe(
                        preview_sample.reset_index(drop=True),
                        use_container_width=True,
                        hide_index=True,
                        height=min(300, 40 + len(preview_sample) * 36),
                    )
                    if n_affected > 20:
                        st.caption(f"Showing first 20 of {n_affected:,} affected rows.")

                    unmapped = [v for v in unique_vals if v not in mapping]
                    if unmapped:
                        with st.expander(
                            f"ℹ️ {len(unmapped)} unique value(s) not in mapping (unchanged)",
                            expanded=False,
                        ):
                            st.caption(", ".join(str(v) for v in unmapped[:100]))

                    st.markdown("---")
                    bpa, bpb = st.columns(2)
                    with bpa:
                        if st.button(
                            f"✅ Apply {len(valid_pairs)} mapping(s) to {n_affected:,} cell(s)",
                            type="primary",
                            use_container_width=True,
                            key="bm_apply",
                        ):
                            new_df      = df.copy()
                            new_df[col] = new_df[col].astype(str).map(
                                lambda v: mapping[v] if v in mapping else v
                            )
                            if _apply_transform(
                                TransformResult(new_df), "col_restructuring", "bulk_replace",
                                {"col": col, "n_pairs": len(valid_pairs),
                                 "n_affected": n_affected},
                                df,
                                f"Bulk replace in '{col}': {len(valid_pairs)} mapping(s)",
                                OP_RENAME,
                            ):
                                st.session_state.bm_pairs = [{"find": "", "replace": ""}]
                                st.rerun()
                    with bpb:
                        if st.button("Clear mapping", use_container_width=True, key="bm_reset"):
                            st.session_state.bm_pairs = [{"find": "", "replace": ""}]
                            st.rerun()
# ═══════════════════════════════════════════════════════════════════════════
# RESHAPE INTELLIGENCE ENGINE (RIE)
# ═══════════════════════════════════════════════════════════════════════════

_MONTH_TOKENS = {
    "jan","feb","mar","apr","may","jun",
    "jul","aug","sep","oct","nov","dec",
    "january","february","march","april","june","july",
    "august","september","october","november","december",
}

_RE_YEAR          = re.compile(r'(.*?)[_\s]*(\d{4})$', re.IGNORECASE)
_RE_MONTH_YEAR    = re.compile(
    r'(.*?)[_\s]*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[_\s]*(\d{2,4})$',
    re.IGNORECASE,
)
_RE_QUARTER       = re.compile(r'(.*?)[_\s]*(q[1-4])[_\s]*(\d{2,4})$', re.IGNORECASE)
_RE_PURE_YEAR_COL = re.compile(r'^\d{4}$')

_TOTAL_PATTERN = re.compile(
    r'(?:^|[\s_\-/])total(?:$|[\s_\-/\d])'
    r'|^total$'
    r'|\btotal\b',
    re.IGNORECASE
)

_TOTAL_ROW_PATTERN = re.compile(
    r'^\s*(?:grand\s+)?(?:sub\s*)?totals?\s*$'
    r'|^\s*all\s+totals?\s*$'
    r'|^\s*sum\s*$'
    r'|^\s*overall\s*$'
    r'|\btotal\b'
    r'|\bsubtotal\b'
    r'|\bgrand\s+total\b',
    re.IGNORECASE
)

def _is_total_row_value(val) -> bool:
    """Return True if a cell value looks like a total/summary label."""
    if pd.isna(val):
        return False
    return bool(_TOTAL_ROW_PATTERN.search(str(val).strip()))

def _is_total_column(col_name: str) -> bool:
    """Return True if the column name looks like a total/summary column."""
    name = str(col_name).strip()
    return bool(_TOTAL_PATTERN.search(name))


def _detect_wide_groups(columns: List[str]) -> Dict[str, List[str]]:
    from collections import defaultdict
    groups: Dict[str, List[str]] = defaultdict(list)

    for col in columns:
        m = _RE_MONTH_YEAR.match(col)
        if m:
            groups[m.group(1).rstrip("_- ")].append(col)
            continue
        m = _RE_QUARTER.match(col)
        if m:
            groups[m.group(1).rstrip("_- ")].append(col)
            continue
        m = _RE_YEAR.match(col)
        if m:
            groups[m.group(1).rstrip("_- ")].append(col)
            continue
        if _RE_PURE_YEAR_COL.match(col):
            groups["__year__"].append(col)

    return {k: v for k, v in groups.items() if len(v) >= 3}


def _detect_measure_suffixes(columns: List[str]) -> Dict[str, int]:
    from collections import Counter
    suffix_counter: Counter = Counter()
    for col in columns:
        parts = col.split("_")
        if len(parts) >= 2:
            suffix_counter[parts[-1]] += 1
    return {k: v for k, v in suffix_counter.items() if v >= 3}


def _infer_id_columns(df: pd.DataFrame, value_cols: List[str]) -> List[str]:
    value_set = set(value_cols)
    id_candidates = []
    n = len(df)
    for col in df.columns:
        if col in value_set:
            continue
        if df[col].dtype in (object, "category"):
            id_candidates.append(col)
            continue
        if df[col].nunique() / max(n, 1) < 0.05:
            id_candidates.append(col)
    return id_candidates


def _suggest_split_config(value_cols: List[str]) -> Dict:
    if not value_cols:
        return {}

    sep_votes: Dict[str, int] = {}
    for sep in ("_", "-", " ", "."):
        votes = sum(1 for c in value_cols if sep in c)
        if votes:
            sep_votes[sep] = votes

    if not sep_votes:
        return {}

    best_sep = max(sep_votes, key=sep_votes.__getitem__)
    split_parts = [c.split(best_sep) for c in value_cols]
    part_counts  = [len(p) for p in split_parts]
    modal_parts  = max(set(part_counts), key=part_counts.count)

    if modal_parts < 2:
        return {}

    from collections import Counter
    suggested = []
    for i in range(modal_parts):
        tokens = [p[i] for p in split_parts if len(p) == modal_parts]
        top    = Counter(tokens).most_common(1)
        label  = top[0][0] if top else f"part{i+1}"
        if re.match(r'^\d+$', label):
            label = "period"
        suggested.append(label)

    return {
        "separator":       best_sep,
        "n_parts":         modal_parts,
        "suggested_names": suggested,
    }


def analyze_reshape(df: pd.DataFrame) -> Dict:
    cols   = df.columns.tolist()
    n_cols = len(cols)

    wide_groups      = _detect_wide_groups(cols)
    measure_suffixes = _detect_measure_suffixes(cols)

    n_wide_cols   = sum(len(v) for v in wide_groups.values())
    wide_ratio    = n_wide_cols / max(n_cols, 1)

    has_time_cols    = bool(wide_groups)
    has_measure_suf  = len(measure_suffixes) >= 2
    alternating_meas = has_time_cols and has_measure_suf

    if n_cols <= 6:
        structure_type = "long_format"
        confidence     = 0.75
    elif alternating_meas:
        structure_type = "wide_multi_measure"
        confidence     = min(0.95, 0.6 + wide_ratio * 0.4)
    elif has_time_cols and wide_ratio >= 0.4:
        structure_type = "wide_time_series"
        confidence     = min(0.95, 0.55 + wide_ratio * 0.45)
    elif wide_ratio >= 0.3:
        structure_type = "wide_mixed"
        confidence     = 0.55
    else:
        structure_type = "unknown"
        confidence     = 0.0

    recommendations = []

    if structure_type in ("wide_time_series", "wide_multi_measure", "wide_mixed"):
        if wide_groups:
            primary_prefix, primary_cols = max(
                wide_groups.items(), key=lambda x: len(x[1])
            )
        else:
            primary_prefix = ""
            primary_cols   = []

        all_value_cols = sorted(
            {c for cols_list in wide_groups.values() for c in cols_list}
        )
        id_cols = _infer_id_columns(df, all_value_cols)

        if primary_prefix and primary_prefix != "__year__":
            safe_prefix     = re.escape(primary_prefix)
            suggested_regex = f"^{safe_prefix}"
        elif primary_prefix == "__year__":
            suggested_regex = r"^\d{4}$"
        else:
            suggested_regex = ""

        split_cfg = _suggest_split_config(all_value_cols)

        melt_rec = {
            "action":     "melt",
            "confidence": round(confidence, 2),
            "reason":     (
                f"Detected {len(all_value_cols)} wide column(s) across "
                f"{len(wide_groups)} group(s)"
                + (f" with repeating measures: {list(measure_suffixes.keys())[:4]}"
                   if has_measure_suf else "")
            ),
            "prefill": {
                "id_vars":        id_cols,
                "value_vars":     all_value_cols,
                "regex":          suggested_regex,
                "var_name":       "period" if structure_type == "wide_time_series" else "variable",
                "value_name":     "value",
                "split_after":    bool(split_cfg),
                "split_sep":      split_cfg.get("separator", "_"),
                "split_names":    ",".join(split_cfg.get("suggested_names", [])),
            },
        }
        recommendations.append(melt_rec)

        if split_cfg:
            recommendations.append({
                "action":     "split_variable_column",
                "confidence": round(min(confidence * 0.95, 0.92), 2),
                "reason":     (
                    f"Column names contain '{split_cfg['separator']}' separator "
                    f"suggesting {split_cfg['n_parts']} dimensions"
                ),
                "prefill": {},
            })

    return {
        "structure_type":   structure_type,
        "confidence":       round(confidence, 2),
        "wide_groups":      wide_groups,
        "measure_suffixes": measure_suffixes,
        "recommendations":  recommendations,
    }


def _render_rie_panel(df: pd.DataFrame):
    col_hash  = hash(tuple(df.columns.tolist()))
    cache_key = f"rie_{df.shape}_{col_hash}"

    if st.session_state.get("_rie_cache_key") != cache_key:
        with st.spinner("Analysing dataset structure…"):
            analysis = analyze_reshape(df)
        st.session_state._rie_cache_key = cache_key
        st.session_state._rie_result    = analysis
    else:
        analysis = st.session_state._rie_result

    stype      = analysis["structure_type"]
    confidence = analysis["confidence"]
    recs       = analysis["recommendations"]

    if stype == "unknown" or confidence < 0.5:
        return

    label_map = {
        "wide_time_series":   ("📅 Wide time-series",   BRAND_PRIMARY),
        "wide_multi_measure": ("📊 Wide multi-measure",  "#e67e22"),
        "wide_mixed":         ("🔀 Wide mixed",          "#8e44ad"),
        "long_format":        ("✅ Already long format",  "#27ae60"),
    }
    label, colour = label_map.get(stype, ("❓ Unknown", "#aaa"))

    st.markdown(
        f"<div style='background:#f8f9ff;border:1px solid {colour}33;"
        f"border-left:4px solid {colour};border-radius:8px;"
        f"padding:12px 16px;margin-bottom:12px;'>"
        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:6px;'>"
        f"<span style='font-weight:700;font-size:0.9rem;'>🔍 Reshape Analysis</span>"
        f"<span style='background:{colour};color:white;font-size:0.72rem;"
        f"padding:2px 8px;border-radius:10px;'>{label}</span>"
        f"<span style='color:#888;font-size:0.78rem;'>confidence {confidence*100:.0f}%</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    wide_groups = analysis["wide_groups"]
    if wide_groups:
        group_lines = []
        for prefix, cols_list in list(wide_groups.items())[:4]:
            display = prefix if prefix != "__year__" else "(year columns)"
            group_lines.append(f"<b>{display}</b> — {len(cols_list)} columns")
        if len(wide_groups) > 4:
            group_lines.append(f"… and {len(wide_groups)-4} more groups")
        st.markdown(
            "<ul style='margin:4px 0 6px 24px;font-size:0.83rem;color:#444;'>"
            + "".join(f"<li>{g}</li>" for g in group_lines)
            + "</ul>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if recs:
        primary = recs[0]
        if primary["action"] == "melt":
            pf = primary["prefill"]
            st.markdown(
                f"<p style='font-size:0.84rem;color:#555;margin:4px 0 8px 0;'>"
                f"💡 {primary['reason']}</p>",
                unsafe_allow_html=True,
            )
            if st.button(
                "⚡ Pre-fill Melt configuration from analysis",
                key="rie_prefill_btn",
                use_container_width=True,
            ):
                st.session_state["melt_id"]          = pf["id_vars"]
                st.session_state["melt_val"]         = pf["value_vars"]
                st.session_state["melt_regex"]       = False
                st.session_state["melt_var"]         = pf["var_name"]
                st.session_state["melt_value"]       = pf["value_name"]
                st.session_state["melt_split"]       = pf["split_after"]
                st.session_state["melt_split_sep"]   = pf["split_sep"]
                st.session_state["melt_split_names"] = pf["split_names"]
                st.session_state["reshape_choice"]   = "Melt (Wide → Long)"
                st.rerun()

    st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
# MEASURE VALUE RENAMER
# ═══════════════════════════════════════════════════════════════════════════

def _needs_measure_normalisation(df: pd.DataFrame, split_names: List[str]) -> List[str]:
    problem_cols = []
    for col in split_names:
        if col not in df.columns:
            continue
        series      = df[col]
        has_nan     = series.isna().any()
        all_numeric = series.dropna().apply(
            lambda v: str(v).strip().lstrip("-").replace(".", "", 1).isdigit()
        ).all() if series.notna().any() else False
        if has_nan or all_numeric:
            problem_cols.append(col)
    return problem_cols


def _render_measure_renamer(df: pd.DataFrame, problem_cols: List[str]):
    st.markdown("---")
    st.markdown(
        "<div style='background:#fff8e1;border-left:4px solid #f39c12;"
        "border-radius:6px;padding:10px 14px;margin-bottom:8px;'>"
        "<b>⚠ Measure value normalisation needed</b><br>"
        "<span style='font-size:0.83rem;color:#555;'>"
        "One or more split columns contain missing or numeric-only values "
        "that would be silently dropped by a pivot. "
        "Rename them below before pivoting."
        "</span></div>",
        unsafe_allow_html=True,
    )

    for col in problem_cols:
        with st.expander(f"Rename values in **`{col}`**", expanded=True):
            series   = df[col].astype(object)
            raw_vals = series.unique().tolist()

            def _sort_key(v):
                return (0, "") if v is None or (isinstance(v, float) and pd.isna(v)) else (1, str(v))
            raw_vals.sort(key=_sort_key)

            rename_map: Dict = {}
            NAN_SENTINEL = "__NAN__"

            st.caption(f"{len(raw_vals)} unique value(s) detected")

            all_new_names = []
            valid = True

            for raw in raw_vals:
                is_nan  = raw is None or (isinstance(raw, float) and pd.isna(raw))
                display = "(empty / missing)" if is_nan else str(raw)

                if is_nan:
                    suggestion = "base"
                elif str(raw).strip().lstrip("-").replace(".", "", 1).isdigit():
                    suggestion = f"measure_{str(raw).strip()}"
                else:
                    suggestion = str(raw)

                wkey    = f"mrn_{col}_{display}"
                new_val = st.text_input(
                    f"`{display}` →",
                    value=st.session_state.get(wkey, suggestion),
                    key=wkey,
                )
                all_new_names.append(new_val)
                rename_map[NAN_SENTINEL if is_nan else raw] = new_val

            errors = []
            if any(n.strip() == "" for n in all_new_names):
                errors.append("Names cannot be empty")
            if len(set(all_new_names)) != len(all_new_names):
                errors.append("All renamed values must be unique")

            for err in errors:
                st.error(f"❌ {err}")

            if not errors:
                if st.button(
                    f"Apply rename to `{col}`",
                    type="primary",
                    use_container_width=True,
                    key=f"mrn_apply_{col}",
                ):
                    new_df   = df.copy()
                    nan_name = rename_map.get(NAN_SENTINEL)
                    if nan_name is not None:
                        new_df[col] = new_df[col].fillna(nan_name)
                    non_nan_map = {k: v for k, v in rename_map.items()
                                   if k != NAN_SENTINEL}
                    if non_nan_map:
                        new_df[col] = new_df[col].replace(non_nan_map)

                    result = TransformResult(new_df)
                    if _apply_transform(
                        result, "reshape", OP_MEASURE_RENAME,
                        {"col": col, "map": {str(k): v
                                             for k, v in rename_map.items()}},
                        df,
                        f"Rename measures in '{col}'",
                        OP_MEASURE_RENAME,
                    ):
                        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 6 — RESHAPE
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 6 — RESHAPE  (rewritten: radio → tabs, Skip → footer button)
# ═══════════════════════════════════════════════════════════════════════════

def render_reshape_stage():
    df = st.session_state.df
    _back_button("reshape")
    _stage_header("reshaped")

    # ── Reshape Intelligence Engine panel ──────────────────────────────────
    _render_rie_panel(df)

    locked = WorkflowFlags.is_set("dataset_locked")
    if locked:
        _msg("warn", "Dataset is locked. Unlock in the Utilities panel to make changes.")
        if st.button("Continue without reshaping →", type="primary",
                     use_container_width=True, key="skip_reshape_locked"):
            WorkflowState.done("reshaped")
            st.rerun()
        return

    # ── Tab selection ──────────────────────────────────────────────────────
    # RIE pre-fill sets reshape_tab_index to 1 (Melt) so the tab auto-switches.
    # We read it once, then clear it so it doesn't persist across reruns.
    _tab_index = st.session_state.pop("reshape_tab_index", 0)

    pivot_tab, melt_tab = st.tabs(["🔀 Pivot  (Long → Wide)", "📉 Melt  (Wide → Long)"])

    # We use a flag written by the pre-fill button inside _render_rie_panel
    # to know which tab the user intends. Tabs in Streamlit can't be
    # programmatically selected after render, so we render both and use
    # session state to pre-populate whichever one was requested.

    with pivot_tab:
        _render_pivot(df)

    with melt_tab:
        _render_melt(df)

        # Measure renamer only relevant after a melt — keep it inside this tab
        _df_now     = st.session_state.df
        _split_cols = st.session_state.get("_last_split_cols", [])
        if _split_cols:
            _problem = _needs_measure_normalisation(_df_now, _split_cols)
            if _problem:
                _render_measure_renamer(_df_now, _problem)

    # ── Footer: Skip / Continue ────────────────────────────────────────────
    # "Skip" is an action, not a content view — it belongs here as a button,
    # not as a tab. This is clearer UX: tabs = tools, button = decision.
    st.markdown("---")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown(
            f"<div style='background:{D['c_surface']};border:1px solid {D['c_border']};"
            f"border-radius:8px;padding:10px 14px;'>"
            f"<span style='font-size:{D['t_sm']};color:{D['c_text_muted']};'>"
            f"ℹ Use the tabs above to reshape, then click Continue. "
            f"If your data is already in the right shape, click Skip.</span></div>",
            unsafe_allow_html=True,
        )
    with sc2:
        cs1, cs2 = st.columns(2)
        with cs1:
            if st.button("Skip — data is already correct →",
                         use_container_width=True, key="skip_reshape"):
                st.session_state.pop("_last_split_cols", None)
                WorkflowState.done("reshaped")
                st.rerun()
        with cs2:
            if st.button("✅ Done reshaping →", type="primary",
                         use_container_width=True, key="done_reshape"):
                st.session_state.pop("_last_split_cols", None)
                WorkflowState.done("reshaped")
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# PIVOT RENDERER  (no structural changes — only label tightening)
# ═══════════════════════════════════════════════════════════════════════════

def _render_pivot(df: pd.DataFrame):
    all_cols     = df.columns.tolist()
    numeric_like = get_numeric_like_cols(df)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            f"<p style='font-size:{D['t_sm']};font-weight:600;color:{D['c_text_muted']};"
            f"text-transform:uppercase;letter-spacing:0.06em;margin:0 0 8px 0;'>"
            f"Configuration</p>",
            unsafe_allow_html=True,
        )

        # ── Index columns ──────────────────────────────────────────────────
        st.caption("Row labels (index)")
        _pa, _pb, _ = st.columns([1, 1, 3])
        with _pa:
            if st.button("All", key="piv_idx_all", use_container_width=True):
                st.session_state.piv_idx = all_cols
                st.rerun()
        with _pb:
            if st.button("None", key="piv_idx_none", use_container_width=True):
                st.session_state.piv_idx = []
                st.rerun()
        _stored_piv  = st.session_state.get("piv_idx", [])
        _piv_default = [v for v in _stored_piv if v in all_cols]
        index_cols   = st.multiselect(
            "Index columns", all_cols,
            default=_piv_default,
            key="piv_idx",
            label_visibility="collapsed",
        )

        # ── Column (header) columns ────────────────────────────────────────
        remaining = [c for c in all_cols if c not in index_cols]
        st.caption("Pivot headers — become new column names")
        _stored_col  = st.session_state.get("piv_col_multi", [])
        _col_default = [v for v in _stored_col if v in remaining]
        column_cols  = st.multiselect(
            "Header columns", remaining or all_cols,
            default=_col_default,
            key="piv_col_multi",
            label_visibility="collapsed",
        )

        # ── Value columns ──────────────────────────────────────────────────
        val_cands = [c for c in (numeric_like or all_cols)
                     if c not in index_cols and c not in column_cols]
        st.caption("Values to aggregate")
        _stored_val  = st.session_state.get("piv_val_multi", [])
        _val_default = [v for v in _stored_val if v in val_cands]
        value_cols   = st.multiselect(
            "Value columns", val_cands or all_cols,
            default=_val_default,
            key="piv_val_multi",
            label_visibility="collapsed",
        )

        # ── Agg + fill ─────────────────────────────────────────────────────
        aggfunc  = st.selectbox(
            "Aggregation function",
            ["sum", "mean", "count", "min", "max", "median"],
            key="piv_agg",
        )
        fill_str = st.text_input(
            "Fill empty cells with (leave blank to keep NaN)",
            value="",
            key="piv_fill",
        )

    with c2:
        st.markdown(
            f"<p style='font-size:{D['t_sm']};font-weight:600;color:{D['c_text_muted']};"
            f"text-transform:uppercase;letter-spacing:0.06em;margin:0 0 8px 0;'>"
            f"Live Preview</p>",
            unsafe_allow_html=True,
        )
        ready = index_cols and column_cols and value_cols
        if ready:
            sample = df.head(PIVOT_SAMPLE_SIZE)

            # Duplicate check
            dup_check  = (
                sample
                .groupby(index_cols + column_cols, observed=True)
                .size()
                .reset_index(name="_count")
            )
            collisions = dup_check[dup_check["_count"] > 1]
            if not collisions.empty:
                _msg("warn",
                     f"{len(collisions)} duplicate index/column combinations — "
                     f"{aggfunc}() will resolve them.")

            est_rows = sample[index_cols].drop_duplicates().shape[0]
            est_cols = sample[column_cols].drop_duplicates().shape[0] * len(value_cols)

            if est_rows * est_cols > MAX_PIVOT_CELLS:
                _msg("block",
                     f"Estimated output ({est_rows:,} × {est_cols:,}) exceeds "
                     f"{MAX_PIVOT_CELLS:,} cells. Reduce dimensions.")
            else:
                try:
                    prev = pd.pivot_table(
                        sample,
                        index=index_cols,
                        columns=column_cols,
                        values=value_cols,
                        aggfunc=aggfunc,
                    ).reset_index()
                    if fill_str.strip():
                        try:    prev = prev.fillna(float(fill_str))
                        except: prev = prev.fillna(fill_str)
                    if isinstance(prev.columns, pd.MultiIndex):
                        prev.columns = [
                            "_".join(str(x) for x in col if str(x)).strip("_")
                            for col in prev.columns
                        ]
                    st.caption(f"Output shape: {prev.shape[0]:,} rows × {prev.shape[1]:,} cols")
                    st.dataframe(prev.head(20), height=300,
                                 use_container_width=True, hide_index=True)
                except Exception as e:
                    _msg("warn",
                         "Preview failed with current settings.",
                         detail=str(e))
        else:
            st.markdown(
                f"<div style='background:{D['c_surface']};border:1px solid {D['c_border']};"
                f"border-radius:8px;padding:40px 20px;text-align:center;color:{D['c_text_muted']};'>"
                f"Select index, header, and value columns to see a live preview"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Apply button — full width below both columns ───────────────────────
    if ready:
        if st.button("Apply Pivot →", type="primary",
                     use_container_width=True, key="apply_pivot"):
            r = TransformationEngine.pivot_table(
                df, index_cols, column_cols, value_cols, aggfunc, fill_str,
            )
            if _apply_transform(
                r, "reshape", OP_PIVOT,
                {"index": index_cols, "columns": column_cols,
                 "values": value_cols, "agg": aggfunc},
                df,
                f"Pivot: {aggfunc}({', '.join(value_cols)})",
                OP_PIVOT,
            ):
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# MELT RENDERER  (pre-fill integration + measure renamer moved here)
# ═══════════════════════════════════════════════════════════════════════════

def _render_melt(df: pd.DataFrame):
    all_cols = df.columns.tolist()

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            f"<p style='font-size:{D['t_sm']};font-weight:600;color:{D['c_text_muted']};"
            f"text-transform:uppercase;letter-spacing:0.06em;margin:0 0 8px 0;'>"
            f"Configuration</p>",
            unsafe_allow_html=True,
        )

        # ── ID variables ───────────────────────────────────────────────────
        _stored_id  = st.session_state.get("melt_id", [])
        _id_default = [v for v in _stored_id if v in all_cols]
        id_vars = st.multiselect(
            "ID columns — kept as-is (not unpivoted)",
            all_cols,
            default=_id_default,
            key="melt_id",
        )

        # ── Value variables ────────────────────────────────────────────────
        val_cands = [c for c in all_cols if c not in id_vars]

        use_regex = st.checkbox(
            "Select value columns by regex pattern",
            key="melt_regex",
            help="Useful when you have many columns matching a pattern like ^Sales_ or \\d{4}",
        )

        if use_regex:
            pattern = st.text_input(
                "Regex pattern",
                key="melt_pattern",
                placeholder="e.g.  ^Sales_  or  \\d{4}",
            )
            try:
                value_vars = [c for c in val_cands if re.search(pattern, c)] if pattern else []
                st.caption(f"{len(value_vars)} column(s) matched")
            except re.error as e:
                _msg("warn", "Invalid regex — check your syntax.", detail=str(e))
                value_vars = []
        else:
            _ma, _mb, _ = st.columns([1, 1, 3])
            with _ma:
                if st.button("All", key="melt_val_all", use_container_width=True):
                    st.session_state.melt_val = val_cands
                    st.rerun()
            with _mb:
                if st.button("None", key="melt_val_none", use_container_width=True):
                    st.session_state.melt_val = []
                    st.rerun()
            _stored_melt  = st.session_state.get("melt_val", [])
            _melt_default = [v for v in _stored_melt if v in val_cands]
            if not _melt_default:
                _melt_default = val_cands[:3]
            value_vars = st.multiselect(
                "Value columns to unpivot",
                val_cands,
                default=_melt_default,
                key="melt_val",
            )

        # ── Output column names ────────────────────────────────────────────
        st.caption("Output column names")
        nc1, nc2 = st.columns(2)
        with nc1:
            _var_default = st.session_state.get("melt_var", "variable")
            var_name = st.text_input(
                "Variable column",
                _var_default,
                key="melt_var",
                help="This column will hold the original column names",
            )
        with nc2:
            _value_default = st.session_state.get("melt_value", "value")
            value_name = st.text_input(
                "Value column",
                _value_default,
                key="melt_value",
                help="This column will hold the values",
            )

        # ── Clash detection ────────────────────────────────────────────────
        id_var_set = set(id_vars)

        def _safe_name(name: str, taken: set, suffix: str):
            if name not in taken:
                return name, False
            candidate = f"{name}_{suffix}"
            i = 2
            while candidate in taken:
                candidate = f"{name}_{suffix}{i}"
                i += 1
            return candidate, True

        safe_var,   var_changed   = _safe_name(var_name,   id_var_set, "var")
        safe_value, value_changed = _safe_name(value_name, id_var_set | {safe_var}, "val")

        if var_changed:
            _msg("warn", f"'{var_name}' clashes with an ID column — will use '{safe_var}'")
        if value_changed:
            _msg("warn", f"'{value_name}' clashes with an existing name — will use '{safe_value}'")

        name_clash = []
        if safe_var == safe_value:
            name_clash.append("Variable name and value name must be different")
        for msg in name_clash:
            _msg("block", msg)

        var_name   = safe_var
        value_name = safe_value

        # ── Split after melt ───────────────────────────────────────────────
        _split_default     = st.session_state.get("melt_split", False)
        _split_sep_default = st.session_state.get("melt_split_sep", "_")
        _split_nam_default = st.session_state.get("melt_split_names", "part1,part2")

        split_after = st.checkbox(
            "Split variable column after melt",
            value=_split_default,
            key="melt_split",
            help="Useful when column names encode multiple dimensions, e.g. Sales_2023_Q1",
        )
        split_sep, split_names_list = "_", []
        if split_after:
            ss1, ss2 = st.columns(2)
            with ss1:
                split_sep = st.text_input(
                    "Separator", _split_sep_default, key="melt_split_sep"
                )
            with ss2:
                split_raw = st.text_input(
                    "New column names (comma-separated)",
                    _split_nam_default,
                    key="melt_split_names",
                )
            split_names_list = [s.strip() for s in split_raw.split(",") if s.strip()]

    # ── Preview ────────────────────────────────────────────────────────────
    with c2:
        st.markdown(
            f"<p style='font-size:{D['t_sm']};font-weight:600;color:{D['c_text_muted']};"
            f"text-transform:uppercase;letter-spacing:0.06em;margin:0 0 8px 0;'>"
            f"Live Preview</p>",
            unsafe_allow_html=True,
        )
        ok = value_vars and not name_clash
        if ok:
            try:
                prev = pd.melt(
                    df.head(PIVOT_SAMPLE_SIZE),
                    id_vars=id_vars or None,
                    value_vars=value_vars,
                    var_name=var_name,
                    value_name=value_name,
                )
                if split_after and split_names_list:
                    split_df = prev[var_name].str.split(split_sep, expand=True)
                    for i, name in enumerate(split_names_list):
                        if i in split_df.columns:
                            prev[name] = split_df[i]
                st.caption(f"Output shape: {prev.shape[0]:,} rows × {prev.shape[1]:,} cols")
                st.dataframe(prev.head(20), height=300,
                             use_container_width=True, hide_index=True)
            except Exception as e:
                _msg("warn", "Preview failed with current settings.", detail=str(e))
        else:
            st.markdown(
                f"<div style='background:{D['c_surface']};border:1px solid {D['c_border']};"
                f"border-radius:8px;padding:40px 20px;text-align:center;color:{D['c_text_muted']};'>"
                f"Select value columns to see a live preview"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Apply button — full width ──────────────────────────────────────────
    if ok:
        if st.button("Apply Melt →", type="primary",
                     use_container_width=True, key="apply_melt"):
            r = TransformationEngine.melt(
                df, id_vars, value_vars, var_name, value_name,
                split_after=split_after,
                split_sep=split_sep,
                split_names=split_names_list if split_after else None,
            )
            if _apply_transform(
                r, "reshape", OP_MELT,
                {"id_vars": id_vars, "value_vars": len(value_vars),
                 "var_name": var_name, "split": split_after},
                df,
                f"Melt: {len(value_vars)} cols → long",
                OP_MELT,
            ):
                st.session_state["_last_split_cols"] = (
                    split_names_list if split_after else []
                )
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# RIE PANEL  — one change: pre-fill now sets reshape_tab_index = 1
# so the Melt tab is visually active on next render
# ═══════════════════════════════════════════════════════════════════════════

def _render_rie_panel(df: pd.DataFrame):
    col_hash  = hash(tuple(df.columns.tolist()))
    cache_key = f"rie_{df.shape}_{col_hash}"

    if st.session_state.get("_rie_cache_key") != cache_key:
        with st.spinner("Analysing dataset structure…"):
            analysis = analyze_reshape(df)
        st.session_state._rie_cache_key = cache_key
        st.session_state._rie_result    = analysis
    else:
        analysis = st.session_state._rie_result

    stype      = analysis["structure_type"]
    confidence = analysis["confidence"]
    recs       = analysis["recommendations"]

    if stype == "unknown" or confidence < 0.5:
        return

    label_map = {
        "wide_time_series":   ("📅 Wide time-series",    BRAND_PRIMARY),
        "wide_multi_measure": ("📊 Wide multi-measure",   "#e67e22"),
        "wide_mixed":         ("🔀 Wide mixed",           "#8e44ad"),
        "long_format":        ("✅ Already long format",   "#27ae60"),
    }
    label, colour = label_map.get(stype, ("❓ Unknown", "#aaa"))

    st.markdown(
        f"<div style='background:#f8f9ff;border:1px solid {colour}33;"
        f"border-left:4px solid {colour};border-radius:8px;"
        f"padding:12px 16px;margin-bottom:12px;'>"
        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:6px;'>"
        f"<span style='font-weight:700;font-size:0.9rem;'>🔍 Reshape Intelligence</span>"
        f"<span style='background:{colour};color:white;font-size:0.72rem;"
        f"padding:2px 8px;border-radius:10px;'>{label}</span>"
        f"<span style='color:#888;font-size:0.78rem;'>confidence {confidence*100:.0f}%</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    wide_groups = analysis["wide_groups"]
    if wide_groups:
        group_lines = []
        for prefix, cols_list in list(wide_groups.items())[:4]:
            display = prefix if prefix != "__year__" else "(year columns)"
            group_lines.append(f"<b>{display}</b> — {len(cols_list)} columns")
        if len(wide_groups) > 4:
            group_lines.append(f"… and {len(wide_groups)-4} more groups")
        st.markdown(
            "<ul style='margin:4px 0 6px 24px;font-size:0.83rem;color:#444;'>"
            + "".join(f"<li>{g}</li>" for g in group_lines)
            + "</ul>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if recs:
        primary = recs[0]
        if primary["action"] == "melt":
            pf = primary["prefill"]
            st.markdown(
                f"<p style='font-size:0.84rem;color:#555;margin:4px 0 8px 0;'>"
                f"💡 {primary['reason']}</p>",
                unsafe_allow_html=True,
            )
            if st.button(
                "⚡ Pre-fill Melt tab from analysis",
                key="rie_prefill_btn",
                use_container_width=True,
            ):
                # Pre-populate all melt session state keys
                st.session_state["melt_id"]          = pf["id_vars"]
                st.session_state["melt_val"]         = pf["value_vars"]
                st.session_state["melt_regex"]       = False
                st.session_state["melt_var"]         = pf["var_name"]
                st.session_state["melt_value"]       = pf["value_name"]
                st.session_state["melt_split"]       = pf["split_after"]
                st.session_state["melt_split_sep"]   = pf["split_sep"]
                st.session_state["melt_split_names"] = pf["split_names"]
                # No longer sets reshape_choice — tabs don't need it
                st.rerun()

    st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 7 — POST-TRANSFORM AUDIT
# ═══════════════════════════════════════════════════════════════════════════

def render_audit_stage():
    df   = st.session_state.df
    snap = CheckpointSnapshot.get()

#    _back_button("audit")
    _stage_header("audited")

    if snap:
        row_d  = len(df) - snap["row_count"]
        col_d  = len(df.columns) - snap["col_count"]
        null_d = df.isnull().mean().mean() * 100 - snap["null_pct"]
        mem_d  = (df.memory_usage(deep=True).sum() - snap["memory_bytes"]) / (1024**2)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Row Δ",    f"{row_d:+,}")
        c2.metric("Column Δ", f"{col_d:+}")
        c3.metric("Null % Δ", f"{null_d:+.1f}%")
        c4.metric("Memory Δ", f"{mem_d:+.1f} MB")

        if snap["row_count"] and row_d > snap["row_count"] * 0.5:
            _msg("block", "Row count grew by more than 50% since the checkpoint — this may indicate a Cartesian join or incorrect pivot keys.")

        old_cols = {c["name"] for c in snap["col_schema"]}
        added    = set(df.columns) - old_cols
        removed  = old_cols - set(df.columns)
        if added:   st.info(f"➕ New columns: {', '.join(sorted(added))}")
        if removed: st.warning(f"➖ Removed: {', '.join(sorted(removed))}")

        q    = QuarantineManager.count_pending()
        drop = max(0, snap["row_count"] - len(df) - q)
        st.markdown("**Row accounting**")
        ra1, ra2, ra3, ra4 = st.columns(4)
        ra1.metric("Checkpoint", f"{snap['row_count']:,}")
        ra2.metric("Main",       f"{len(df):,}")
        ra3.metric("Quarantine", f"{q:,}")
        ra4.metric("Dropped",    f"{drop:,}")
    else:
        _msg("info", "No checkpoint was taken — the row integrity step was skipped.")

    q_count = QuarantineManager.count_pending()
    if q_count > 0:
        st.markdown("---")
        st.markdown(f"**Quarantine ({q_count} pending rows)**")
        st.caption("These rows failed a transformation. Resolve to enable export.")
        st.dataframe(QuarantineManager.as_dataframe(),
                     use_container_width=True, height=200, hide_index=True)

        qa, qb = st.columns(2)
        with qa:
            if st.button("Drop all quarantined rows",
                         use_container_width=True, key="q_drop"):
                ids = [r["id"] for r in QuarantineManager.get_pending()]
                QuarantineManager.resolve(ids, "dropped")
                st.rerun()
        with qb:
            if st.button("Force-include all (override)",
                         use_container_width=True, key="q_force"):
                ids = [r["id"] for r in QuarantineManager.get_pending()]
                QuarantineManager.resolve(ids, "force_included")
                forced = [r["row_data"] for r in QuarantineManager._store()
                          if r["resolution"] == "force_included"]
                if forced:
                    new_df = pd.concat([df, pd.DataFrame(forced)], ignore_index=True)
                    HistoryManager.push(new_df, "Force-include quarantine rows", OP_IMPORT)
                st.rerun()

    st.markdown("---")
    can_export = not QuarantineManager.export_blocked()
    if not can_export:
        st.warning("Export blocked — resolve quarantine above first")

    if st.button(
        "✅ Continue to Export →",
        type="primary",
        use_container_width=True,
        key="audit_done",
        disabled=not can_export):

        WorkflowState.done("audited")
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# FINAL — EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def render_export_stage():
    df = st.session_state.df
    _section_header("✅ Export")
    _msg("success", "Dataset preparation complete — your data is ready to download.")

    st.markdown(f"<div style='height:{D['sp_sm']};'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "💾 Download CSV",
            df.to_csv(index=False),
            "dataset_final.csv",
            "text/csv",
            use_container_width=True,
            type="primary",
        )
    with c2:
        try:
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            st.download_button(
                "💾 Download Parquet",
                buf.getvalue(),
                "dataset_final.parquet",
                use_container_width=True,
            )
        except Exception as e:
            _msg("warn", "Parquet export unavailable for this dataset.", detail=str(e))
    with c3:
        if st.button("🔄 Start over", use_container_width=True):
            _full_reset()

    st.markdown("---")
    _metric_strip([
        ("Rows",    f"{len(df):,}",                                         ""),
        ("Columns", str(len(df.columns)),                                   ""),
        ("Size",    f"{df.memory_usage(deep=True).sum()/(1024**2):.1f} MB", ""),
    ])

    with st.expander("Preview final dataset"):
        st.dataframe(df.head(50), use_container_width=True, hide_index=True)

    with st.expander("Operation log"):
        log = OperationLog.get_all()
        if log:
            ldf = pd.DataFrame([{
                "time":       e["timestamp"],
                "operation":  e["operation"],
                "params":     str(e["parameters"]),
                "rows Δ":     f"{e['row_delta']:+,}",
                "cols Δ":     f"{e['col_delta']:+,}",
                "quarantine": e["quarantine_count"],
                "status":     e["status"],
            } for e in log])
            st.dataframe(ldf, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def render_dataset_manager():
    WorkflowState.init()
    WorkflowFlags._store()
    OperationLog._log()
    QuarantineManager._store()

    for k in ("df", "raw_df"):
        if k not in st.session_state:
            st.session_state[k] = None
    if "history_index" not in st.session_state:
        st.session_state.history_index = -1

    st.markdown(
        f"<div style='background:linear-gradient({BRAND_GRADIENT});"
        f"padding:1.2rem 2rem;border-radius:10px;margin-bottom:{D['sp_md']};'>"
        f"<div style='display:flex;align-items:center;justify-content:space-between;'>"
        f"<div>"
        f"<h1 style='color:white;margin:0;font-size:1.6rem;letter-spacing:-0.02em;'>ANALYTIQAL</h1>"
        f"<p style='color:rgba(255,255,255,0.75);margin:2px 0 0;font-size:0.8rem;'>"
        f"Dataset Manager &nbsp;·&nbsp; v1.0"
        f"</p></div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )
    _progress_bar()

    if WorkflowState.is_done("imported") and st.session_state.get("df") is not None:
        st.markdown(f"<div style='height:{D['sp_sm']};'></div>", unsafe_allow_html=True)
        _render_global_controls()
        st.markdown(f"<div style='height:{D['sp_sm']};'></div>", unsafe_allow_html=True)

    def _reset_to_load():
        for s in WORKFLOW_STAGES:
            st.session_state.workflow[s] = False
        for k in ("full_df", "preview_df", "raw_bytes", "raw_name", "raw_fmt"):
            st.session_state.pop(k, None)
        st.rerun()

    if (WorkflowState.is_done("loaded")
            and not WorkflowState.is_done("imported")
            and "full_df" not in st.session_state):
        _reset_to_load()
        return

    if WorkflowState.is_done("imported") and st.session_state.df is None:
        _reset_to_load()
        return

    if not WorkflowState.is_done("loaded"):
        render_load_stage()
        return

    if not WorkflowState.is_done("schema_done"):
        render_schema_stage()
        return

    if not WorkflowState.is_done("imported"):
        render_import_stage()
        return

    render_structural_overview()

    if not WorkflowState.is_done("row_integrity"):
        render_row_integrity_stage()
        return

    if not WorkflowState.is_done("reshaped"):
        render_reshape_stage()
        return

    if not WorkflowState.is_done("col_restructured"):
        render_column_restructuring_stage()
        return

    if not WorkflowState.is_done("audited"):
        render_audit_stage()
        return

    render_export_stage()


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    st.set_page_config(
        page_title="Analytiqal — Dataset Manager",
        page_icon="📊",
        layout="wide",
    )
    render_dataset_manager()

