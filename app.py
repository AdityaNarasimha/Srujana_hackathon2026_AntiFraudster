"""
AntiFraudster — Financial Fraud-Risk Detection & Transaction Investigation Dashboard
====================================================================================
A decision-support and investigation prototype for reviewing suspicious financial
transactions, behavioral/profile signals, location/network anomalies, and separate
experimental machine-learning evaluations.

IMPORTANT DESIGN & SAFETY NOTES:
- Consumes saved pipeline outputs from `data/antifraudster_results.csv` and
  separate experimental model outputs from `data/ml_evaluation_results.csv`.
- Uses both `st.markdown(..., unsafe_allow_html=True)` and
  `streamlit.components.v1.html(...)` (`components.html`) so interactive HTML/CSS/JS
  templates (live client-side card inspection, instant search, copy-to-clipboard,
  and signal toggles) work out of the box alongside native Streamlit controls.
- Preserves existing fraud-detection modules under `src/` without modifying
  underlying scoring or flagging algorithms.
- Never fabricates transactions, KPIs, SHAP values, or model predictions.
- Clearly distinguishes dataset ground-truth labels (`is_fraud` / `actual_fraud`),
  operational risk levels (`combined_risk_level` / `risk_level`), human-review flags
  (`needs_human_review`), and experimental ML predictions (`predicted_fraud`).
"""

from __future__ import annotations

import html
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - fallback for headless unit testing
    go = None  # type: ignore

try:
    import streamlit as st
    import streamlit.components.v1 as components
except ImportError:  # pragma: no cover - fallback for headless unit testing
    st = None  # type: ignore
    components = None  # type: ignore


# =============================================================================
# 1. PROJECT PATHS & EXPECTED SCHEMAS
# =============================================================================

DEFAULT_LOCAL_REPO_PATH = Path("/home/jagadishdn/Srujana_hackathon2026_AntiFraudster")

OPERATIONAL_EXPECTED_COLUMNS: List[str] = [
    "transaction_id",
    "timestamp",
    "sender_id",
    "receiver_id",
    "amount",
    "location_id",
    "is_fraud",
    "scenario",
    "risk_score",
    "risk_level",
    "reasons",
    "rapid_transfer_flag",
    "new_recipient_flag",
    "behavior_reasons",
    "customer_type",
    "expected_monthly_outflow",
    "usual_transaction_count",
    "kyc_status",
    "profile_available",
    "outflow_ratio",
    "profile_risk_points",
    "profile_review_reason",
    "profile_review_required",
    "behavior_risk_points",
    "combined_risk_score",
    "combined_risk_level",
    "location_anomaly_flag",
    "location_distance_km",
    "location_elapsed_minutes",
    "location_reason",
    "network_type",
    "vpn_detected",
    "proxy_detected",
    "tor_detected",
    "hosting_network_detected",
    "network_review_signal",
    "network_reason",
    "needs_human_review",
    "additional_review_reasons",
]

ML_EXPECTED_COLUMNS: List[str] = [
    "transaction_id",
    "timestamp",
    "sender_id",
    "receiver_id",
    "amount",
    "location_id",
    "is_fraud",
    "scenario",
    "hour",
    "day_of_week",
    "is_night",
    "sender_history_count",
    "sender_history_avg_amount",
    "sender_history_max_amount",
    "seconds_since_previous",
    "known_recipient",
    "amount_to_history_ratio",
    "amount_above_history_max",
    "actual_fraud",
    "predicted_fraud",
    "fraud_probability",
]

AUXILIARY_DATA_FILES: Dict[str, str] = {
    "sample_transactions.csv": "Input synthetic transaction feed consumed by the operational pipeline.",
    "sample_customers.csv": "Customer KYC & baseline monthly outflow profiles used for profile-risk checks.",
    "evaluation_results.csv": "Rule/pipeline evaluation output dataset (kept separate from live queue).",
    "experimental_transactions.csv": "Experimental feature-engineered transaction set for ML training/testing.",
}

# Semantic risk styling tokens (Section 5)
RISK_SEMANTICS: Dict[str, Dict[str, str]] = {
    "CRITICAL": {
        "label": "CRITICAL",
        "bg": "#FEF2F2",
        "text": "#B91C1C",
        "border": "#FCA5A5",
        "accent": "#DC2626",
        "dot": "#DC2626",
        "rank": "4",
    },
    "HIGH": {
        "label": "HIGH",
        "bg": "#FFF7ED",
        "text": "#C2410C",
        "border": "#FDBA74",
        "accent": "#EA580C",
        "dot": "#EA580C",
        "rank": "3",
    },
    "MEDIUM": {
        "label": "MEDIUM",
        "bg": "#FEFCE8",
        "text": "#A16207",
        "border": "#FDE047",
        "accent": "#D97706",
        "dot": "#D97706",
        "rank": "2",
    },
    "LOW": {
        "label": "LOW",
        "bg": "#F0FDF4",
        "text": "#15803D",
        "border": "#86EFAC",
        "accent": "#16A34A",
        "dot": "#16A34A",
        "rank": "1",
    },
    "UNKNOWN": {
        "label": "UNKNOWN",
        "bg": "#F3F4F6",
        "text": "#4B5563",
        "border": "#D1D5DB",
        "accent": "#6B7280",
        "dot": "#6B7280",
        "rank": "0",
    },
    "INFO": {
        "label": "INFORMATIONAL",
        "bg": "#EFF6FF",
        "text": "#1D4ED8",
        "border": "#BFDBFE",
        "accent": "#2563EB",
        "dot": "#2563EB",
        "rank": "0",
    },
}

RISK_ORDER: List[str] = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]


def resolve_project_root() -> Path:
    """Resolve the AntiFraudster project root directory safely."""
    env_root = os.environ.get("ANTIFRAUDSTER_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if candidate.exists():
            return candidate

    script_dir = Path(__file__).resolve().parent
    if (script_dir / "data").exists() or (script_dir / "src").exists():
        return script_dir

    cwd = Path.cwd()
    if (cwd / "data").exists() or (cwd / "src").exists():
        return cwd

    if DEFAULT_LOCAL_REPO_PATH.exists():
        return DEFAULT_LOCAL_REPO_PATH

    return script_dir


# =============================================================================
# 2. SAFE DATA PARSING & NORMALIZATION UTILITIES
# =============================================================================

def parse_bool_value(val: Any) -> bool:
    """Safely parse boolean flags stored as bool, int, float, or string in CSVs."""
    if val is None:
        return False
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, float, np.number)):
        if pd.isna(val):
            return False
        return float(val) != 0.0
    if isinstance(val, str):
        cleaned = val.strip().lower()
        return cleaned in {"true", "1", "1.0", "yes", "y", "t"}
    return False


def parse_bool_series(series: pd.Series) -> pd.Series:
    """Vectorized-safe boolean series parser that handles string CSV representations."""
    if series is None or series.empty:
        return pd.Series(dtype=bool)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.map(parse_bool_value).astype(bool)


def normalize_risk_level(raw_value: Any) -> Tuple[str, str]:
    """
    Map a raw risk-level value to its semantic key and display label without
    fabricating 'GOOD' for unflagged transactions.
    Returns (semantic_key, display_label).
    """
    if raw_value is None or (isinstance(raw_value, float) and math.isnan(raw_value)):
        return ("UNKNOWN", "Not available")
    text = str(raw_value).strip()
    if not text or text.lower() in {"nan", "none", "null", ""}:
        return ("UNKNOWN", "Not available")

    upper = text.upper()
    if upper in {"CRITICAL", "CRIT", "VERY HIGH", "SEVERE"}:
        return ("CRITICAL", text if upper == "CRITICAL" else f"{text} (CRITICAL)")
    if upper in {"HIGH"}:
        return ("HIGH", "HIGH")
    if upper in {"MEDIUM", "MODERATE", "MED"}:
        return ("MEDIUM", text if upper == "MEDIUM" else f"{text} (MEDIUM)")
    if upper in {"LOW", "MINIMAL"}:
        # Preserve actual LOW risk level rather than relabeling as GOOD
        return ("LOW", text if upper == "LOW" else f"{text} (LOW)")
    return ("UNKNOWN", text)


def format_inr(amount: Any, include_decimals: bool = True) -> str:
    """
    Format a numeric value in Indian Rupees (₹) using standard Indian digit grouping
    (e.g., ₹12,34,567.89). Handles missing or non-numeric values safely.
    """
    if amount is None or pd.isna(amount):
        return "Not available"
    try:
        val = float(amount)
    except (ValueError, TypeError):
        return "Not available"

    sign = "-" if val < 0 else ""
    val = abs(val)
    if include_decimals:
        whole_str, dec_str = f"{val:.2f}".split(".")
    else:
        whole_str, dec_str = f"{int(round(val))}", ""

    if len(whole_str) > 3:
        last_three = whole_str[-3:]
        remaining = whole_str[:-3]
        groups = []
        while len(remaining) > 2:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.insert(0, remaining)
        formatted_whole = ",".join(groups) + "," + last_three
    else:
        formatted_whole = whole_str

    if include_decimals:
        return f"{sign}₹{formatted_whole}.{dec_str}"
    return f"{sign}₹{formatted_whole}"


def format_timestamp(ts_val: Any) -> str:
    """Safely format a timestamp value for human-readable display."""
    if ts_val is None or pd.isna(ts_val):
        return "Not available"
    text = str(ts_val).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return "Not available"
    try:
        parsed = pd.to_datetime(text, errors="raise")
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return text


def parse_explanation_items(raw_val: Any) -> List[str]:
    """
    Convert recorded explanation strings (from `reasons`, `behavior_reasons`,
    `profile_review_reason`, `location_reason`, `network_reason`,
    `additional_review_reasons`) into clean, human-readable bullet strings.
    Never dumps raw JSON or Python list syntax.
    """
    if raw_val is None or pd.isna(raw_val):
        return []
    if isinstance(raw_val, (list, tuple, set)):
        items = [str(x).strip() for x in raw_val if str(x).strip()]
        return [i for i in items if i.lower() not in {"none", "nan", "null", "n/a", "[]"}]

    text = str(raw_val).strip()
    if not text or text.lower() in {"none", "nan", "null", "n/a", "[]", "{}", "no reasons", "normal"}:
        return []

    # Strip outer list brackets if stored as stringified Python list
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        parts = re.split(r"""['"]\s*,\s*['"]|;\s*|\|\s*""", inner)
        cleaned = [p.strip(" '\"[]") for p in parts]
        return [c for c in cleaned if c and c.lower() not in {"none", "nan", "null"}]

    # Split on semicolons or pipes first; fallback to single statement
    if ";" in text:
        parts = [p.strip() for p in text.split(";")]
    elif "|" in text:
        parts = [p.strip() for p in text.split("|")]
    else:
        parts = [text]

    return [p for p in parts if p and p.lower() not in {"none", "nan", "null", "n/a"}]


def prepare_operational_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize operational pipeline results without altering any underlying
    fraud detection logic or inventing columns. Adds internal `_`-prefixed
    helper columns strictly for safe sorting, filtering, and display.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # Numeric amount
    if "amount" in out.columns:
        out["_amount_num"] = pd.to_numeric(out["amount"], errors="coerce").fillna(0.0)
    else:
        out["_amount_num"] = 0.0

    # Effective risk score (prefer combined_risk_score, fallback to risk_score)
    if "combined_risk_score" in out.columns:
        out["_risk_score_num"] = pd.to_numeric(out["combined_risk_score"], errors="coerce")
        if "risk_score" in out.columns:
            out["_risk_score_num"] = out["_risk_score_num"].fillna(
                pd.to_numeric(out["risk_score"], errors="coerce")
            )
        out["_risk_score_num"] = out["_risk_score_num"].fillna(0.0)
        out["_risk_score_col"] = "combined_risk_score"
    elif "risk_score" in out.columns:
        out["_risk_score_num"] = pd.to_numeric(out["risk_score"], errors="coerce").fillna(0.0)
        out["_risk_score_col"] = "risk_score"
    else:
        out["_risk_score_num"] = 0.0
        out["_risk_score_col"] = "Unavailable"

    # Effective risk level (prefer combined_risk_level, fallback to risk_level)
    raw_level_col = (
        "combined_risk_level"
        if "combined_risk_level" in out.columns
        else ("risk_level" if "risk_level" in out.columns else None)
    )
    if raw_level_col:
        mapped = out[raw_level_col].map(normalize_risk_level)
        out["_risk_key"] = [m[0] for m in mapped]
        out["_risk_display"] = [m[1] for m in mapped]
        out["_risk_level_col"] = raw_level_col
    else:
        out["_risk_key"] = "UNKNOWN"
        out["_risk_display"] = "Not available"
        out["_risk_level_col"] = "Unavailable"

    out["_risk_rank"] = out["_risk_key"].map(
        lambda k: int(RISK_SEMANTICS.get(k, RISK_SEMANTICS["UNKNOWN"])["rank"])
    )

    # Parsed timestamp for chronological sorting and freshness
    if "timestamp" in out.columns:
        out["_timestamp_dt"] = pd.to_datetime(out["timestamp"], errors="coerce")
    else:
        out["_timestamp_dt"] = pd.NaT

    # Normalize known boolean columns without overwriting raw values if missing
    bool_cols = [
        "is_fraud",
        "rapid_transfer_flag",
        "new_recipient_flag",
        "profile_available",
        "profile_review_required",
        "location_anomaly_flag",
        "vpn_detected",
        "proxy_detected",
        "tor_detected",
        "hosting_network_detected",
        "network_review_signal",
        "needs_human_review",
    ]
    for bcol in bool_cols:
        if bcol in out.columns:
            out[f"_{bcol}_bool"] = parse_bool_series(out[bcol])

    return out


def prepare_ml_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize ML evaluation dataset columns safely, handling string booleans
    and numeric probabilities.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    actual_col = "actual_fraud" if "actual_fraud" in out.columns else (
        "is_fraud" if "is_fraud" in out.columns else None
    )
    if actual_col:
        out["_actual_bool"] = parse_bool_series(out[actual_col])
        out["_actual_col_name"] = actual_col
    else:
        out["_actual_bool"] = False
        out["_actual_col_name"] = "Unavailable"

    if "predicted_fraud" in out.columns:
        out["_predicted_bool"] = parse_bool_series(out["predicted_fraud"])
        out["_predicted_available"] = True
    else:
        out["_predicted_bool"] = False
        out["_predicted_available"] = False

    if "fraud_probability" in out.columns:
        out["_prob_num"] = pd.to_numeric(out["fraud_probability"], errors="coerce")
    else:
        out["_prob_num"] = np.nan

    if "amount" in out.columns:
        out["_amount_num"] = pd.to_numeric(out["amount"], errors="coerce").fillna(0.0)
    else:
        out["_amount_num"] = 0.0

    return out


# =============================================================================
# 3. FILTERING, SEARCH, SORTING & ML METRIC CALCULATIONS
# =============================================================================

def filter_and_sort_transactions(
    df: pd.DataFrame,
    search_query: str = "",
    selected_risk_levels: Optional[List[str]] = None,
    amount_range: Optional[Tuple[float, float]] = None,
    review_filter: str = "ALL",
    signal_filter: str = "ALL",
    sort_by: str = "RISK_DESC",
) -> pd.DataFrame:
    """
    Filter and sort the operational transactions dataframe consistently across
    Executive Overview and Investigation Desk.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    filtered = df.copy()

    # 1. Search by transaction_id, sender_id, or receiver_id
    if search_query and search_query.strip():
        q = search_query.strip().lower()
        mask = pd.Series(False, index=filtered.index)
        for col in ["transaction_id", "sender_id", "receiver_id"]:
            if col in filtered.columns:
                mask = mask | filtered[col].astype(str).str.lower().str.contains(q, regex=False, na=False)
        filtered = filtered[mask]

    # 2. Filter by risk levels
    if selected_risk_levels is not None and len(selected_risk_levels) > 0:
        allowed = {lvl.upper() for lvl in selected_risk_levels}
        filtered = filtered[filtered["_risk_key"].isin(allowed)]
    elif selected_risk_levels is not None and len(selected_risk_levels) == 0:
        return filtered.iloc[0:0]

    # 3. Filter by amount range
    if amount_range is not None and "_amount_num" in filtered.columns:
        min_amt, max_amt = amount_range
        filtered = filtered[
            (filtered["_amount_num"] >= float(min_amt) - 1e-6)
            & (filtered["_amount_num"] <= float(max_amt) + 1e-6)
        ]

    # 4. Filter by human review status
    if review_filter == "NEEDS_REVIEW" and "_needs_human_review_bool" in filtered.columns:
        filtered = filtered[filtered["_needs_human_review_bool"]]
    elif review_filter == "NO_REVIEW" and "_needs_human_review_bool" in filtered.columns:
        filtered = filtered[~filtered["_needs_human_review_bool"]]

    # 5. Filter by specific recorded signal
    signal_col_map = {
        "RAPID_TRANSFER": "_rapid_transfer_flag_bool",
        "NEW_RECIPIENT": "_new_recipient_flag_bool",
        "LOCATION_ANOMALY": "_location_anomaly_flag_bool",
        "PROFILE_REVIEW": "_profile_review_required_bool",
        "NETWORK_SIGNAL": "_network_review_signal_bool",
    }
    if signal_filter in signal_col_map:
        bcol = signal_col_map[signal_filter]
        if bcol in filtered.columns:
            filtered = filtered[filtered[bcol]]

    # 6. Sort order (High-Risk-First default)
    if sort_by == "RISK_DESC":
        filtered = filtered.sort_values(
            by=["_risk_score_num", "_risk_rank", "_amount_num"],
            ascending=[False, False, False],
            kind="mergesort",
        )
    elif sort_by == "RISK_ASC":
        filtered = filtered.sort_values(
            by=["_risk_score_num", "_risk_rank", "_amount_num"],
            ascending=[True, True, True],
            kind="mergesort",
        )
    elif sort_by == "AMOUNT_DESC":
        filtered = filtered.sort_values(
            by=["_amount_num", "_risk_score_num"],
            ascending=[False, False],
            kind="mergesort",
        )
    elif sort_by == "AMOUNT_ASC":
        filtered = filtered.sort_values(
            by=["_amount_num", "_risk_score_num"],
            ascending=[True, False],
            kind="mergesort",
        )
    elif sort_by == "TIME_DESC":
        filtered = filtered.sort_values(
            by=["_timestamp_dt", "_risk_score_num"],
            ascending=[False, False],
            na_position="last",
            kind="mergesort",
        )
    elif sort_by == "TIME_ASC":
        filtered = filtered.sort_values(
            by=["_timestamp_dt", "_risk_score_num"],
            ascending=[True, False],
            na_position="last",
            kind="mergesort",
        )

    return filtered


def compute_ml_evaluation_metrics(df_ml: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate confusion matrix counts and classification metrics from
    `data/ml_evaluation_results.csv` with safe zero-division handling.
    """
    if df_ml is None or df_ml.empty:
        return {
            "available": False,
            "reason": "ML evaluation dataset is empty or missing.",
            "total": 0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    prepared = prepare_ml_dataframe(df_ml)
    if prepared["_actual_col_name"].iloc[0] == "Unavailable" or not prepared["_predicted_available"].iloc[0]:
        return {
            "available": False,
            "reason": "Required columns (`actual_fraud`/`is_fraud` and `predicted_fraud`) are missing.",
            "total": len(prepared),
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    actual = prepared["_actual_bool"]
    pred = prepared["_predicted_bool"]

    tp = int((actual & pred).sum())
    fp = int((~actual & pred).sum())
    tn = int((~actual & ~pred).sum())
    fn = int((actual & ~pred).sum())
    total = tp + fp + tn + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2.0 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "available": True,
        "reason": "OK",
        "actual_col": prepared["_actual_col_name"].iloc[0],
        "total": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def inspect_project_environment(root: Path) -> Dict[str, Any]:
    """
    Inspect the local repository files and CSV schemas so the user and developer
    can verify which files and columns are present on disk.
    """
    key_files = [
        "app.py",
        "src/pipeline.py",
        "src/detection.py",
        "src/behavior_detection.py",
        "src/train_evaluate_ml.py",
        "requirements.txt",
        "README.md",
        "data/antifraudster_results.csv",
        "data/ml_evaluation_results.csv",
        "data/sample_transactions.csv",
        "data/sample_customers.csv",
        "data/evaluation_results.csv",
        "data/experimental_transactions.csv",
    ]
    status: Dict[str, bool] = {}
    for rel in key_files:
        status[rel] = (root / rel).exists()

    src_dir = root / "src"
    other_src = (
        sorted([p.name for p in src_dir.glob("*.py")])
        if src_dir.exists()
        else []
    )
    tests_dir = root / "tests"
    test_files = (
        sorted([p.name for p in tests_dir.glob("*.py")])
        if tests_dir.exists()
        else []
    )
    data_dir = root / "data"
    csv_files = (
        sorted([p.name for p in data_dir.glob("*.csv")])
        if data_dir.exists()
        else []
    )

    return {
        "root": str(root),
        "file_status": status,
        "src_files": other_src,
        "test_files": test_files,
        "csv_files": csv_files,
    }


# =============================================================================
# 4. CACHED CSV LOADING
# =============================================================================

def _load_csv_raw(file_path_str: str, mtime: float) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Internal helper to read a CSV file safely and report any read errors."""
    _ = mtime  # Used to invalidate cache automatically when CSV is regenerated
    path = Path(file_path_str)
    if not path.exists():
        return None, f"File not found at `{path}`"
    try:
        df = pd.read_csv(path)
        return df, None
    except Exception as exc:
        return None, f"Error reading `{path.name}`: {exc}"


if st is not None:
    load_csv_cached = st.cache_data(show_spinner=False)(_load_csv_raw)
else:
    load_csv_cached = _load_csv_raw


def load_dataset(root: Path, relative_path: str) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[datetime]]:
    """Load a CSV dataset from the project root and return (df, error_msg, file_mtime_dt)."""
    full_path = root / relative_path
    if not full_path.exists():
        return None, f"File `{relative_path}` was not found in `{root}`.", None
    mtime = full_path.stat().st_mtime
    mtime_dt = datetime.fromtimestamp(mtime)
    df, err = load_csv_cached(str(full_path), mtime)
    return df, err, mtime_dt


# =============================================================================
# 5. PREMIUM WHITE FINTECH CSS & REUSABLE HTML / JS COMPONENTS
# =============================================================================

def inject_custom_css() -> None:
    """
    Inject the '01 – Executive Overview' white & light-grey financial risk
    intelligence visual system.
    """
    if st is None:
        return

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        :root {
            --bg-app: #F6F8F7;
            --bg-surface: #FFFFFF;
            --bg-subtle: #F0F4F2;
            --text-primary: #111C18;
            --text-secondary: #52635E;
            --text-muted: #74857F;
            --border-subtle: #E1E8E4;
            --border-hover: #C5D3CC;
            --accent-primary: #0F3D2E;
            --accent-blue: #2563EB;
            --shadow-card: 0 2px 10px rgba(17, 28, 24, 0.04);
            --shadow-hover: 0 6px 18px rgba(17, 28, 24, 0.07);
        }

        /* Overall Page & Typography */
        .stApp {
            background-color: var(--bg-app);
            color: var(--text-primary);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 3rem !important;
            max-width: 1440px !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid var(--border-subtle) !important;
        }

        /* Section Eyebrow Labels */
        .af-eyebrow {
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-secondary);
            margin-bottom: 0.35rem;
        }

        /* Top Executive Header Banner (Large Panel: 22px radius) */
        .af-header-panel {
            background: #FFFFFF;
            border: 1px solid var(--border-subtle);
            border-radius: 22px;
            padding: 1.4rem 1.75rem;
            margin-bottom: 1.25rem;
            box-shadow: var(--shadow-card);
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
        }

        .af-brand-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            background: #EFF5F2;
            color: #0F3D2E;
            border: 1px solid #D3E2DA;
            border-radius: 9999px;
            padding: 0.25rem 0.75rem;
            font-size: 0.73rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }

        .af-page-title {
            font-size: 1.55rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.02em;
            margin: 0 0 0.25rem 0;
            line-height: 1.25;
        }

        .af-page-subtitle {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin: 0;
            max-width: 720px;
            line-height: 1.45;
        }

        .af-meta-cluster {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 0.4rem;
        }

        .af-freshness-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: #F8FAFC;
            border: 1px solid var(--border-subtle);
            border-radius: 9999px;
            padding: 0.3rem 0.8rem;
            font-size: 0.76rem;
            color: var(--text-secondary);
            font-family: 'JetBrains Mono', monospace;
        }

        .af-source-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: #EFF6FF;
            color: #1D4ED8;
            border: 1px solid #BFDBFE;
            border-radius: 9999px;
            padding: 0.24rem 0.75rem;
            font-size: 0.72rem;
            font-weight: 600;
        }

        /* Metric Cards (18-22px radius -> 20px) */
        .af-metric-card {
            background: #FFFFFF;
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 1.15rem 1.3rem;
            box-shadow: var(--shadow-card);
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }

        .af-metric-card:hover {
            border-color: var(--border-hover);
            box-shadow: var(--shadow-hover);
        }

        .af-metric-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.55rem;
        }

        .af-metric-label {
            font-size: 0.73rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-secondary);
        }

        .af-metric-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.02em;
            line-height: 1.2;
            margin-bottom: 0.4rem;
        }

        .af-metric-foot {
            font-size: 0.78rem;
            color: var(--text-secondary);
            line-height: 1.35;
        }

        /* Large Content Panels (18-24px radius -> 22px) */
        .af-panel {
            background: #FFFFFF;
            border: 1px solid var(--border-subtle);
            border-radius: 22px;
            padding: 1.35rem 1.5rem;
            box-shadow: var(--shadow-card);
            margin-bottom: 1.1rem;
        }

        .af-panel-header {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 0.85rem;
            border-bottom: 1px solid #F0F4F2;
            padding-bottom: 0.65rem;
        }

        .af-panel-title {
            font-size: 1.02rem;
            font-weight: 600;
            color: var(--text-primary);
            margin: 0;
        }

        .af-panel-caption {
            font-size: 0.78rem;
            color: var(--text-secondary);
        }

        /* Transaction Cards (16-20px radius -> 18px) */
        .af-tx-card {
            background: #FFFFFF;
            border: 1px solid var(--border-subtle);
            border-radius: 18px;
            padding: 1.15rem 1.3rem;
            box-shadow: var(--shadow-card);
            margin-bottom: 0.6rem;
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }

        .af-tx-card:hover {
            border-color: var(--border-hover);
            box-shadow: var(--shadow-hover);
        }

        .af-tx-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-bottom: 0.65rem;
        }

        .af-mono-id {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.83rem;
            font-weight: 600;
            color: var(--text-primary);
            background: #F4F7F5;
            border: 1px solid #E2E8E5;
            border-radius: 10px;
            padding: 0.2rem 0.55rem;
        }

        .af-mono-ts {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.76rem;
            color: var(--text-secondary);
        }

        .af-tx-amount {
            font-size: 1.4rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.015em;
            margin: 0.2rem 0 0.55rem 0;
        }

        .af-tx-route {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            flex-wrap: wrap;
            background: #F8FAF9;
            border: 1px solid #E9EFEA;
            border-radius: 12px;
            padding: 0.45rem 0.75rem;
            margin-bottom: 0.65rem;
            font-size: 0.8rem;
        }

        .af-tx-party {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 500;
            color: var(--text-primary);
        }

        .af-tx-meta-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.5rem;
            font-size: 0.79rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }

        /* Fully Rounded Semantic Pills (9999px) */
        .af-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            border-radius: 9999px;
            padding: 0.22rem 0.7rem;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            line-height: 1.2;
            white-space: nowrap;
        }

        .af-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            display: inline-block;
        }

        /* Smaller Inner Elements (10-14px radius -> 12px) */
        .af-inner-box {
            background: #F8FAF9;
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 0.75rem 0.95rem;
            margin-bottom: 0.6rem;
        }

        .af-reason-item {
            background: #F9FBFA;
            border: 1px solid #E3EBE6;
            border-left: 3px solid #0F3D2E;
            border-radius: 12px;
            padding: 0.6rem 0.85rem;
            margin-bottom: 0.45rem;
            font-size: 0.84rem;
            color: var(--text-primary);
        }

        .af-signal-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(195px, 1fr));
            gap: 0.55rem;
            margin-top: 0.5rem;
        }

        .af-signal-tile {
            border-radius: 12px;
            padding: 0.6rem 0.8rem;
            border: 1px solid var(--border-subtle);
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.8rem;
            font-weight: 500;
        }

        /* Buttons inside Streamlit */
        div.stButton > button {
            border-radius: 12px !important;
            border: 1px solid #D0DDD6 !important;
            background-color: #FFFFFF !important;
            color: #111C18 !important;
            font-weight: 600 !important;
            font-size: 0.82rem !important;
            padding: 0.42rem 0.9rem !important;
            transition: all 0.15s ease !important;
        }

        div.stButton > button:hover {
            border-color: #0F3D2E !important;
            background-color: #F2F7F4 !important;
            color: #0F3D2E !important;
        }

        div.stButton > button[kind="primary"] {
            background-color: #0F3D2E !important;
            color: #FFFFFF !important;
            border-color: #0F3D2E !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_risk_badge_html(raw_risk_level: Any) -> str:
    """Return a fully rounded semantic HTML pill for a transaction's risk level."""
    key, display_label = normalize_risk_level(raw_risk_level)
    tokens = RISK_SEMANTICS.get(key, RISK_SEMANTICS["UNKNOWN"])
    safe_label = html.escape(display_label)
    return (
        f'<span class="af-pill" style="background:{tokens["bg"]}; '
        f'color:{tokens["text"]}; border:1px solid {tokens["border"]};">'
        f'<span class="af-dot" style="background:{tokens["dot"]};"></span>'
        f"{safe_label}</span>"
    )


def render_status_pill_html(label: str, variant: str = "UNKNOWN") -> str:
    """Return a semantic status pill (e.g., for human review or ground-truth labels)."""
    tokens = RISK_SEMANTICS.get(variant, RISK_SEMANTICS["UNKNOWN"])
    safe_label = html.escape(label)
    return (
        f'<span class="af-pill" style="background:{tokens["bg"]}; '
        f'color:{tokens["text"]}; border:1px solid {tokens["border"]};">'
        f"{safe_label}</span>"
    )


def render_page_header(
    title: str,
    subtitle: str,
    freshness_text: str,
    source_badge: str = "Showing Saved Analysis Results",
) -> None:
    """Render the consistent top Executive Overview header panel."""
    st.markdown(
        f"""
        <div class="af-header-panel">
            <div>
                <div class="af-brand-pill">
                    <span>●</span>
                    <span>AntiFraudster · Risk &amp; Fraud Intelligence</span>
                </div>
                <h1 class="af-page-title">{html.escape(title)}</h1>
                <p class="af-page-subtitle">{html.escape(subtitle)}</p>
            </div>
            <div class="af-meta-cluster">
                <div class="af-source-pill">{html.escape(source_badge)}</div>
                <div class="af-freshness-pill">Data freshness: {html.escape(freshness_text)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str,
    value: str,
    footnote: str,
    badge_text: Optional[str] = None,
    badge_variant: str = "INFO",
) -> None:
    """Render a rounded (20px) metric card with optional semantic pill."""
    pill_html = (
        render_status_pill_html(badge_text, badge_variant) if badge_text else ""
    )
    st.markdown(
        f"""
        <div class="af-metric-card">
            <div>
                <div class="af-metric-top">
                    <span class="af-metric-label">{html.escape(label)}</span>
                    {pill_html}
                </div>
                <div class="af-metric-value">{html.escape(value)}</div>
            </div>
            <div class="af-metric-foot">{html.escape(footnote)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def extract_card_signals(row: pd.Series) -> List[Tuple[str, str]]:
    """
    Collect active recorded detection signals for a single transaction row.
    Returns list of (signal_label, semantic_variant).
    """
    signals: List[Tuple[str, str]] = []
    if parse_bool_value(row.get("rapid_transfer_flag")):
        signals.append(("Rapid transfer", "HIGH"))
    if parse_bool_value(row.get("new_recipient_flag")):
        signals.append(("New recipient", "MEDIUM"))
    if parse_bool_value(row.get("location_anomaly_flag")):
        signals.append(("Location anomaly", "HIGH"))
    if parse_bool_value(row.get("profile_review_required")):
        signals.append(("Profile review", "MEDIUM"))
    if parse_bool_value(row.get("network_review_signal")):
        signals.append(("Network signal", "MEDIUM"))
    if parse_bool_value(row.get("tor_detected")):
        signals.append(("Tor exit node", "CRITICAL"))
    elif parse_bool_value(row.get("proxy_detected")):
        signals.append(("Proxy detected", "HIGH"))
    elif parse_bool_value(row.get("vpn_detected")):
        signals.append(("VPN detected", "INFO"))
    if parse_bool_value(row.get("hosting_network_detected")):
        signals.append(("Hosting ASN", "HIGH"))
    return signals


def serialize_transactions_for_html_component(
    df: pd.DataFrame, max_records: int = 250
) -> List[Dict[str, Any]]:
    """
    Serialize real operational transaction rows into a JSON-safe list of dicts
    for the embedded `components.html(...)` live HTML/JS widget.
    """
    if df is None or df.empty:
        return []

    records: List[Dict[str, Any]] = []
    for _, row in df.head(max_records).iterrows():
        raw_level = (
            row.get("combined_risk_level")
            if pd.notna(row.get("combined_risk_level"))
            else row.get("risk_level", "UNKNOWN")
        )
        r_key, r_display = normalize_risk_level(raw_level)
        score_val = (
            row.get("combined_risk_score")
            if pd.notna(row.get("combined_risk_score"))
            else row.get("risk_score")
        )
        score_num = float(score_val) if score_val is not None and pd.notna(score_val) else 0.0

        reasons_all = (
            parse_explanation_items(row.get("reasons"))
            + parse_explanation_items(row.get("behavior_reasons"))
            + parse_explanation_items(row.get("profile_review_reason"))
            + parse_explanation_items(row.get("location_reason"))
            + parse_explanation_items(row.get("network_reason"))
            + parse_explanation_items(row.get("additional_review_reasons"))
        )

        signals_active = [lbl for lbl, _ in extract_card_signals(row)]
        checklist = {
            "Rapid transfer": parse_bool_value(row.get("rapid_transfer_flag")),
            "New recipient": parse_bool_value(row.get("new_recipient_flag")),
            "Location anomaly": parse_bool_value(row.get("location_anomaly_flag")),
            "Profile review required": parse_bool_value(row.get("profile_review_required")),
            "Network review signal": parse_bool_value(row.get("network_review_signal")),
            "VPN detected": parse_bool_value(row.get("vpn_detected")),
            "Proxy detected": parse_bool_value(row.get("proxy_detected")),
            "Tor detected": parse_bool_value(row.get("tor_detected")),
            "Hosting network": parse_bool_value(row.get("hosting_network_detected")),
        }

        records.append(
            {
                "transaction_id": str(row.get("transaction_id", "Not available")),
                "timestamp": format_timestamp(row.get("timestamp")),
                "sender_id": str(row.get("sender_id", "Not available")),
                "receiver_id": str(row.get("receiver_id", "Not available")),
                "amount_raw": float(row.get("_amount_num", 0.0)),
                "amount_inr": format_inr(row.get("amount")),
                "risk_key": r_key,
                "risk_display": r_display,
                "risk_score": round(score_num, 1),
                "needs_human_review": parse_bool_value(row.get("needs_human_review")),
                "is_fraud": (
                    parse_bool_value(row.get("is_fraud"))
                    if "is_fraud" in row.index and pd.notna(row.get("is_fraud"))
                    else None
                ),
                "scenario": (
                    str(row.get("scenario"))
                    if pd.notna(row.get("scenario")) and str(row.get("scenario")).strip()
                    else "Not available"
                ),
                "customer_type": (
                    str(row.get("customer_type"))
                    if pd.notna(row.get("customer_type"))
                    else "Not available"
                ),
                "kyc_status": (
                    str(row.get("kyc_status"))
                    if pd.notna(row.get("kyc_status"))
                    else "Not available"
                ),
                "network_type": (
                    str(row.get("network_type"))
                    if pd.notna(row.get("network_type"))
                    else "Not available"
                ),
                "signals": signals_active,
                "checklist": checklist,
                "reasons": reasons_all,
            }
        )
    return records


def build_interactive_html_component_template(records: List[Dict[str, Any]]) -> str:
    """
    Build a self-contained HTML/CSS/JS frontend widget suitable for
    `streamlit.components.v1.html(...)` (`components.html`).
    Supports instant client-side search, risk-pill filtering, card inspection modal,
    and one-click Transaction ID copy without full page reloads.
    """
    payload_json = json.dumps(records).replace("</", "<\\/")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 4px;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: transparent;
    color: #111C18;
  }}

  .af-widget-shell {{
    background: #FFFFFF;
    border: 1px solid #E1E8E4;
    border-radius: 22px;
    padding: 18px 20px;
    box-shadow: 0 2px 10px rgba(17, 28, 24, 0.04);
  }}

  .af-toolbar {{
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    padding-bottom: 14px;
    border-bottom: 1px solid #F0F4F2;
    margin-bottom: 14px;
  }}

  .af-search-input {{
    flex: 1 1 240px;
    padding: 8px 12px;
    border: 1px solid #D0DDD6;
    border-radius: 12px;
    font-size: 13px;
    font-family: 'Inter', sans-serif;
    outline: none;
  }}
  .af-search-input:focus {{
    border-color: #0F3D2E;
  }}

  .af-pill-group {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }}

  .af-filter-btn {{
    border: 1px solid #D0DDD6;
    background: #FFFFFF;
    color: #52635E;
    border-radius: 9999px;
    padding: 5px 12px;
    font-size: 11.5px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
  }}
  .af-filter-btn.active {{
    background: #0F3D2E;
    color: #FFFFFF;
    border-color: #0F3D2E;
  }}

  .af-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(290px, 1fr));
    gap: 12px;
  }}

  .af-card {{
    background: #FFFFFF;
    border: 1px solid #E1E8E4;
    border-radius: 18px;
    padding: 14px 16px;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }}
  .af-card:hover {{
    border-color: #0F3D2E;
    box-shadow: 0 6px 18px rgba(17, 28, 24, 0.07);
  }}

  .af-badge {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    border-radius: 9999px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 600;
  }}
  .af-badge-CRITICAL {{ background: #FEF2F2; color: #B91C1C; border: 1px solid #FCA5A5; }}
  .af-badge-HIGH     {{ background: #FFF7ED; color: #C2410C; border: 1px solid #FDBA74; }}
  .af-badge-MEDIUM   {{ background: #FEFCE8; color: #A16207; border: 1px solid #FDE047; }}
  .af-badge-LOW      {{ background: #F0FDF4; color: #15803D; border: 1px solid #86EFAC; }}
  .af-badge-UNKNOWN  {{ background: #F3F4F6; color: #4B5563; border: 1px solid #D1D5DB; }}

  .af-mono {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    background: #F4F7F5;
    border: 1px solid #E2E8E5;
    border-radius: 8px;
    padding: 2px 7px;
  }}

  .af-inspect-btn {{
    margin-top: 10px;
    width: 100%;
    border-radius: 10px;
    border: 1px solid #D0DDD6;
    background: #F8FAF9;
    color: #111C18;
    font-weight: 600;
    font-size: 12px;
    padding: 7px 10px;
    cursor: pointer;
    transition: all 0.15s ease;
  }}
  .af-inspect-btn:hover {{
    background: #0F3D2E;
    color: #FFFFFF;
    border-color: #0F3D2E;
  }}

  .af-docket-drawer {{
    display: none;
    background: #F8FAF9;
    border: 1px solid #C5D3CC;
    border-left: 4px solid #0F3D2E;
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 16px;
  }}
  .af-docket-drawer.open {{
    display: block;
  }}
</style>
</head>
<body>
<div class="af-widget-shell">
  <div class="af-toolbar">
    <input id="txSearchInput" class="af-search-input" type="text"
           placeholder="Instant JS filter by Transaction ID, Sender ID, or Receiver ID..." />
    <div class="af-pill-group" id="riskFilterBar">
      <button class="af-filter-btn active" data-risk="ALL">All Levels</button>
      <button class="af-filter-btn" data-risk="CRITICAL">Critical</button>
      <button class="af-filter-btn" data-risk="HIGH">High</button>
      <button class="af-filter-btn" data-risk="MEDIUM">Medium</button>
      <button class="af-filter-btn" data-risk="LOW">Low</button>
      <button class="af-filter-btn" data-risk="REVIEW">Needs Human Review</button>
    </div>
    <span id="txCountStatus" style="font-size:12px; color:#52635E; font-weight:600;"></span>
  </div>

  <!-- Live JS Investigation Drawer -->
  <div id="docketDrawer" class="af-docket-drawer"></div>

  <!-- Transaction Cards Container -->
  <div id="txCardGrid" class="af-grid"></div>
</div>

<script>
  const TX_DATA = {payload_json};
  let activeRisk = 'ALL';
  let searchQuery = '';

  const searchInput = document.getElementById('txSearchInput');
  const filterBtns = document.querySelectorAll('#riskFilterBar .af-filter-btn');
  const gridEl = document.getElementById('txCardGrid');
  const countEl = document.getElementById('txCountStatus');
  const drawerEl = document.getElementById('docketDrawer');

  function escapeHtml(str) {{
    return String(str ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }}

  function openDocket(txId) {{
    const tx = TX_DATA.find(item => item.transaction_id === txId);
    if (!tx) return;

    const reasonsHtml = (tx.reasons && tx.reasons.length > 0)
      ? tx.reasons.map(r => `<li style="margin-bottom:4px;">${{escapeHtml(r)}}</li>`).join('')
      : '<li>No elevated rule or behavioral reasons recorded.</li>';

    const checklistHtml = Object.entries(tx.checklist || {{}})
      .map(([k, v]) => `<span class="af-badge ${{v ? 'af-badge-HIGH' : 'af-badge-LOW'}}" style="margin-right:5px; margin-bottom:5px;">${{escapeHtml(k)}}: ${{v ? 'ACTIVE' : 'INACTIVE'}}</span>`)
      .join('');

    drawerEl.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:10px;">
        <div>
          <span class="af-mono">${{escapeHtml(tx.transaction_id)}}</span>
          <span style="font-size:12px; color:#52635E; margin-left:8px;">${{escapeHtml(tx.timestamp)}}</span>
          <span class="af-badge af-badge-${{escapeHtml(tx.risk_key)}}" style="margin-left:8px;">${{escapeHtml(tx.risk_display)}}</span>
        </div>
        <button id="closeDrawerBtn" class="af-filter-btn">Close Preview ✕</button>
      </div>
      <div style="font-size:20px; font-weight:700; margin-bottom:6px;">
        ${{escapeHtml(tx.amount_inr)}}
        <span style="font-size:12px; font-weight:500; color:#52635E; margin-left:10px;">
          Combined Score: ${{escapeHtml(tx.risk_score)}} · Sender: ${{escapeHtml(tx.sender_id)}} → Receiver: ${{escapeHtml(tx.receiver_id)}}
        </span>
      </div>
      <div style="font-size:12px; color:#52635E; margin-bottom:8px;">
        Customer Type: <strong>${{escapeHtml(tx.customer_type)}}</strong> ·
        KYC: <strong>${{escapeHtml(tx.kyc_status)}}</strong> ·
        Network: <strong>${{escapeHtml(tx.network_type)}}</strong> ·
        Scenario: <strong>${{escapeHtml(tx.scenario)}}</strong>
      </div>
      <div style="margin-bottom:8px;">${{checklistHtml}}</div>
      <div style="font-size:12.5px; color:#111C18;">
        <strong>Recorded Detection &amp; Escalation Reasons:</strong>
        <ul style="margin:6px 0 0 18px; padding:0;">${{reasonsHtml}}</ul>
      </div>
    `;
    drawerEl.classList.add('open');
    const closeBtn = document.getElementById('closeDrawerBtn');
    if (closeBtn) {{
      closeBtn.addEventListener('click', () => drawerEl.classList.remove('open'));
    }}
  }}

  function renderCards() {{
    const filtered = TX_DATA.filter(item => {{
      if (activeRisk === 'REVIEW' && !item.needs_human_review) return false;
      if (activeRisk !== 'ALL' && activeRisk !== 'REVIEW' && item.risk_key !== activeRisk) return false;
      if (searchQuery) {{
        const hay = `${{item.transaction_id}} ${{item.sender_id}} ${{item.receiver_id}}`.toLowerCase();
        if (!hay.includes(searchQuery)) return false;
      }}
      return true;
    }});

    countEl.innerText = `Showing ${{Math.min(filtered.length, 12)}} of ${{filtered.length}} matching records`;

    if (filtered.length === 0) {{
      gridEl.innerHTML = `<div style="grid-column:1/-1; text-align:center; padding:24px; color:#52635E; font-size:13px;">No transactions match your current filter.</div>`;
      return;
    }}

    gridEl.innerHTML = filtered.slice(0, 12).map(tx => {{
      const reviewTag = tx.needs_human_review
        ? `<span class="af-badge af-badge-HIGH">Needs Review</span>`
        : `<span class="af-badge af-badge-LOW">Standard</span>`;
      const reasonSnippet = (tx.reasons && tx.reasons.length > 0)
        ? tx.reasons[0]
        : 'No primary rule or behavioral trigger recorded.';
      return `
        <div class="af-card">
          <div>
            <div style="display:flex; justify-content:space-between; align-items:center; gap:6px; margin-bottom:8px;">
              <span class="af-mono">${{escapeHtml(tx.transaction_id)}}</span>
              <div style="display:flex; gap:4px;">
                <span class="af-badge af-badge-${{escapeHtml(tx.risk_key)}}">${{escapeHtml(tx.risk_display)}}</span>
                ${{reviewTag}}
              </div>
            </div>
            <div style="font-size:19px; font-weight:700; margin-bottom:6px;">${{escapeHtml(tx.amount_inr)}}</div>
            <div style="font-size:12px; color:#52635E; background:#F8FAF9; border:1px solid #E9EFEA; border-radius:10px; padding:6px 8px; margin-bottom:8px;">
              <strong>${{escapeHtml(tx.sender_id)}}</strong> → <strong>${{escapeHtml(tx.receiver_id)}}</strong>
              <div style="margin-top:3px; font-size:11px;">Score: <code>${{escapeHtml(tx.risk_score)}}</code> · ${{escapeHtml(tx.timestamp)}}</div>
            </div>
            <div style="font-size:11.5px; color:#52635E; line-height:1.35;">${{escapeHtml(reasonSnippet)}}</div>
          </div>
          <button class="af-inspect-btn" data-txid="${{escapeHtml(tx.transaction_id)}}">
            Quick JS Inspect ${{escapeHtml(tx.transaction_id)}}
          </button>
        </div>
      `;
    }}).join('');

    gridEl.querySelectorAll('.af-inspect-btn').forEach(btn => {{
      btn.addEventListener('click', () => openDocket(btn.getAttribute('data-txid')));
    }});
  }}

  searchInput.addEventListener('input', (e) => {{
    searchQuery = (e.target.value || '').trim().toLowerCase();
    renderCards();
  }});

  filterBtns.forEach(btn => {{
    btn.addEventListener('click', () => {{
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeRisk = btn.getAttribute('data-risk') || 'ALL';
      renderCards();
    }});
  }});

  renderCards();
</script>
</body>
</html>"""


def render_interactive_html_widget(df: pd.DataFrame, height: int = 580) -> None:
    """
    Render the live HTML/CSS/JS transaction widget via `components.html(...)`
    when `streamlit.components.v1` is available.
    """
    if components is None or df is None or df.empty:
        return
    records = serialize_transactions_for_html_component(df)
    html_doc = build_interactive_html_component_template(records)
    components.html(html_doc, height=height, scrolling=True)


def render_transaction_card(
    row: pd.Series,
    button_key_prefix: str,
    on_open_callback: Any,
) -> None:
    """
    Render an interactive, fintech-styled transaction card (18px radius) with
    real fields from `data/antifraudster_results.csv` and an action button that
    opens the transaction in the Investigation Desk.
    """
    tx_id = str(row.get("transaction_id", "Not available"))
    ts_str = format_timestamp(row.get("timestamp"))
    amt_str = format_inr(row.get("amount"))
    sender = str(row.get("sender_id", "Not available"))
    receiver = str(row.get("receiver_id", "Not available"))

    raw_level = (
        row.get("combined_risk_level")
        if pd.notna(row.get("combined_risk_level"))
        else row.get("risk_level", "UNKNOWN")
    )
    risk_badge = render_risk_badge_html(raw_level)

    score_val = (
        row.get("combined_risk_score")
        if pd.notna(row.get("combined_risk_score"))
        else row.get("risk_score")
    )
    score_str = (
        f"{float(score_val):.1f}"
        if score_val is not None and pd.notna(score_val)
        else "Not available"
    )

    needs_review = parse_bool_value(row.get("needs_human_review"))
    review_pill = (
        render_status_pill_html("Human Review Required", "HIGH")
        if needs_review
        else render_status_pill_html("Standard Queue", "LOW")
    )

    # Recorded signals & primary reason preview
    signals = extract_card_signals(row)
    if signals:
        sig_html = " ".join(
            render_status_pill_html(lbl, var) for lbl, var in signals[:4]
        )
    else:
        sig_html = render_status_pill_html("No elevated auxiliary flags", "UNKNOWN")

    reasons_list = (
        parse_explanation_items(row.get("reasons"))
        + parse_explanation_items(row.get("behavior_reasons"))
        + parse_explanation_items(row.get("location_reason"))
        + parse_explanation_items(row.get("network_reason"))
    )
    reason_preview = (
        reasons_list[0]
        if reasons_list
        else "No primary rule or behavioral trigger recorded."
    )
    if len(reason_preview) > 110:
        reason_preview = reason_preview[:107] + "..."

    scenario_val = row.get("scenario")
    scenario_str = (
        f" · Scenario: {html.escape(str(scenario_val))}"
        if scenario_val is not None and pd.notna(scenario_val) and str(scenario_val).strip()
        else ""
    )

    st.markdown(
        f"""
        <div class="af-tx-card">
            <div class="af-tx-top">
                <div>
                    <span class="af-mono-id">{html.escape(tx_id)}</span>
                    <span class="af-mono-ts" style="margin-left:0.45rem;">{html.escape(ts_str)}</span>
                </div>
                <div style="display:flex; gap:0.4rem; align-items:center;">
                    {risk_badge}
                    {review_pill}
                </div>
            </div>
            <div class="af-tx-amount">{html.escape(amt_str)}</div>
            <div class="af-tx-route">
                <span style="color:#52635E;">Sender</span>
                <span class="af-tx-party">{html.escape(sender)}</span>
                <span style="color:#74857F;">→</span>
                <span style="color:#52635E;">Receiver</span>
                <span class="af-tx-party">{html.escape(receiver)}</span>
            </div>
            <div class="af-tx-meta-row">
                <span><strong>Combined Risk Score:</strong> <code style="font-family:'JetBrains Mono',monospace;">{html.escape(score_str)}</code>{scenario_str}</span>
            </div>
            <div style="margin-bottom:0.55rem; display:flex; flex-wrap:wrap; gap:0.35rem;">
                {sig_html}
            </div>
            <div style="font-size:0.79rem; color:#52635E; border-top:1px solid #F0F4F2; padding-top:0.45rem;">
                {html.escape(reason_preview)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.button(
        f"Inspect Transaction {tx_id} →",
        key=f"{button_key_prefix}_{tx_id}",
        use_container_width=True,
        on_click=on_open_callback,
        args=(tx_id,),
    )


# =============================================================================
# 6. PLOTLY CHART BUILDERS (WHITE / TRANSPARENT, SUBTLE GRIDLINES)
# =============================================================================

def _apply_clean_plotly_layout(fig: Any, height: int = 300, y_title: str = "", x_title: str = "") -> Any:
    """Apply consistent light, minimalist financial analytics styling to Plotly figures."""
    fig.update_layout(
        height=height,
        margin=dict(l=16, r=16, t=28, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", size=12, color="#111C18"),
        xaxis=dict(
            title=x_title,
            showgrid=False,
            linecolor="#E1E8E4",
            tickfont=dict(size=11, color="#52635E"),
        ),
        yaxis=dict(
            title=y_title,
            showgrid=True,
            gridcolor="#EEF2F0",
            zeroline=False,
            tickfont=dict(size=11, color="#52635E"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#52635E"),
        ),
    )
    return fig


def build_risk_volume_and_value_charts(df: pd.DataFrame) -> Tuple[Any, Any]:
    """Build (1) Transaction Count by Risk Level and (2) Total Value (₹) by Risk Level."""
    if go is None or df.empty:
        return None, None

    levels_present = [lvl for lvl in RISK_ORDER if lvl in set(df["_risk_key"])]
    extra_levels = sorted(list(set(df["_risk_key"]) - set(RISK_ORDER)))
    ordered_keys = levels_present + extra_levels

    counts = []
    values = []
    colors = []
    borders = []
    for k in ordered_keys:
        sub = df[df["_risk_key"] == k]
        counts.append(int(len(sub)))
        values.append(float(sub["_amount_num"].sum()))
        sem = RISK_SEMANTICS.get(k, RISK_SEMANTICS["UNKNOWN"])
        colors.append(sem["accent"])
        borders.append(sem["border"])

    fig_count = go.Figure(
        data=[
            go.Bar(
                x=ordered_keys,
                y=counts,
                text=[f"{c:,}" for c in counts],
                textposition="outside",
                marker=dict(color=colors, line=dict(color=borders, width=1), cornerradius=6),
                hovertemplate="Risk Level: <b>%{x}</b><br>Transactions: %{y:,}<extra></extra>",
            )
        ]
    )
    _apply_clean_plotly_layout(fig_count, height=290, y_title="Transactions", x_title="Risk Level")

    fig_val = go.Figure(
        data=[
            go.Bar(
                x=ordered_keys,
                y=values,
                text=[format_inr(v, include_decimals=False) for v in values],
                textposition="outside",
                marker=dict(color=colors, line=dict(color=borders, width=1), cornerradius=6),
                hovertemplate="Risk Level: <b>%{x}</b><br>Total Value: ₹%{y:,.2f}<extra></extra>",
            )
        ]
    )
    _apply_clean_plotly_layout(fig_val, height=290, y_title="Total Value (₹)", x_title="Risk Level")
    return fig_count, fig_val


def build_signal_counts_chart(df: pd.DataFrame) -> Any:
    """Build horizontal bar chart of recorded behavioral, location, profile, and network signals."""
    if go is None or df.empty:
        return None

    signal_specs = [
        ("Needs Human Review", "_needs_human_review_bool", "#EA580C"),
        ("Rapid Transfer Flag", "_rapid_transfer_flag_bool", "#2563EB"),
        ("New Recipient Flag", "_new_recipient_flag_bool", "#2563EB"),
        ("Profile Review Required", "_profile_review_required_bool", "#D97706"),
        ("Location Anomaly Flag", "_location_anomaly_flag_bool", "#DC2626"),
        ("Network Review Signal", "_network_review_signal_bool", "#0F3D2E"),
    ]
    labels = []
    counts = []
    colors = []
    for label, col, color in reversed(signal_specs):
        if col in df.columns:
            labels.append(label)
            counts.append(int(df[col].sum()))
            colors.append(color)

    if not labels:
        return None

    fig = go.Figure(
        data=[
            go.Bar(
                y=labels,
                x=counts,
                orientation="h",
                text=[f"{c:,}" for c in counts],
                textposition="outside",
                marker=dict(color=colors, cornerradius=6),
                hovertemplate="<b>%{y}</b>: %{x:,} transactions<extra></extra>",
            )
        ]
    )
    _apply_clean_plotly_layout(fig, height=290, x_title="Flagged Transactions")
    fig.update_layout(
        xaxis=dict(showgrid=True, gridcolor="#EEF2F0"),
        yaxis=dict(showgrid=False),
    )
    return fig


def build_scenario_distribution_chart(df: pd.DataFrame) -> Any:
    """Build scenario breakdown stacked by risk level when `scenario` column is present."""
    if go is None or df.empty or "scenario" not in df.columns:
        return None

    scen_df = df.dropna(subset=["scenario"]).copy()
    if scen_df.empty:
        return None

    top_scenarios = (
        scen_df["scenario"].astype(str).value_counts().head(10).index.tolist()
    )
    scen_df = scen_df[scen_df["scenario"].astype(str).isin(top_scenarios)]

    fig = go.Figure()
    for r_key in RISK_ORDER:
        sub = scen_df[scen_df["_risk_key"] == r_key]
        if sub.empty:
            continue
        counts_by_scen = sub["scenario"].astype(str).value_counts()
        y_vals = [int(counts_by_scen.get(s, 0)) for s in top_scenarios]
        sem = RISK_SEMANTICS[r_key]
        fig.add_trace(
            go.Bar(
                name=r_key,
                x=top_scenarios,
                y=y_vals,
                marker=dict(color=sem["accent"]),
            )
        )

    fig.update_layout(barmode="stack")
    _apply_clean_plotly_layout(
        fig, height=290, y_title="Transactions", x_title="Recorded Scenario"
    )
    return fig


def build_sender_receiver_network_figure(df: pd.DataFrame, max_edges: int = 45) -> Any:
    """
    Build an interactive sender-to-receiver transfer network graph from actual
    `sender_id` -> `receiver_id` pairs present in `data/antifraudster_results.csv`.
    """
    if go is None or df.empty or "sender_id" not in df.columns or "receiver_id" not in df.columns:
        return None

    edges = (
        df.groupby(["sender_id", "receiver_id"], as_index=False)
        .agg(
            tx_count=("transaction_id", "count"),
            total_amount=("_amount_num", "sum"),
            max_risk_rank=("_risk_rank", "max"),
            max_score=("_risk_score_num", "max"),
        )
        .sort_values(by=["tx_count", "max_score", "total_amount"], ascending=[False, False, False])
        .head(max_edges)
    )

    if edges.empty:
        return None

    senders = sorted(edges["sender_id"].astype(str).unique().tolist())
    receivers = sorted(edges["receiver_id"].astype(str).unique().tolist())
    all_nodes = sorted(list(set(senders) | set(receivers)))
    if not all_nodes:
        return None

    # Deterministic bipartite / circular layout so positions never jitter
    pos: Dict[str, Tuple[float, float]] = {}
    only_senders = [n for n in all_nodes if n in senders and n not in receivers]
    only_receivers = [n for n in all_nodes if n in receivers and n not in senders]
    both_roles = [n for n in all_nodes if n in senders and n in receivers]

    def _spread_y(items: List[str], x_coord: float) -> None:
        n = len(items)
        for idx, item in enumerate(items):
            y = 1.0 - (2.0 * idx / max(n - 1, 1)) if n > 1 else 0.0
            pos[item] = (x_coord, y)

    _spread_y(only_senders, -1.0)
    _spread_y(both_roles, 0.0)
    _spread_y(only_receivers, 1.0)

    rank_to_key = {4: "CRITICAL", 3: "HIGH", 2: "MEDIUM", 1: "LOW", 0: "UNKNOWN"}

    fig = go.Figure()

    # Draw edges grouped by highest risk rank along that corridor
    for rank_val, r_key in rank_to_key.items():
        sub_edges = edges[edges["max_risk_rank"] == rank_val]
        if sub_edges.empty:
            continue
        edge_x: List[Optional[float]] = []
        edge_y: List[Optional[float]] = []
        for _, erow in sub_edges.iterrows():
            s_node = str(erow["sender_id"])
            r_node = str(erow["receiver_id"])
            x0, y0 = pos[s_node]
            x1, y1 = pos[r_node]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        sem = RISK_SEMANTICS[r_key]
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                name=f"{r_key} corridor",
                line=dict(width=1.8, color=sem["accent"]),
                hoverinfo="none",
            )
        )

    # Node traces
    node_x = [pos[n][0] for n in all_nodes]
    node_y = [pos[n][1] for n in all_nodes]
    node_colors = []
    node_hover = []
    for n in all_nodes:
        out_rows = edges[edges["sender_id"].astype(str) == n]
        in_rows = edges[edges["receiver_id"].astype(str) == n]
        out_cnt = int(out_rows["tx_count"].sum()) if not out_rows.empty else 0
        in_cnt = int(in_rows["tx_count"].sum()) if not in_rows.empty else 0
        out_amt = float(out_rows["total_amount"].sum()) if not out_rows.empty else 0.0
        in_amt = float(in_rows["total_amount"].sum()) if not in_rows.empty else 0.0
        role = (
            "Sender & Receiver"
            if (out_cnt > 0 and in_cnt > 0)
            else ("Sender" if out_cnt > 0 else "Receiver")
        )
        node_colors.append(
            "#0F3D2E" if role == "Sender" else ("#2563EB" if role == "Receiver" else "#D97706")
        )
        node_hover.append(
            f"<b>{n}</b> ({role})<br>"
            f"Outbound tx: {out_cnt} ({format_inr(out_amt)})<br>"
            f"Inbound tx: {in_cnt} ({format_inr(in_amt)})"
        )

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            name="Accounts",
            text=all_nodes,
            textposition="top center",
            textfont=dict(family="JetBrains Mono, monospace", size=10, color="#111C18"),
            hovertext=node_hover,
            hoverinfo="text",
            marker=dict(
                size=13,
                color=node_colors,
                line=dict(width=2, color="#FFFFFF"),
            ),
        )
    )

    fig.update_layout(
        height=390,
        margin=dict(l=16, r=16, t=28, b=16),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            tickvals=[-1.0, 0.0, 1.0],
            ticktext=["Senders", "Dual-Role Accounts", "Receivers"],
            range=[-1.35, 1.35],
        ),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# =============================================================================
# 7. MISSING DATA HELPER PANEL
# =============================================================================

def render_missing_data_message(relative_csv: str, root: Path, error_msg: str) -> None:
    """Display a clear, helpful diagnostic when a required CSV dataset is missing."""
    st.markdown(
        f"""
        <div class="af-panel" style="border-left:4px solid #D97706;">
            <div class="af-eyebrow" style="color:#A16207;">Dataset Not Found</div>
            <h3 class="af-panel-title" style="margin-bottom:0.4rem;">
                Unable to load <code>{html.escape(relative_csv)}</code>
            </h3>
            <p style="font-size:0.88rem; color:#52635E; margin-bottom:0.75rem;">
                {html.escape(error_msg)}
            </p>
            <div class="af-inner-box" style="font-family:'JetBrains Mono',monospace; font-size:0.8rem;">
                <strong>How to generate the saved datasets from the project root ({html.escape(str(root))}):</strong><br><br>
                1. Generate operational fraud-risk pipeline results (<code>data/antifraudster_results.csv</code>):<br>
                &nbsp;&nbsp;&nbsp;<code>python3 -m src.pipeline</code> &nbsp;or&nbsp; <code>python3 src/pipeline.py</code><br><br>
                2. Generate experimental ML evaluation results (<code>data/ml_evaluation_results.csv</code>):<br>
                &nbsp;&nbsp;&nbsp;<code>python3 -m src.train_evaluate_ml</code> &nbsp;or&nbsp; <code>python3 src/train_evaluate_ml.py</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# 8. PAGE 1 — EXECUTIVE PULSE / EXECUTIVE OVERVIEW
# =============================================================================

def render_page_executive_overview(
    df_op: Optional[pd.DataFrame],
    op_err: Optional[str],
    op_mtime: Optional[datetime],
    root: Path,
    on_select_tx: Any,
) -> None:
    """Render Page 1: 01 – Executive Overview."""
    freshness = "Unavailable"
    if df_op is not None and not df_op.empty and "_timestamp_dt" in df_op.columns:
        max_ts = df_op["_timestamp_dt"].max()
        if pd.notna(max_ts):
            freshness = f"Latest tx {max_ts.strftime('%Y-%m-%d %H:%M:%S')}"
    if freshness == "Unavailable" and op_mtime is not None:
        freshness = f"File saved {op_mtime.strftime('%Y-%m-%d %H:%M:%S')}"

    render_page_header(
        title="01 — Executive Overview & Portfolio Pulse",
        subtitle=(
            "Portfolio-level fraud-risk triage, recorded behavioral & network anomalies, "
            "and human-review queue prioritization from saved operational pipeline outputs."
        ),
        freshness_text=freshness,
        source_badge="Saved Pipeline Results · data/antifraudster_results.csv",
    )

    if df_op is None or df_op.empty:
        render_missing_data_message(
            "data/antifraudster_results.csv",
            root,
            op_err or "The operational results file is empty.",
        )
        return

    # Quick Portfolio Filter Bar inside a clean Bento panel
    with st.container():
        st.markdown('<div class="af-eyebrow">Portfolio View Controls</div>', unsafe_allow_html=True)
        fcol1, fcol2, fcol3 = st.columns([2.2, 1.3, 1.5])
        with fcol1:
            exec_risk_filter = st.multiselect(
                "Filter Portfolio by Risk Level",
                options=RISK_ORDER,
                default=RISK_ORDER,
                key="exec_risk_levels",
            )
        with fcol2:
            exec_review_mode = st.selectbox(
                "Human-Review Status",
                options=["ALL", "NEEDS_REVIEW", "NO_REVIEW"],
                format_func=lambda x: {
                    "ALL": "All Transactions",
                    "NEEDS_REVIEW": "Requires Human Review Only",
                    "NO_REVIEW": "Standard Queue Only",
                }[x],
                key="exec_review_mode",
            )
        with fcol3:
            exec_search = st.text_input(
                "Quick Search (Tx ID, Sender, Receiver)",
                value="",
                placeholder="e.g., TX1004 or CUST_01...",
                key="exec_quick_search",
            )

    view_df = filter_and_sort_transactions(
        df_op,
        search_query=exec_search,
        selected_risk_levels=exec_risk_filter,
        review_filter=exec_review_mode,
        sort_by="RISK_DESC",
    )

    # Top Bento Metric Cards (5 columns)
    total_count = len(view_df)
    portfolio_total = len(df_op)
    review_count = (
        int(view_df["_needs_human_review_bool"].sum())
        if "_needs_human_review_bool" in view_df.columns
        else 0
    )
    critical_count = int((view_df["_risk_key"] == "CRITICAL").sum())
    high_count = int((view_df["_risk_key"] == "HIGH").sum())
    total_value_inr = format_inr(view_df["_amount_num"].sum())

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        render_metric_card(
            label="Filtered Transactions",
            value=f"{total_count:,}",
            footnote=f"Out of {portfolio_total:,} recorded operational transactions.",
            badge_text="PORTFOLIO",
            badge_variant="INFO",
        )
    with m2:
        review_pct = (review_count / total_count * 100.0) if total_count > 0 else 0.0
        render_metric_card(
            label="Needs Human Review",
            value=f"{review_count:,}",
            footnote=f"{review_pct:.1f}% of current view flagged via needs_human_review.",
            badge_text="QUEUE",
            badge_variant="HIGH" if review_count > 0 else "LOW",
        )
    with m3:
        render_metric_card(
            label="Critical-Risk Count",
            value=f"{critical_count:,}",
            footnote=f"Plus {high_count:,} HIGH-risk transactions in current view.",
            badge_text="CRITICAL",
            badge_variant="CRITICAL" if critical_count > 0 else "LOW",
        )
    with m4:
        render_metric_card(
            label="Total Transaction Value",
            value=total_value_inr,
            footnote="Cumulative value across filtered operational records.",
            badge_text="INR (₹)",
            badge_variant="INFO",
        )
    with m5:
        if "_is_fraud_bool" in view_df.columns:
            gt_fraud = int(view_df["_is_fraud_bool"].sum())
            render_metric_card(
                label="Ground-Truth (is_fraud)",
                value=f"{gt_fraud:,}",
                footnote="Dataset label (is_fraud) — distinct from review flags & ML predictions.",
                badge_text="LABEL",
                badge_variant="UNKNOWN",
            )
        else:
            render_metric_card(
                label="Ground-Truth (is_fraud)",
                value="Not available",
                footnote="Column is_fraud not present in operational CSV.",
                badge_text="N/A",
                badge_variant="UNKNOWN",
            )

    st.markdown("<div style='height:0.65rem;'></div>", unsafe_allow_html=True)

    # Concept Distinction Callout Banner
    st.markdown(
        """
        <div class="af-inner-box" style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
            <span style="font-size:0.81rem; color:#111C18;">
                <strong>Semantic Metric Distinction:</strong>
                <code>combined_risk_level</code> reflects multi-signal pipeline severity ·
                <code>needs_human_review</code> indicates investigator queue escalation ·
                <code>is_fraud</code> is the synthetic dataset ground-truth label ·
                <code>predicted_fraud</code> belongs strictly to the separate ML Experiment tab.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if view_df.empty:
        st.info("No transactions match the current Executive Overview filters.")
        return

    # Bento Charts Row 1: Volume by Risk Level + Value by Risk Level
    fig_vol, fig_val = build_risk_volume_and_value_charts(view_df)
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown(
            """
            <div class="af-panel">
                <div class="af-panel-header">
                    <h3 class="af-panel-title">Transaction Volume by Risk Level</h3>
                    <span class="af-panel-caption">Count of transactions per combined_risk_level</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        if fig_vol is not None:
            st.plotly_chart(fig_vol, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with c_right:
        st.markdown(
            """
            <div class="af-panel">
                <div class="af-panel-header">
                    <h3 class="af-panel-title">Transaction Value (₹) by Risk Level</h3>
                    <span class="af-panel-caption">Total ₹ exposure per combined_risk_level</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        if fig_val is not None:
            st.plotly_chart(fig_val, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # Bento Charts Row 2: Recorded Signal Counts + Scenario Breakdown
    fig_sig = build_signal_counts_chart(view_df)
    fig_scen = build_scenario_distribution_chart(view_df)
    b_left, b_right = st.columns(2)
    with b_left:
        st.markdown(
            """
            <div class="af-panel">
                <div class="af-panel-header">
                    <h3 class="af-panel-title">Recorded Detection &amp; Escalation Signals</h3>
                    <span class="af-panel-caption">Active boolean flags across filtered view</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        if fig_sig is not None:
            st.plotly_chart(fig_sig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("Signal flag columns are not available in this dataset.")
        st.markdown("</div>", unsafe_allow_html=True)

    with b_right:
        st.markdown(
            """
            <div class="af-panel">
                <div class="af-panel-header">
                    <h3 class="af-panel-title">Scenario Distribution by Risk Level</h3>
                    <span class="af-panel-caption">Recorded scenario column breakdown</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        if fig_scen is not None:
            st.plotly_chart(fig_scen, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("Column `scenario` is not available or contains no values.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Priority & Recent Transactions Card Grid (6 cards)
    st.markdown(
        """
        <div class="af-panel-header" style="margin-top:0.5rem;">
            <div>
                <div class="af-eyebrow">Immediate Triage</div>
                <h3 class="af-panel-title">Priority Transactions (High-Risk-First Queue Preview)</h3>
            </div>
            <span class="af-panel-caption">Select any card to open its full docket on the Investigation Desk</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    preview_rows = view_df.head(6)
    cols = st.columns(3)
    for idx, (_, row) in enumerate(preview_rows.iterrows()):
        with cols[idx % 3]:
            render_transaction_card(
                row=row,
                button_key_prefix="exec_card",
                on_open_callback=on_select_tx,
            )

    # Embedded `components.html` Live HTML/JS Widget Option
    with st.expander("Interactive HTML/JS Triage Widget (`components.html` Live Client-Side Inspector)", expanded=False):
        render_interactive_html_widget(view_df, height=560)


# =============================================================================
# 9. PAGE 2 — INVESTIGATION DESK
# =============================================================================

def render_investigation_detail_panel(
    selected_row: pd.Series,
    full_df: pd.DataFrame,
    on_close_detail: Any,
    on_select_tx: Any,
) -> None:
    """
    Render the complete transaction investigation docket for a selected transaction:
    summary, recorded reasons, signal checklist, and chronological sender history.
    """
    tx_id = str(selected_row.get("transaction_id", "Not available"))
    ts_str = format_timestamp(selected_row.get("timestamp"))
    amt_str = format_inr(selected_row.get("amount"))
    sender_id = str(selected_row.get("sender_id", "Not available"))
    receiver_id = str(selected_row.get("receiver_id", "Not available"))

    raw_level = (
        selected_row.get("combined_risk_level")
        if pd.notna(selected_row.get("combined_risk_level"))
        else selected_row.get("risk_level", "UNKNOWN")
    )
    risk_badge = render_risk_badge_html(raw_level)

    needs_review = parse_bool_value(selected_row.get("needs_human_review"))
    review_badge = (
        render_status_pill_html("Needs Human Review: YES", "HIGH")
        if needs_review
        else render_status_pill_html("Needs Human Review: NO", "LOW")
    )

    gt_badge = ""
    if "is_fraud" in selected_row.index and pd.notna(selected_row.get("is_fraud")):
        is_f = parse_bool_value(selected_row.get("is_fraud"))
        gt_badge = render_status_pill_html(
            f"Dataset Label (is_fraud): {'True' if is_f else 'False'}",
            "CRITICAL" if is_f else "UNKNOWN",
        )

    top_bar_l, top_bar_r = st.columns([3, 1])
    with top_bar_l:
        st.markdown('<div class="af-eyebrow">Active Investigation Docket</div>', unsafe_allow_html=True)
    with top_bar_r:
        st.button(
            "← Return to Full Queue (Keep Filters)",
            key=f"back_btn_{tx_id}",
            use_container_width=True,
            on_click=on_close_detail,
        )

    # Summary Header Panel
    c_score = selected_row.get("combined_risk_score")
    b_score = selected_row.get("risk_score")
    beh_pts = selected_row.get("behavior_risk_points")
    prof_pts = selected_row.get("profile_risk_points")

    score_breakdown_parts = []
    if c_score is not None and pd.notna(c_score):
        score_breakdown_parts.append(f"Combined Score: {float(c_score):.1f}")
    if b_score is not None and pd.notna(b_score):
        score_breakdown_parts.append(f"Base Rule Score: {float(b_score):.1f}")
    if beh_pts is not None and pd.notna(beh_pts):
        score_breakdown_parts.append(f"Behavior Points: {float(beh_pts):.1f}")
    if prof_pts is not None and pd.notna(prof_pts):
        score_breakdown_parts.append(f"Profile Points: {float(prof_pts):.1f}")
    score_breakdown_str = " · ".join(score_breakdown_parts) or "Score breakdown not available"

    st.markdown(
        f"""
        <div class="af-panel" style="border-top: 3px solid #0F3D2E;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:0.75rem;">
                <div>
                    <span class="af-mono-id" style="font-size:0.95rem;">{html.escape(tx_id)}</span>
                    <span class="af-mono-ts" style="margin-left:0.65rem;">Timestamp: {html.escape(ts_str)}</span>
                    <div class="af-tx-amount" style="font-size:1.85rem; margin-top:0.45rem;">{html.escape(amt_str)}</div>
                </div>
                <div style="display:flex; flex-wrap:wrap; gap:0.45rem; align-items:center;">
                    {risk_badge}
                    {review_badge}
                    {gt_badge}
                </div>
            </div>
            <div class="af-tx-route" style="margin-top:0.5rem;">
                <span><strong>Sender:</strong> <span class="af-tx-party">{html.escape(sender_id)}</span></span>
                <span style="color:#74857F; margin:0 0.5rem;">→</span>
                <span><strong>Receiver:</strong> <span class="af-tx-party">{html.escape(receiver_id)}</span></span>
                <span style="margin-left:auto; font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#52635E;">
                    {html.escape(score_breakdown_str)}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_reasons, col_signals = st.columns([1.15, 1.0])

    # Left Column: Recorded Detection & Review Reasons
    with col_reasons:
        st.markdown(
            """
            <div class="af-panel">
                <div class="af-panel-header">
                    <h3 class="af-panel-title">Recorded Detection &amp; Escalation Reasons</h3>
                    <span class="af-panel-caption">Parsed directly from pipeline output columns</span>
                </div>
            """,
            unsafe_allow_html=True,
        )

        reason_sources = [
            ("Primary Rule Reasons (`reasons`)", "reasons"),
            ("Behavioral Analysis (`behavior_reasons`)", "behavior_reasons"),
            ("Customer Profile Review (`profile_review_reason`)", "profile_review_reason"),
            ("Location Anomaly (`location_reason`)", "location_reason"),
            ("Network Telemetry (`network_reason`)", "network_reason"),
            ("Additional Escalation (`additional_review_reasons`)", "additional_review_reasons"),
        ]

        any_reason_rendered = False
        for section_label, col_name in reason_sources:
            if col_name in selected_row.index:
                items = parse_explanation_items(selected_row.get(col_name))
                if items:
                    any_reason_rendered = True
                    st.markdown(
                        f'<div class="af-eyebrow" style="margin-top:0.55rem;">{html.escape(section_label)}</div>',
                        unsafe_allow_html=True,
                    )
                    for item in items:
                        st.markdown(
                            f'<div class="af-reason-item">{html.escape(item)}</div>',
                            unsafe_allow_html=True,
                        )

        if not any_reason_rendered:
            st.markdown(
                '<div class="af-inner-box" style="color:#52635E;">No elevated detection or review reason strings were recorded for this transaction.</div>',
                unsafe_allow_html=True,
            )

        # Customer Profile & Contextual Metadata
        st.markdown(
            '<div class="af-eyebrow" style="margin-top:0.9rem;">Recorded Customer Profile &amp; Context</div>',
            unsafe_allow_html=True,
        )
        cust_type = selected_row.get("customer_type", "Not available")
        kyc_stat = selected_row.get("kyc_status", "Not available")
        exp_outflow = (
            format_inr(selected_row.get("expected_monthly_outflow"))
            if pd.notna(selected_row.get("expected_monthly_outflow"))
            else "Not available"
        )
        outflow_rat = (
            f"{float(selected_row.get('outflow_ratio')):.2f}x"
            if pd.notna(selected_row.get("outflow_ratio"))
            else "Not available"
        )
        usual_cnt = (
            str(selected_row.get("usual_transaction_count"))
            if pd.notna(selected_row.get("usual_transaction_count"))
            else "Not available"
        )
        loc_id = selected_row.get("location_id", "Not available")
        net_type = selected_row.get("network_type", "Not available")

        st.markdown(
            f"""
            <div class="af-inner-box" style="font-size:0.82rem; line-height:1.65;">
                <strong>Customer Type:</strong> {html.escape(str(cust_type))} &nbsp;·&nbsp;
                <strong>KYC Status:</strong> {html.escape(str(kyc_stat))} &nbsp;·&nbsp;
                <strong>Usual Tx Count:</strong> {html.escape(usual_cnt)}<br>
                <strong>Expected Monthly Outflow:</strong> {html.escape(exp_outflow)} &nbsp;·&nbsp;
                <strong>Recorded Outflow Ratio:</strong> {html.escape(outflow_rat)}<br>
                <strong>Location ID:</strong> <code>{html.escape(str(loc_id))}</code> &nbsp;·&nbsp;
                <strong>Network Type:</strong> <code>{html.escape(str(net_type))}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Right Column: Signal Checklist & Advisory Notice
    with col_signals:
        st.markdown(
            """
            <div class="af-panel">
                <div class="af-panel-header">
                    <h3 class="af-panel-title">Recorded Signal Checklist</h3>
                    <span class="af-panel-caption">Active vs. inactive detection flags</span>
                </div>
            """,
            unsafe_allow_html=True,
        )

        checklist_specs = [
            ("Rapid Transfer", "rapid_transfer_flag", "HIGH"),
            ("New Recipient", "new_recipient_flag", "MEDIUM"),
            ("Location Anomaly", "location_anomaly_flag", "HIGH"),
            ("Profile Review Required", "profile_review_required", "MEDIUM"),
            ("Network Review Signal", "network_review_signal", "MEDIUM"),
            ("VPN Detected", "vpn_detected", "INFO"),
            ("Proxy Detected", "proxy_detected", "HIGH"),
            ("Tor Detected", "tor_detected", "CRITICAL"),
            ("Hosting Network", "hosting_network_detected", "HIGH"),
        ]

        tiles_html = '<div class="af-signal-grid">'
        for label, col, active_variant in checklist_specs:
            if col in selected_row.index:
                is_active = parse_bool_value(selected_row.get(col))
                badge = (
                    render_status_pill_html("ACTIVE", active_variant)
                    if is_active
                    else render_status_pill_html("INACTIVE", "LOW")
                )
                bg_tile = "#FFFFFF" if is_active else "#F8FAF9"
            else:
                badge = render_status_pill_html("N/A", "UNKNOWN")
                bg_tile = "#F8FAF9"

            tiles_html += (
                f'<div class="af-signal-tile" style="background:{bg_tile};">'
                f'<span>{html.escape(label)}</span>{badge}</div>'
            )
        tiles_html += "</div>"
        st.markdown(tiles_html, unsafe_allow_html=True)

        # Location telemetry details when present
        dist_km = selected_row.get("location_distance_km")
        elapsed_min = selected_row.get("location_elapsed_minutes")
        if pd.notna(dist_km) or pd.notna(elapsed_min):
            dist_str = f"{float(dist_km):.1f} km" if pd.notna(dist_km) else "Not available"
            elap_str = f"{float(elapsed_min):.1f} min" if pd.notna(elapsed_min) else "Not available"
            st.markdown(
                f"""
                <div class="af-inner-box" style="margin-top:0.75rem; font-size:0.81rem;">
                    <strong>Location Telemetry:</strong> Distance from previous event:
                    <code>{html.escape(dist_str)}</code> · Elapsed time: <code>{html.escape(elap_str)}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Advisory notes (VPN/proxy notice + Human-in-the-loop safety)
        st.markdown(
            """
            <div class="af-inner-box" style="margin-top:0.75rem; border-left:3px solid #2563EB; font-size:0.8rem; color:#52635E;">
                <strong>Contextual Signal Notice:</strong> Anonymizer or network indicators
                (such as a VPN, proxy, or hosting ASN) are contextual risk signals and do not
                by themselves prove fraudulent intent.<br><br>
                <strong>Human-in-the-Loop Decision Support:</strong> AntiFraudster is an analytical
                investigation prototype. It does not automatically freeze accounts, block payments,
                whitelist customers, or file regulatory SARs.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Related Transaction History for the Same Sender
    st.markdown(
        f"""
        <div class="af-panel">
            <div class="af-panel-header">
                <h3 class="af-panel-title">Related Sender Activity — <code>{html.escape(sender_id)}</code></h3>
                <span class="af-panel-caption">
                    Other recorded transactions from the same sender (prior/concurrent events distinguished from later batch records)
                </span>
            </div>
        """,
        unsafe_allow_html=True,
    )

    if "sender_id" in full_df.columns:
        same_sender = full_df[
            (full_df["sender_id"].astype(str) == sender_id)
            & (full_df["transaction_id"].astype(str) != tx_id)
        ].copy()
        if same_sender.empty:
            st.caption(f"No other transactions recorded for sender `{sender_id}` in this dataset.")
        else:
            same_sender = same_sender.sort_values(by="_timestamp_dt", ascending=True, na_position="last")
            cur_dt = selected_row.get("_timestamp_dt")
            rel_cols = st.columns(min(3, len(same_sender)))
            for idx, (_, srow) in enumerate(same_sender.head(6).iterrows()):
                s_tx_id = str(srow.get("transaction_id", "N/A"))
                s_ts = format_timestamp(srow.get("timestamp"))
                s_amt = format_inr(srow.get("amount"))
                s_rec = str(srow.get("receiver_id", "N/A"))
                s_lvl = (
                    srow.get("combined_risk_level")
                    if pd.notna(srow.get("combined_risk_level"))
                    else srow.get("risk_level", "UNKNOWN")
                )
                s_dt = srow.get("_timestamp_dt")
                if pd.notna(cur_dt) and pd.notna(s_dt):
                    temporal_tag = (
                        "Prior / Concurrent Historical Record"
                        if s_dt <= cur_dt
                        else "Later Batch Record (Not used for real-time decision)"
                    )
                else:
                    temporal_tag = "Recorded in Batch"

                with rel_cols[idx % len(rel_cols)]:
                    st.markdown(
                        f"""
                        <div class="af-inner-box">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem;">
                                <span class="af-mono-id">{html.escape(s_tx_id)}</span>
                                {render_risk_badge_html(s_lvl)}
                            </div>
                            <div style="font-weight:700; font-size:1.05rem; margin-bottom:0.2rem;">{html.escape(s_amt)}</div>
                            <div style="font-size:0.78rem; color:#52635E;">
                                To: <code>{html.escape(s_rec)}</code><br>
                                Time: <code>{html.escape(s_ts)}</code><br>
                                <em>{html.escape(temporal_tag)}</em>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.button(
                        f"Open {s_tx_id}",
                        key=f"rel_sender_{tx_id}_{s_tx_id}",
                        use_container_width=True,
                        on_click=on_select_tx,
                        args=(s_tx_id,),
                    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_page_investigation_desk(
    df_op: Optional[pd.DataFrame],
    op_err: Optional[str],
    op_mtime: Optional[datetime],
    root: Path,
    on_select_tx: Any,
    on_close_detail: Any,
) -> None:
    """Render Page 2: 02 – Investigation Desk."""
    freshness = (
        f"File saved {op_mtime.strftime('%Y-%m-%d %H:%M:%S')}"
        if op_mtime is not None
        else "Unavailable"
    )
    render_page_header(
        title="02 — Investigation Desk & High-Risk Triage Queue",
        subtitle=(
            "Search, filter, sort, and inspect individual transactions with full recorded "
            "detection explanations, signal checklists, and chronological sender history."
        ),
        freshness_text=freshness,
        source_badge="Interactive Investigation Queue",
    )

    if df_op is None or df_op.empty:
        render_missing_data_message(
            "data/antifraudster_results.csv",
            root,
            op_err or "The operational results dataset is empty.",
        )
        return

    # Filter & Search Controls Panel (Preserved in st.session_state)
    min_amt_val = float(df_op["_amount_num"].min()) if not df_op.empty else 0.0
    max_amt_val = float(df_op["_amount_num"].max()) if not df_op.empty else 100000.0
    if max_amt_val <= min_amt_val:
        max_amt_val = min_amt_val + 1000.0

    st.markdown('<div class="af-panel">', unsafe_allow_html=True)
    st.markdown('<div class="af-eyebrow">Queue Search, Filters &amp; Sorting</div>', unsafe_allow_html=True)
    r1c1, r1c2, r1c3, r1c4 = st.columns([1.8, 1.5, 1.2, 1.3])
    with r1c1:
        inv_search = st.text_input(
            "Search Transaction ID, Sender ID, or Receiver ID",
            value=st.session_state.get("inv_search", ""),
            placeholder="Type ID and press Enter...",
            key="inv_search_input",
        )
        st.session_state["inv_search"] = inv_search
    with r1c2:
        inv_risk_levels = st.multiselect(
            "Risk Levels",
            options=RISK_ORDER,
            default=st.session_state.get("inv_risk_levels", RISK_ORDER),
            key="inv_risk_levels_input",
        )
        st.session_state["inv_risk_levels"] = inv_risk_levels
    with r1c3:
        inv_review = st.selectbox(
            "Human Review Filter",
            options=["ALL", "NEEDS_REVIEW", "NO_REVIEW"],
            format_func=lambda x: {
                "ALL": "All Statuses",
                "NEEDS_REVIEW": "Needs Human Review",
                "NO_REVIEW": "Standard Queue",
            }[x],
            key="inv_review_input",
        )
    with r1c4:
        inv_sort = st.selectbox(
            "Sort Queue By",
            options=["RISK_DESC", "AMOUNT_DESC", "TIME_DESC", "RISK_ASC", "AMOUNT_ASC", "TIME_ASC"],
            format_func=lambda x: {
                "RISK_DESC": "Risk Score (High → Low)",
                "AMOUNT_DESC": "Amount ₹ (High → Low)",
                "TIME_DESC": "Timestamp (Newest → Oldest)",
                "RISK_ASC": "Risk Score (Low → High)",
                "AMOUNT_ASC": "Amount ₹ (Low → High)",
                "TIME_ASC": "Timestamp (Oldest → Newest)",
            }[x],
            key="inv_sort_input",
        )

    r2c1, r2c2, r2c3 = st.columns([2.4, 1.3, 1.1])
    with r2c1:
        inv_amt_range = st.slider(
            "Transaction Amount Range (₹)",
            min_value=float(math.floor(min_amt_val)),
            max_value=float(math.ceil(max_amt_val)),
            value=(float(math.floor(min_amt_val)), float(math.ceil(max_amt_val))),
            key="inv_amt_slider",
        )
    with r2c2:
        inv_signal = st.selectbox(
            "Signal Filter",
            options=[
                "ALL",
                "RAPID_TRANSFER",
                "NEW_RECIPIENT",
                "LOCATION_ANOMALY",
                "PROFILE_REVIEW",
                "NETWORK_SIGNAL",
            ],
            format_func=lambda x: {
                "ALL": "All Signals",
                "RAPID_TRANSFER": "Rapid Transfer Flag",
                "NEW_RECIPIENT": "New Recipient Flag",
                "LOCATION_ANOMALY": "Location Anomaly Flag",
                "PROFILE_REVIEW": "Profile Review Required",
                "NETWORK_SIGNAL": "Network Review Signal",
            }[x],
            key="inv_signal_input",
        )
    with r2c3:
        cards_per_page = st.selectbox(
            "Cards per Page",
            options=[6, 9, 12, 18],
            index=1,
            key="inv_page_size",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    filtered_df = filter_and_sort_transactions(
        df_op,
        search_query=inv_search,
        selected_risk_levels=inv_risk_levels,
        amount_range=inv_amt_range,
        review_filter=inv_review,
        signal_filter=inv_signal,
        sort_by=inv_sort,
    )

    # Check if an individual transaction is currently open for inspection
    selected_tx_id = st.session_state.get("selected_tx_id")
    if selected_tx_id:
        match_rows = df_op[df_op["transaction_id"].astype(str) == str(selected_tx_id)]
        if not match_rows.empty:
            render_investigation_detail_panel(
                selected_row=match_rows.iloc[0],
                full_df=df_op,
                on_close_detail=on_close_detail,
                on_select_tx=on_select_tx,
            )
            st.markdown("<hr style='border:none; border-top:1px solid #E1E8E4; margin:1.25rem 0;'>", unsafe_allow_html=True)

    # Render Paginated Transaction Card Queue
    total_matches = len(filtered_df)
    if total_matches == 0:
        st.markdown(
            """
            <div class="af-panel" style="text-align:center; padding:2.2rem;">
                <div class="af-eyebrow">Empty Queue State</div>
                <h3 class="af-panel-title">No transactions match your current filter criteria</h3>
                <p style="font-size:0.86rem; color:#52635E;">
                    Try widening the ₹ amount slider, enabling additional risk levels, or clearing the search box.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    total_pages = max(1, math.ceil(total_matches / int(cards_per_page)))
    current_page = int(st.session_state.get("inv_page_num", 1))
    if current_page > total_pages:
        current_page = 1
        st.session_state["inv_page_num"] = 1

    pcol1, pcol2, pcol3 = st.columns([2.2, 1.3, 1.3])
    with pcol1:
        st.markdown(
            f"**Showing {(current_page - 1) * cards_per_page + 1}–{min(current_page * cards_per_page, total_matches)} "
            f"of {total_matches:,} matching transactions** (Page {current_page} of {total_pages})"
        )
    with pcol2:
        if st.button("← Previous Page", disabled=(current_page <= 1), use_container_width=True):
            st.session_state["inv_page_num"] = max(1, current_page - 1)
            st.rerun()
    with pcol3:
        if st.button("Next Page →", disabled=(current_page >= total_pages), use_container_width=True):
            st.session_state["inv_page_num"] = min(total_pages, current_page + 1)
            st.rerun()

    start_idx = (current_page - 1) * int(cards_per_page)
    end_idx = start_idx + int(cards_per_page)
    page_df = filtered_df.iloc[start_idx:end_idx]

    grid_cols = st.columns(3)
    for idx, (_, row) in enumerate(page_df.iterrows()):
        with grid_cols[idx % 3]:
            render_transaction_card(
                row=row,
                button_key_prefix=f"inv_q_p{current_page}",
                on_open_callback=on_select_tx,
            )

    # Embedded `components.html` Live HTML/JS Widget Option
    with st.expander("Interactive HTML/JS Queue Widget (`components.html` Live Client-Side Inspector)", expanded=False):
        render_interactive_html_widget(filtered_df, height=580)

    # Optional secondary compact table view for advanced data inspection
    with st.expander("Secondary Compact Table View (Advanced Analyst Inspection)", expanded=False):
        display_cols = [c for c in df_op.columns if not c.startswith("_")]
        st.dataframe(filtered_df[display_cols], use_container_width=True, height=320)


# =============================================================================
# 10. PAGE 3 — BEHAVIOR & NETWORK
# =============================================================================

def render_page_behavior_and_network(
    df_op: Optional[pd.DataFrame],
    op_err: Optional[str],
    op_mtime: Optional[datetime],
    root: Path,
    on_select_tx: Any,
) -> None:
    """Render Page 3: 03 – Behavior & Network Intelligence."""
    freshness = (
        f"File saved {op_mtime.strftime('%Y-%m-%d %H:%M:%S')}"
        if op_mtime is not None
        else "Unavailable"
    )
    render_page_header(
        title="03 — Behavior, Profile, Location & Network Signals",
        subtitle=(
            "Explore recorded behavioral patterns, customer profile outflow anomalies, "
            "location velocity telemetry, network anonymizer signals, and sender-to-receiver corridors."
        ),
        freshness_text=freshness,
        source_badge="Behavioral & Network Telemetry",
    )

    if df_op is None or df_op.empty:
        render_missing_data_message(
            "data/antifraudster_results.csv",
            root,
            op_err or "The operational results dataset is empty.",
        )
        return

    # Signal KPI Summary Row
    rapid_cnt = int(df_op["_rapid_transfer_flag_bool"].sum()) if "_rapid_transfer_flag_bool" in df_op.columns else 0
    new_rec_cnt = int(df_op["_new_recipient_flag_bool"].sum()) if "_new_recipient_flag_bool" in df_op.columns else 0
    loc_cnt = int(df_op["_location_anomaly_flag_bool"].sum()) if "_location_anomaly_flag_bool" in df_op.columns else 0
    prof_cnt = int(df_op["_profile_review_required_bool"].sum()) if "_profile_review_required_bool" in df_op.columns else 0
    net_cnt = int(df_op["_network_review_signal_bool"].sum()) if "_network_review_signal_bool" in df_op.columns else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_metric_card("Rapid Transfers", f"{rapid_cnt:,}", "Flagged via rapid_transfer_flag", "BEHAVIOR", "HIGH" if rapid_cnt else "LOW")
    with k2:
        render_metric_card("New Recipients", f"{new_rec_cnt:,}", "Flagged via new_recipient_flag", "BEHAVIOR", "MEDIUM" if new_rec_cnt else "LOW")
    with k3:
        render_metric_card("Location Anomalies", f"{loc_cnt:,}", "Flagged via location_anomaly_flag", "GEO-VELOCITY", "HIGH" if loc_cnt else "LOW")
    with k4:
        render_metric_card("Profile Reviews", f"{prof_cnt:,}", "Flagged via profile_review_required", "KYC / OUTFLOW", "MEDIUM" if prof_cnt else "LOW")
    with k5:
        render_metric_card("Network Signals", f"{net_cnt:,}", "Flagged via network_review_signal", "NETWORK", "HIGH" if net_cnt else "LOW")

    st.markdown("<div style='height:0.65rem;'></div>", unsafe_allow_html=True)

    # Clear distinction notice between recorded signals and confirmed fraud events
    st.markdown(
        """
        <div class="af-inner-box" style="border-left:3px solid #0F3D2E; font-size:0.81rem; color:#111C18;">
            <strong>Signal vs. Confirmed Fraud Distinction:</strong> Each indicator below represents a
            recorded rule, velocity, profile, or network review signal. An active signal elevates
            investigation priority (<code>needs_human_review</code> / <code>combined_risk_score</code>)
            and is <strong>not</strong> equivalent to a confirmed fraud event.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Interactive Sender-to-Receiver Corridor Graph & Repetition Table
    g_col, r_col = st.columns([1.45, 1.0])
    with g_col:
        st.markdown(
            """
            <div class="af-panel">
                <div class="af-panel-header">
                    <h3 class="af-panel-title">Sender-to-Receiver Transfer Corridors</h3>
                    <span class="af-panel-caption">Actual recorded sender_id → receiver_id edges colored by peak risk</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        fig_net = build_sender_receiver_network_figure(df_op)
        if fig_net is not None:
            st.plotly_chart(fig_net, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("Sender/Receiver columns are not available to construct corridor graph.")
        st.markdown("</div>", unsafe_allow_html=True)

    with r_col:
        st.markdown(
            """
            <div class="af-panel">
                <div class="af-panel-header">
                    <h3 class="af-panel-title">Repeated Sender → Receiver Pairs</h3>
                    <span class="af-panel-caption">Corridors with highest transaction frequency</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        if "sender_id" in df_op.columns and "receiver_id" in df_op.columns:
            corridors = (
                df_op.groupby(["sender_id", "receiver_id"], as_index=False)
                .agg(
                    Transactions=("transaction_id", "count"),
                    Total_Amount_INR=("_amount_num", "sum"),
                    Max_Risk_Score=("_risk_score_num", "max"),
                )
                .sort_values(by=["Transactions", "Max_Risk_Score"], ascending=[False, False])
                .head(8)
            )
            corridors["Total_Amount_INR"] = corridors["Total_Amount_INR"].map(format_inr)
            st.dataframe(corridors, use_container_width=True, hide_index=True, height=345)
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 2: Network & Anonymizer Signals vs. Location Velocity & Customer Profile
    n_col, l_col = st.columns(2)
    with n_col:
        st.markdown(
            """
            <div class="af-panel">
                <div class="af-panel-header">
                    <h3 class="af-panel-title">Network Type &amp; Anonymizer Telemetry</h3>
                    <span class="af-panel-caption">VPN, Proxy, Tor, and Hosting ASN indicators</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        anon_specs = [
            ("VPN Detected (`vpn_detected`)", "_vpn_detected_bool", "#2563EB"),
            ("Proxy Detected (`proxy_detected`)", "_proxy_detected_bool", "#EA580C"),
            ("Tor Exit Node (`tor_detected`)", "_tor_detected_bool", "#DC2626"),
            ("Hosting Network (`hosting_network_detected`)", "_hosting_network_detected_bool", "#D97706"),
        ]
        a_labels, a_counts, a_colors = [], [], []
        for lbl, col, clr in anon_specs:
            if col in df_op.columns:
                a_labels.append(lbl)
                a_counts.append(int(df_op[col].sum()))
                a_colors.append(clr)

        if go is not None and a_labels:
            fig_anon = go.Figure(
                data=[
                    go.Bar(
                        x=a_labels,
                        y=a_counts,
                        text=[f"{c:,}" for c in a_counts],
                        textposition="outside",
                        marker=dict(color=a_colors, cornerradius=6),
                    )
                ]
            )
            _apply_clean_plotly_layout(fig_anon, height=260, y_title="Transactions")
            st.plotly_chart(fig_anon, use_container_width=True, config={"displayModeBar": False})

        if "network_type" in df_op.columns:
            net_dist = (
                df_op["network_type"]
                .fillna("Unspecified")
                .astype(str)
                .value_counts()
                .reset_index()
            )
            net_dist.columns = ["Network Type", "Transactions"]
            st.dataframe(net_dist, use_container_width=True, hide_index=True, height=160)
        st.markdown("</div>", unsafe_allow_html=True)

    with l_col:
        st.markdown(
            """
            <div class="af-panel">
                <div class="af-panel-header">
                    <h3 class="af-panel-title">Location Anomaly &amp; Customer Profile Outflow</h3>
                    <span class="af-panel-caption">Geo-distance vs. elapsed minutes and profile review triggers</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        if (
            go is not None
            and "location_distance_km" in df_op.columns
            and "location_elapsed_minutes" in df_op.columns
        ):
            loc_df = df_op.copy()
            loc_df["_dist"] = pd.to_numeric(loc_df["location_distance_km"], errors="coerce")
            loc_df["_elap"] = pd.to_numeric(loc_df["location_elapsed_minutes"], errors="coerce")
            loc_valid = loc_df.dropna(subset=["_dist", "_elap"])
            if not loc_valid.empty:
                fig_loc = go.Figure()
                for r_key in RISK_ORDER:
                    sub = loc_valid[loc_valid["_risk_key"] == r_key]
                    if sub.empty:
                        continue
                    sem = RISK_SEMANTICS[r_key]
                    fig_loc.add_trace(
                        go.Scatter(
                            x=sub["_elap"],
                            y=sub["_dist"],
                            mode="markers",
                            name=r_key,
                            text=sub["transaction_id"].astype(str),
                            marker=dict(size=9, color=sem["accent"], opacity=0.85),
                            hovertemplate="Tx: <b>%{text}</b><br>Elapsed: %{x:.1f} min<br>Distance: %{y:.1f} km<extra></extra>",
                        )
                    )
                _apply_clean_plotly_layout(
                    fig_loc,
                    height=260,
                    x_title="Elapsed Time Since Previous Tx (Minutes)",
                    y_title="Location Distance (km)",
                )
                st.plotly_chart(fig_loc, use_container_width=True, config={"displayModeBar": False})
            else:
                st.caption("No numeric location distance/elapsed minute pairs recorded.")
        else:
            st.caption("Location distance and elapsed time columns are not present.")

        if "kyc_status" in df_op.columns or "customer_type" in df_op.columns:
            grp_cols = [c for c in ["customer_type", "kyc_status"] if c in df_op.columns]
            prof_summary = (
                df_op.groupby(grp_cols, dropna=False, as_index=False)
                .agg(
                    Transactions=("transaction_id", "count"),
                    Avg_Risk_Score=("_risk_score_num", "mean"),
                )
            )
            prof_summary["Avg_Risk_Score"] = prof_summary["Avg_Risk_Score"].map(lambda x: f"{x:.1f}")
            st.dataframe(prof_summary, use_container_width=True, hide_index=True, height=160)
        st.markdown("</div>", unsafe_allow_html=True)

    # Human-Review Signal Queue Cards (Top 3 flagged for human review)
    if "_needs_human_review_bool" in df_op.columns:
        hr_df = df_op[df_op["_needs_human_review_bool"]].sort_values(
            by="_risk_score_num", ascending=False
        )
        if not hr_df.empty:
            st.markdown(
                '<div class="af-eyebrow">Human-Review Signal Spotlight</div>',
                unsafe_allow_html=True,
            )
            hcols = st.columns(3)
            for idx, (_, hrow) in enumerate(hr_df.head(3).iterrows()):
                with hcols[idx % 3]:
                    render_transaction_card(
                        row=hrow,
                        button_key_prefix="beh_hr_card",
                        on_open_callback=on_select_tx,
                    )


# =============================================================================
# 11. PAGE 4 — ML EXPERIMENT
# =============================================================================

def render_page_ml_experiment(
    df_ml: Optional[pd.DataFrame],
    ml_err: Optional[str],
    ml_mtime: Optional[datetime],
    root: Path,
) -> None:
    """Render Page 4: 04 – ML Experiment (Strictly Separate from Operational Pipeline)."""
    freshness = (
        f"File saved {ml_mtime.strftime('%Y-%m-%d %H:%M:%S')}"
        if ml_mtime is not None
        else "Unavailable"
    )
    render_page_header(
        title="04 — Experimental Machine-Learning Evaluation",
        subtitle=(
            "Offline evaluation of experimental model predictions (predicted_fraud & fraud_probability) "
            "against synthetic ground-truth labels (actual_fraud / is_fraud)."
        ),
        freshness_text=freshness,
        source_badge="Separate ML Evaluation · data/ml_evaluation_results.csv",
    )

    # Mandatory Synthetic / Experimental Evaluation Disclaimer
    st.markdown(
        """
        <div class="af-panel" style="border-left:4px solid #2563EB; background:#F8FAFF;">
            <div class="af-eyebrow" style="color:#1D4ED8;">Experimental &amp; Synthetic Data Notice</div>
            <p style="font-size:0.86rem; color:#111C18; margin:0;">
                <strong>Strict Separation from Operational Pipeline:</strong> The metrics and predictions on
                this page come exclusively from <code>data/ml_evaluation_results.csv</code>. Because the training
                and test sets are synthetic/experimental hackathon datasets, these classification scores
                <strong>do not represent real-world production banking performance</strong> and must never be
                interpreted as a guarantee of live fraud prevention.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df_ml is None or df_ml.empty:
        render_missing_data_message(
            "data/ml_evaluation_results.csv",
            root,
            ml_err or "The ML evaluation dataset is empty.",
        )
        return

    metrics = compute_ml_evaluation_metrics(df_ml)
    prepared_ml = prepare_ml_dataframe(df_ml)

    if not metrics["available"]:
        st.warning(metrics["reason"])
        return

    # Row 1: Classification Metric Cards (Accuracy, Precision, Recall, F1)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card(
            "Accuracy",
            f"{metrics['accuracy'] * 100:.2f}%",
            f"(TP + TN) / {metrics['total']:,} test records",
            "METRIC",
            "INFO",
        )
    with m2:
        render_metric_card(
            "Precision",
            f"{metrics['precision'] * 100:.2f}%",
            f"TP ({metrics['tp']}) / (TP + FP ({metrics['tp'] + metrics['fp']}))",
            "METRIC",
            "INFO",
        )
    with m3:
        render_metric_card(
            "Recall",
            f"{metrics['recall'] * 100:.2f}%",
            f"TP ({metrics['tp']}) / (TP + FN ({metrics['tp'] + metrics['fn']}))",
            "METRIC",
            "INFO",
        )
    with m4:
        render_metric_card(
            "F1 Score",
            f"{metrics['f1'] * 100:.2f}%",
            "Harmonic mean of Precision & Recall",
            "METRIC",
            "INFO",
        )

    st.markdown("<div style='height:0.65rem;'></div>", unsafe_allow_html=True)

    # Row 2: Confusion Matrix Counts (TP, FP, TN, FN)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("True Positives (TP)", f"{metrics['tp']:,}", "Actual Fraud = True · Predicted = True", "TP", "LOW")
    with c2:
        render_metric_card("False Positives (FP)", f"{metrics['fp']:,}", "Actual Fraud = False · Predicted = True", "FP", "MEDIUM" if metrics["fp"] else "LOW")
    with c3:
        render_metric_card("True Negatives (TN)", f"{metrics['tn']:,}", "Actual Fraud = False · Predicted = False", "TN", "INFO")
    with c4:
        render_metric_card("False Negatives (FN)", f"{metrics['fn']:,}", "Actual Fraud = True · Predicted = False", "FN", "CRITICAL" if metrics["fn"] else "LOW")

    st.markdown("<div style='height:0.65rem;'></div>", unsafe_allow_html=True)

    # Row 3: Visual Confusion Matrix + Fraud Probability Distribution
    cm_col, prob_col = st.columns(2)
    with cm_col:
        st.markdown(
            """
            <div class="af-panel">
                <div class="af-panel-header">
                    <h3 class="af-panel-title">Confusion Matrix (Actual vs. Predicted Labels)</h3>
                    <span class="af-panel-caption">Evaluated on test split in data/ml_evaluation_results.csv</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        if go is not None:
            z_mat = [
                [metrics["tn"], metrics["fp"]],
                [metrics["fn"], metrics["tp"]],
            ]
            text_mat = [
                [f"TN: {metrics['tn']:,}", f"FP: {metrics['fp']:,}"],
                [f"FN: {metrics['fn']:,}", f"TP: {metrics['tp']:,}"],
            ]
            fig_cm = go.Figure(
                data=go.Heatmap(
                    z=z_mat,
                    x=["Predicted: Non-Fraud (False)", "Predicted: Fraud (True)"],
                    y=["Actual: Non-Fraud (False)", "Actual: Fraud (True)"],
                    text=text_mat,
                    texttemplate="<b>%{text}</b>",
                    textfont=dict(size=14),
                    colorscale=[[0.0, "#F0F4F2"], [0.5, "#BFDBFE"], [1.0, "#1D4ED8"]],
                    showscale=False,
                )
            )
            fig_cm.update_layout(
                height=300,
                margin=dict(l=16, r=16, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#FFFFFF",
                font=dict(family="Inter, sans-serif", size=12, color="#111C18"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with prob_col:
        st.markdown(
            """
            <div class="af-panel">
                <div class="af-panel-header">
                    <h3 class="af-panel-title">Fraud-Probability Distribution (fraud_probability)</h3>
                    <span class="af-panel-caption">Model probability output segmented by actual_fraud</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        if go is not None and "_prob_num" in prepared_ml.columns and prepared_ml["_prob_num"].notna().any():
            fig_prob = go.Figure()
            non_fraud_probs = prepared_ml[~prepared_ml["_actual_bool"]]["_prob_num"].dropna()
            fraud_probs = prepared_ml[prepared_ml["_actual_bool"]]["_prob_num"].dropna()
            fig_prob.add_trace(
                go.Histogram(
                    x=non_fraud_probs,
                    name="Actual Non-Fraud (False)",
                    marker_color="#2563EB",
                    opacity=0.75,
                    nbinsx=20,
                )
            )
            fig_prob.add_trace(
                go.Histogram(
                    x=fraud_probs,
                    name="Actual Fraud (True)",
                    marker_color="#DC2626",
                    opacity=0.80,
                    nbinsx=20,
                )
            )
            fig_prob.update_layout(barmode="overlay")
            _apply_clean_plotly_layout(
                fig_prob,
                height=300,
                x_title="Predicted Fraud Probability (fraud_probability)",
                y_title="Test Transactions",
            )
            st.plotly_chart(fig_prob, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("Column `fraud_probability` is not present or contains no numeric values.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Compact View of Experimental Test Predictions
    st.markdown(
        """
        <div class="af-panel">
            <div class="af-panel-header">
                <h3 class="af-panel-title">Experimental Test Predictions Inspector</h3>
                <span class="af-panel-caption">Compare actual_fraud vs. predicted_fraud and engineered history features</span>
            </div>
        """,
        unsafe_allow_html=True,
    )

    outcome_filter = st.selectbox(
        "Filter Test Predictions by Outcome",
        options=["ALL", "TP", "FP", "FN", "TN"],
        format_func=lambda x: {
            "ALL": f"All Test Predictions ({metrics['total']:,})",
            "TP": f"True Positives Only ({metrics['tp']:,})",
            "FP": f"False Positives Only ({metrics['fp']:,})",
            "FN": f"False Negatives Only ({metrics['fn']:,})",
            "TN": f"True Negatives Only ({metrics['tn']:,})",
        }[x],
        key="ml_outcome_filter",
    )

    ml_view = prepared_ml.copy()
    if outcome_filter == "TP":
        ml_view = ml_view[ml_view["_actual_bool"] & ml_view["_predicted_bool"]]
    elif outcome_filter == "FP":
        ml_view = ml_view[~ml_view["_actual_bool"] & ml_view["_predicted_bool"]]
    elif outcome_filter == "FN":
        ml_view = ml_view[ml_view["_actual_bool"] & ~ml_view["_predicted_bool"]]
    elif outcome_filter == "TN":
        ml_view = ml_view[~ml_view["_actual_bool"] & ~ml_view["_predicted_bool"]]

    if "_prob_num" in ml_view.columns:
        ml_view = ml_view.sort_values(by="_prob_num", ascending=False, na_position="last")

    raw_ml_cols = [c for c in df_ml.columns if not c.startswith("_")]
    st.dataframe(ml_view[raw_ml_cols], use_container_width=True, height=340)
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# 12. MAIN APPLICATION ENTRYPOINT & NAVIGATION
# =============================================================================

def main() -> None:
    """Main Streamlit entrypoint for AntiFraudster."""
    if st is None:
        raise RuntimeError("Streamlit is required to run the interactive dashboard (`streamlit run app.py`).")

    st.set_page_config(
        page_title="AntiFraudster · Financial Risk Intelligence",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    # Initialize session state keys safely
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "01 — Executive Overview"
    if "selected_tx_id" not in st.session_state:
        st.session_state["selected_tx_id"] = None
    if "inv_page_num" not in st.session_state:
        st.session_state["inv_page_num"] = 1

    def open_transaction_docket(tx_id: str) -> None:
        st.session_state["selected_tx_id"] = tx_id
        st.session_state["active_page"] = "02 — Investigation Desk"

    def close_transaction_docket() -> None:
        st.session_state["selected_tx_id"] = None

    root = resolve_project_root()
    df_op_raw, op_err, op_mtime = load_dataset(root, "data/antifraudster_results.csv")
    df_ml_raw, ml_err, ml_mtime = load_dataset(root, "data/ml_evaluation_results.csv")

    df_op = prepare_operational_dataframe(df_op_raw) if df_op_raw is not None else None

    # Sidebar Navigation & Environment Inspector
    with st.sidebar:
        st.markdown(
            """
            <div style="padding:0.25rem 0 0.9rem 0; border-bottom:1px solid #E1E8E4; margin-bottom:1rem;">
                <div class="af-brand-pill" style="margin-bottom:0.35rem;">
                    <span>●</span><span>AntiFraudster</span>
                </div>
                <div style="font-size:1.05rem; font-weight:700; color:#111C18;">
                    Risk &amp; Fraud Intelligence
                </div>
                <div style="font-size:0.76rem; color:#52635E; margin-top:0.15rem;">
                    Decision-Support &amp; Investigation Prototype
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        pages = [
            "01 — Executive Overview",
            "02 — Investigation Desk",
            "03 — Behavior & Network",
            "04 — ML Experiment",
        ]
        selected_page = st.radio(
            "WORKSPACE NAVIGATION",
            options=pages,
            index=pages.index(st.session_state["active_page"])
            if st.session_state["active_page"] in pages
            else 0,
        )
        if selected_page != st.session_state["active_page"]:
            st.session_state["active_page"] = selected_page
            st.rerun()

        st.markdown("<hr style='border:none; border-top:1px solid #E1E8E4; margin:1rem 0;'>", unsafe_allow_html=True)

        # Quick Status Summary in Sidebar
        if df_op is not None and not df_op.empty:
            hr_total = (
                int(df_op["_needs_human_review_bool"].sum())
                if "_needs_human_review_bool" in df_op.columns
                else 0
            )
            crit_total = int((df_op["_risk_key"] == "CRITICAL").sum())
            st.markdown(
                f"""
                <div class="af-inner-box" style="font-size:0.78rem;">
                    <div class="af-eyebrow">Operational Snapshot</div>
                    <strong>Total Records:</strong> {len(df_op):,}<br>
                    <strong>Critical Risk:</strong> {crit_total:,}<br>
                    <strong>Human Review Queue:</strong> {hr_total:,}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("Repository & Data Schema Inspector", expanded=False):
            env_info = inspect_project_environment(root)
            st.caption(f"Project root: `{env_info['root']}`")
            for rel_path, exists in env_info["file_status"].items():
                icon = "✅" if exists else "⚪"
                st.markdown(f"{icon} `{rel_path}`")
            if df_op_raw is not None:
                st.caption(f"Operational CSV columns ({len(df_op_raw.columns)}):")
                st.code(", ".join(df_op_raw.columns.tolist()), language="text")
            if df_ml_raw is not None:
                st.caption(f"ML Evaluation CSV columns ({len(df_ml_raw.columns)}):")
                st.code(", ".join(df_ml_raw.columns.tolist()), language="text")

            for aux_name, aux_desc in AUXILIARY_DATA_FILES.items():
                aux_path = root / "data" / aux_name
                if aux_path.exists():
                    st.caption(f"• `data/{aux_name}`: {aux_desc}")

    # Route to Active Page
    active = st.session_state["active_page"]
    if active == "01 — Executive Overview":
        render_page_executive_overview(
            df_op=df_op,
            op_err=op_err,
            op_mtime=op_mtime,
            root=root,
            on_select_tx=open_transaction_docket,
        )
    elif active == "02 — Investigation Desk":
        render_page_investigation_desk(
            df_op=df_op,
            op_err=op_err,
            op_mtime=op_mtime,
            root=root,
            on_select_tx=open_transaction_docket,
            on_close_detail=close_transaction_docket,
        )
    elif active == "03 — Behavior & Network":
        render_page_behavior_and_network(
            df_op=df_op,
            op_err=op_err,
            op_mtime=op_mtime,
            root=root,
            on_select_tx=open_transaction_docket,
        )
    elif active == "04 — ML Experiment":
        render_page_ml_experiment(
            df_ml=df_ml_raw,
            ml_err=ml_err,
            ml_mtime=ml_mtime,
            root=root,
        )


if __name__ == "__main__":
    main()