"""Shared feature schema for training, the API, and the Gradio app."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "nsch_2024_youth.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "xgb_youth_mh.json"
SIDECAR_PATH = MODEL_DIR / "xgb_youth_mh_features.json"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

TARGET_COLUMN = "has_mental_health_condition"
WEIGHT_COLUMN = "FWC"

FAMILY_LEVELS: list[str] = [
    "two_parent_married",
    "two_parent_unmarried",
    "single_parent",
    "grandparent_household",
    "other",
    "missing",
]

FEATURE_COLUMNS: list[str] = [
    "SC_AGE_YEARS",
    "FPL",
    "family_structure",
    "ace_parent_divorced",
    "ace_victim_of_violence",
    "ace_treated_unfairly_race",
    "ace_items_missing",
]

ACE_COLUMNS: list[str] = [
    "ace_parent_divorced",
    "ace_victim_of_violence",
    "ace_treated_unfairly_race",
]

NUMERIC_COLUMNS: list[str] = [
    "SC_AGE_YEARS",
    "FPL",
    *ACE_COLUMNS,
    "ace_items_missing",
]


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the model matrix with XGBoost dtypes.

    Numeric columns become float so unanswered ACE items stay ``NaN``.
    ``family_structure`` is a categorical with the training level order.
    ``ace_count`` and ``FWC`` are not included.

    Args:
        frame: Table that contains :data:`FEATURE_COLUMNS`.

    Returns:
        A copy restricted to the model features.
    """
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Feature table is missing columns: {joined}")

    features = frame.loc[:, FEATURE_COLUMNS].copy()
    for column in NUMERIC_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce").astype("float64")
    features["family_structure"] = pd.Categorical(
        features["family_structure"].astype("string"),
        categories=FAMILY_LEVELS,
    )
    return features


def _require_int(value: object, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    number = int(value)
    if not low <= number <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return number


def _optional_binary(value: object, name: str) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return float("nan")
    try:
        if pd.isna(value):
            return float("nan")
    except TypeError:
        pass
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be 0, 1, or null")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be 0, 1, or null")
    number = int(value)
    if number not in (0, 1):
        raise ValueError(f"{name} must be 0, 1, or null")
    return float(number)


def validate_feature_row(payload: Mapping[str, object]) -> dict[str, object]:
    """Check one scoring payload and return cleaned Python values.

    Args:
        payload: Mapping of the seven model features. ``FWC`` is rejected.

    Returns:
        Values ready to place in a one-row feature frame. Missing ACE
        answers are ``None``.
    """
    unexpected = sorted(set(payload) - set(FEATURE_COLUMNS))
    if unexpected:
        joined = ", ".join(unexpected)
        raise ValueError(f"Unexpected features: {joined}")
    missing = [column for column in FEATURE_COLUMNS if column not in payload]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing features: {joined}")

    family = payload["family_structure"]
    if family not in FAMILY_LEVELS:
        allowed = ", ".join(FAMILY_LEVELS)
        raise ValueError(f"family_structure must be one of: {allowed}")

    ace_values = {
        column: _optional_binary(payload[column], column) for column in ACE_COLUMNS
    }
    cleaned: dict[str, object] = {
        "SC_AGE_YEARS": _require_int(payload["SC_AGE_YEARS"], "SC_AGE_YEARS", 12, 17),
        "FPL": _require_int(payload["FPL"], "FPL", 50, 400),
        "family_structure": str(family),
        "ace_items_missing": _require_int(payload["ace_items_missing"], "ace_items_missing", 0, 10),
    }
    for column, value in ace_values.items():
        cleaned[column] = None if pd.isna(value) else int(value)
    return cleaned


def feature_frame_from_row(payload: Mapping[str, object]) -> pd.DataFrame:
    """Validate one row and return it as a one-row model matrix."""
    cleaned = validate_feature_row(payload)
    return prepare_features(pd.DataFrame([cleaned]))
