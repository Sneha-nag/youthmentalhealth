"""Score one youth and return SHAP values for that row."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src.models.schema import (
    FEATURE_COLUMNS,
    MODEL_PATH,
    SIDECAR_PATH,
    feature_frame_from_row,
)


@dataclass(frozen=True)
class ShapContribution:
    """SHAP contribution of one feature for a single youth."""

    feature: str
    value: float | str | None
    shap_value: float


@dataclass(frozen=True)
class PredictionExplanation:
    """Positive-class probability and the SHAP values that add up to the log-odds."""

    probability: float
    base_value: float
    shap_values: list[ShapContribution]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready dictionary."""
        return {
            "probability": self.probability,
            "base_value": self.base_value,
            "shap_values": [
                {
                    "feature": item.feature,
                    "value": item.value,
                    "shap_value": item.shap_value,
                }
                for item in self.shap_values
            ],
        }


def _display_value(value: object) -> float | str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if number.is_integer():
            return int(number)
        return number
    if isinstance(value, (int, np.integer)):
        return int(value)
    return str(value)


def _positive_class_values(explanation: shap.Explanation) -> np.ndarray:
    values = np.asarray(explanation.values, dtype="float64")
    if values.ndim == 3:
        values = values[:, :, -1]
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return values


def _positive_base_value(explanation: shap.Explanation) -> float:
    """Return the average model output, E[f(x)], on the positive-class log-odds scale."""
    base = np.asarray(explanation.base_values, dtype="float64")
    if base.ndim >= 2:
        base = base[..., -1]
    return float(base.reshape(-1)[0])


@lru_cache(maxsize=1)
def load_model() -> tuple[xgb.XGBClassifier, shap.TreeExplainer]:
    """Load the saved booster and a tree explainer.

    Returns:
        The classifier and a SHAP explainer bound to that classifier.

    Raises:
        FileNotFoundError: The booster JSON or the feature sidecar is missing.
    """
    if not MODEL_PATH.is_file() or not SIDECAR_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run python -m src.models.train first."
        )
    sidecar = json.loads(SIDECAR_PATH.read_text(encoding="utf-8"))
    saved_features = sidecar.get("features")
    if saved_features != FEATURE_COLUMNS:
        raise ValueError("Saved feature order does not match the model schema")

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    explainer = shap.TreeExplainer(model)
    return model, explainer


def predict_and_explain(features: Mapping[str, object]) -> PredictionExplanation:
    """Return the mental-health probability and per-feature SHAP values.

    SHAP values are the log-odds contributions for
    ``has_mental_health_condition = 1``. Unanswered ACE items stay missing.

    Args:
        features: The seven predictors. ``FWC`` and ``ace_count`` are rejected.

    Returns:
        Probability on the unit interval, the average log-odds, and one SHAP row per feature.
    """
    frame = feature_frame_from_row(features)
    model, explainer = load_model()
    probability = float(model.predict_proba(frame)[0, 1])
    explanation = explainer(frame)
    shap_matrix = _positive_class_values(explanation)[0]
    contributions = [
        ShapContribution(
            feature=column,
            value=_display_value(frame.iloc[0][column]),
            shap_value=float(shap_matrix[index]),
        )
        for index, column in enumerate(FEATURE_COLUMNS)
    ]
    return PredictionExplanation(
        probability=probability,
        base_value=_positive_base_value(explanation),
        shap_values=contributions,
    )
