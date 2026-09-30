"""Train the youth mental health XGBoost model and log it to MLflow."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

from src.models.predict import predict_and_explain
from src.models.schema import (
    DATA_PATH,
    FEATURE_COLUMNS,
    FAMILY_LEVELS,
    MLFLOW_DB,
    MLRUNS_DIR,
    MODEL_DIR,
    MODEL_PATH,
    SIDECAR_PATH,
    TARGET_COLUMN,
    WEIGHT_COLUMN,
    prepare_features,
)

LOGGER = logging.getLogger(__name__)

EXPERIMENT_NAME = "nsch-youth-mh"
RANDOM_STATE = 42
TEST_SIZE = 0.2
DECISION_THRESHOLD = 0.5

MODEL_PARAMS: dict[str, object] = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5.0,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "enable_categorical": True,
    "random_state": RANDOM_STATE,
}


def _configure_tracking() -> None:
    """Use a local SQLite tracking store and a project folder for artifacts."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + MLFLOW_DB.resolve().as_posix())
    client = mlflow.tracking.MlflowClient()
    if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
        client.create_experiment(EXPERIMENT_NAME, artifact_location=MLRUNS_DIR.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)


def _shap_summary(model: xgb.XGBClassifier, features: pd.DataFrame, path: Path) -> None:
    """Write a test-set SHAP beeswarm plot."""
    explainer = shap.TreeExplainer(model)
    explanation = explainer(features)
    plt.close("all")
    shap.plots.beeswarm(explanation, show=False, max_display=len(FEATURE_COLUMNS))
    figure = plt.gcf()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def train(data_path: Path = DATA_PATH) -> dict[str, float]:
    """Fit the classifier, save it, and log metrics to MLflow.

    ``FWC`` is the survey weight used for fitting and for weighted metrics.
    It is not a predictor. ``ace_count`` is left out because it repeats the
    individual ACE items.

    Args:
        data_path: Youth feature matrix written by the ingestion module.

    Returns:
        Weighted ROC-AUC, weighted F1, unweighted ROC-AUC, and the
        survey-weighted positive prevalence on the test set.
    """
    if not data_path.is_file():
        raise FileNotFoundError(f"Feature matrix not found: {data_path}")

    frame = pd.read_parquet(data_path)
    features = prepare_features(frame)
    target = frame[TARGET_COLUMN].astype("int8")
    weights = pd.to_numeric(frame[WEIGHT_COLUMN], errors="coerce").astype("float64")
    if bool(target.isna().any()) or bool(weights.isna().any()):
        raise ValueError("Target and survey weight must be complete")

    train_x, test_x, train_y, test_y, train_w, test_w = train_test_split(
        features,
        target,
        weights,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    model = xgb.XGBClassifier(**MODEL_PARAMS)
    model.fit(train_x, train_y, sample_weight=train_w.to_numpy())

    probability = model.predict_proba(test_x)[:, 1]
    predicted = (probability >= DECISION_THRESHOLD).astype(int)
    metrics = {
        "weighted_roc_auc": float(roc_auc_score(test_y, probability, sample_weight=test_w)),
        "weighted_f1": float(f1_score(test_y, predicted, sample_weight=test_w)),
        "unweighted_roc_auc": float(roc_auc_score(test_y, probability)),
        "weighted_positive_prevalence": float(np.average(test_y.to_numpy(), weights=test_w.to_numpy())),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    sidecar = {
        "features": FEATURE_COLUMNS,
        "family_structure_levels": FAMILY_LEVELS,
        "target": TARGET_COLUMN,
        "weight": WEIGHT_COLUMN,
        "decision_threshold": DECISION_THRESHOLD,
    }
    SIDECAR_PATH.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    _configure_tracking()
    with mlflow.start_run(run_name="xgb-youth-mh"):
        mlflow.log_params({key: value for key, value in MODEL_PARAMS.items()})
        mlflow.log_param("features", ",".join(FEATURE_COLUMNS))
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("decision_threshold", DECISION_THRESHOLD)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, name="model")
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_path = Path(temp_dir) / "shap_summary.png"
            _shap_summary(model, test_x, plot_path)
            mlflow.log_artifact(plot_path, artifact_path="shap")

    for name, value in metrics.items():
        LOGGER.info("%s: %.4f", name, value)
    LOGGER.info("Saved model to %s", MODEL_PATH)
    return metrics


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train()


if __name__ == "__main__":
    main()

__all__ = ["train", "predict_and_explain"]
