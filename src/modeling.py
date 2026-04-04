"""Model training helpers for early attrition detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from src.data_utils import ADDITIONAL_FEATURES, BASE_FEATURES, TARGET_COLUMN, get_categorical_features


@dataclass
class TrainResult:
    pipeline: Pipeline
    metrics: dict


def build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    """Build a stochastic polynomial model.

    We use:
      1) PolynomialFeatures for nonlinear interactions.
      2) SGDClassifier (stochastic optimization) for scalability.
      3) Calibrated probabilities for actionable risk scores.
    """
    numeric_preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ]
    )

    categorical_preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=0.01),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_preprocess, numeric_features),
            ("cat", categorical_preprocess, categorical_features),
        ]
    )

    base_model = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=1e-4,
        l1_ratio=0.2,
        max_iter=2000,
        random_state=42,
    )

    calibrated = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", calibrated),
        ]
    )


def train_model(df: pd.DataFrame) -> TrainResult:
    """Train and evaluate the attrition model."""
    feature_cols = BASE_FEATURES + ADDITIONAL_FEATURES
    categorical_cols = get_categorical_features(df)

    X = df[feature_cols + categorical_cols]
    y = df[TARGET_COLUMN]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    pipeline = build_pipeline(feature_cols, categorical_cols)
    pipeline.fit(X_train, y_train)

    valid_scores = pipeline.predict_proba(X_valid)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_valid, valid_scores)),
        "pr_auc": float(average_precision_score(y_valid, valid_scores)),
        "validation_samples": int(len(y_valid)),
        "attrition_rate_validation": float(np.mean(y_valid)),
    }
    return TrainResult(pipeline=pipeline, metrics=metrics)


def save_artifacts(model: Pipeline, output_dir: Path, metrics: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "attrition_model.joblib")
    pd.Series(metrics).to_json(output_dir / "metrics.json", indent=2)
