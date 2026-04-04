"""Utilities for loading, validating, and enriching attrition data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BASE_FEATURES = [
    "age",
    "tenure_months",
    "performance_rating",
    "engagement_score",
    "manager_feedback_score",
    "salary_growth_pct",
    "promotion_wait_months",
]

ADDITIONAL_FEATURES = [
    "gptw_team_score",
    "ceo_chat_sentiment",
    "feedback_resolution_days",
    "same_batch_attrition_rate",
    "posh_cases_last_12m",
    "safety_issues_last_12m",
]

TARGET_COLUMN = "voluntary_attrition"


@dataclass
class DataConfig:
    """Configuration for loading and preprocessing data."""

    csv_path: Path
    id_column: str = "employee_id"


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def get_categorical_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_category") or c.endswith("_band")]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return BASE_FEATURES + ADDITIONAL_FEATURES + get_categorical_features(df)


def load_dataset(config: DataConfig) -> pd.DataFrame:
    """Load dataset and run minimal schema checks.

    Notes:
        - The target contains *only* voluntary resignations.
        - Retained employees (target=0) are used as the scoring population.
    """
    df = pd.read_csv(config.csv_path)
    required = BASE_FEATURES + ADDITIONAL_FEATURES + [TARGET_COLUMN]
    ensure_columns(df, required)

    if config.id_column not in df.columns:
        df[config.id_column] = np.arange(1, len(df) + 1)

    return df


def split_train_score_population(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df.copy()
    score_df = df[df[TARGET_COLUMN] == 0].copy()
    return train_df, score_df
