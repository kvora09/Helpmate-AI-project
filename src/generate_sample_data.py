"""Generate a realistic synthetic dataset for attrition modeling demos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "employee_id": np.arange(1, n + 1),
            "age": rng.integers(22, 60, n),
            "tenure_months": rng.integers(1, 180, n),
            "performance_rating": rng.uniform(2.0, 5.0, n).round(2),
            "engagement_score": rng.uniform(20, 100, n).round(1),
            "manager_feedback_score": rng.uniform(1, 5, n).round(2),
            "salary_growth_pct": rng.normal(8, 4, n).clip(-5, 30).round(2),
            "promotion_wait_months": rng.integers(1, 84, n),
            "gptw_team_score": rng.uniform(20, 100, n).round(1),
            "ceo_chat_sentiment": rng.uniform(-1, 1, n).round(3),
            "feedback_resolution_days": rng.integers(1, 45, n),
            "same_batch_attrition_rate": rng.uniform(0, 0.45, n).round(3),
            "posh_cases_last_12m": rng.integers(0, 3, n),
            "safety_issues_last_12m": rng.integers(0, 5, n),
            "department_category": rng.choice(["Sales", "Tech", "Support", "HR"], n),
            "work_mode_band": rng.choice(["Remote", "Hybrid", "Onsite"], n),
        }
    )

    # Non-linear logit to simulate real world attrition dynamics.
    logit = (
        -2.0
        + 0.012 * df["promotion_wait_months"]
        - 0.025 * df["engagement_score"]
        - 0.018 * df["gptw_team_score"]
        + 0.90 * df["same_batch_attrition_rate"]
        + 0.30 * df["posh_cases_last_12m"]
        + 0.15 * df["safety_issues_last_12m"]
        - 0.50 * df["ceo_chat_sentiment"]
        + 0.006 * df["feedback_resolution_days"] ** 1.3
    )
    prob = 1 / (1 + np.exp(-logit))
    df["voluntary_attrition"] = rng.binomial(1, np.clip(prob, 0.01, 0.85), len(df))
    return df


def main() -> None:
    output = Path("data/attrition_dataset.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    generate().to_csv(output, index=False)
    print(f"Synthetic dataset written to {output}")


if __name__ == "__main__":
    main()
