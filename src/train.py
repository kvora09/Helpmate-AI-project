"""CLI for training the attrition model."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_utils import DataConfig, get_feature_columns, load_dataset, split_train_score_population
from src.modeling import save_artifacts, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train early attrition model")
    parser.add_argument("--data", type=Path, default=Path("data/attrition_dataset.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(DataConfig(csv_path=args.data))
    train_df, score_df = split_train_score_population(df)

    result = train_model(train_df)
    save_artifacts(result.pipeline, args.output, result.metrics)

    feature_cols = get_feature_columns(train_df)
    scored = score_df.copy()
    scored["attrition_risk"] = result.pipeline.predict_proba(score_df[feature_cols])[:, 1]
    args.output.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output / "retained_employee_scores.csv", index=False)

    print("Training complete")
    print(result.metrics)
    print(f"Scored retained employees: {len(scored)}")


if __name__ == "__main__":
    main()
