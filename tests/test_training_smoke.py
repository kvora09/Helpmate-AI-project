from src.generate_sample_data import generate
from src.modeling import train_model


def test_train_smoke() -> None:
    df = generate(n=300, seed=7)
    result = train_model(df)
    assert 0.5 <= result.metrics["roc_auc"] <= 1.0
    assert 0.0 <= result.metrics["pr_auc"] <= 1.0
