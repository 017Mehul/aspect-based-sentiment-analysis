import pandas as pd
import pytest

from absa_project import clean_text, load_data, predict_aspect_sentiment


def test_clean_text_preserves_negation():
    cleaned = clean_text("This is NOT good!")
    assert "not" in cleaned.split()
    assert "good" in cleaned.split()


def test_aspect_sentiment_handles_mixed_opinions():
    result = predict_aspect_sentiment(
        "The coffee is excellent but the packaging is terrible.",
        ["coffee", "packaging"],
    )
    labels = {item["aspect"]: item["sentiment"] for item in result}
    assert labels["coffee"] == "positive"
    assert labels["packaging"] == "negative"


def test_aspect_sentiment_negation_flips_polarity():
    result = predict_aspect_sentiment("The taste is not good.", ["taste"])
    assert result[0]["sentiment"] == "negative"


def test_unknown_when_no_lexicon_evidence():
    result = predict_aspect_sentiment("The cup is on the table.", ["cup"])
    assert result[0]["sentiment"] == "unknown"


def test_load_data_checks_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_data(tmp_path / "missing.csv")


def test_load_data_checks_required_columns(tmp_path):
    path = tmp_path / "reviews.csv"
    pd.DataFrame({"review": ["nice"], "rating": [5]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required column"):
        load_data(path)


def test_evaluate_models_writes_all_expected_artifacts(tmp_path):
    from absa_project import evaluate_models

    positive = [
        "excellent fresh coffee", "great tasty food", "wonderful delicious meal",
        "good quality product", "love this amazing drink", "perfect flavor",
        "fantastic fresh taste", "very good and pleasant", "best delicious snack",
        "happy with excellent quality",
    ]
    negative = [
        "terrible stale coffee", "awful bad food", "horrible disgusting meal",
        "poor quality product", "hate this broken drink", "worst flavor",
        "rotten stale taste", "very bad and disappointing", "waste terrible snack",
        "unfortunately poor quality",
    ]
    df = pd.DataFrame({
        "review": positive + negative,
        "sentiment": ["positive"] * len(positive) + ["negative"] * len(negative),
    })

    results = evaluate_models(df, tmp_path, max_features=100)

    assert len(results) == 6
    assert {"accuracy", "macro_f1", "weighted_f1"}.issubset(results.columns)
    assert (tmp_path / "model_results.csv").is_file()
    assert (tmp_path / "model_results.json").is_file()
    assert (tmp_path / "accuracy_comparison.png").is_file()
    assert (tmp_path / "macro_f1_comparison.png").is_file()
    assert (tmp_path / "best_review_sentiment_pipeline.joblib").is_file()
    assert len(list(tmp_path.glob("confusion_*.csv"))) == 6
    assert len(list(tmp_path.glob("report_*.json"))) == 6

