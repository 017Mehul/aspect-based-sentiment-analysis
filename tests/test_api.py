from api import AnalyzeRequest, analyze


def test_api_uses_existing_aspect_heuristic(monkeypatch, tmp_path):
    monkeypatch.setenv("ABSA_REVIEW_MODEL_PATH", str(tmp_path / "missing-model.joblib"))
    analyze.__globals__["_load_review_model"].cache_clear()
    try:
        result = analyze(AnalyzeRequest(
            text="The coffee is excellent but the packaging is terrible.",
            aspects=["coffee", "packaging"],
        ))
        labels = {item["aspect"]: item["sentiment"] for item in result["aspects"]}
        assert labels == {"coffee": "positive", "packaging": "negative"}
        assert result["aspect_method_is_heuristic"] is True
        assert result["review_sentiment"] is None
        assert result["review_model_available"] is False
    finally:
        analyze.__globals__["_load_review_model"].cache_clear()


def test_api_does_not_claim_review_prediction_without_artifact(monkeypatch, tmp_path):
    monkeypatch.setenv("ABSA_REVIEW_MODEL_PATH", str(tmp_path / "no-model.joblib"))
    analyze.__globals__["_load_review_model"].cache_clear()
    try:
        result = analyze(AnalyzeRequest(text="The taste is not good.", aspects=["taste"]))
        assert result["review_sentiment"] is None
        assert result["review_model_available"] is False
        assert result["aspects"][0]["sentiment"] == "negative"
    finally:
        analyze.__globals__["_load_review_model"].cache_clear()
