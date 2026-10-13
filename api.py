"""HTTP inference adapter for the existing ABSA Python pipeline.

Aspect polarity is the repository's transparent lexicon/proximity baseline.
Review-level prediction is returned only when a trained joblib pipeline exists.
No synthetic scores or placeholder predictions are generated.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from absa_project import clean_text, extract_aspects, predict_aspect_sentiment

app = FastAPI(
    title="ABSA Inference API",
    description="Inference adapter for the repository's existing Python sentiment pipeline.",
    version="1.0.0",
)


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    aspects: list[str] | None = Field(default=None, max_length=20)


@lru_cache(maxsize=1)
def _load_nlp():
    from absa_project import load_spacy_model
    return load_spacy_model()


@lru_cache(maxsize=1)
def _load_review_model():
    configured = os.getenv("ABSA_REVIEW_MODEL_PATH", "outputs/best_review_sentiment_pipeline.joblib")
    path = Path(configured)
    if not path.is_file():
        return None
    return joblib.load(path)


@app.get("/health")
def health() -> dict[str, Any]:
    model_path = Path(os.getenv("ABSA_REVIEW_MODEL_PATH", "outputs/best_review_sentiment_pipeline.joblib"))
    return {
        "status": "ok",
        "service": "absa-inference",
        "review_model_available": model_path.is_file(),
        "review_model_path": str(model_path),
        "aspect_method": "lexicon-proximity baseline",
        "note": "Aspect polarity is heuristic; review-level sentiment requires a trained model artifact.",
    }


@app.post("/analyze")
def analyze(payload: AnalyzeRequest) -> dict[str, Any]:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Review text cannot be blank.")

    supplied_aspects = [a.strip() for a in (payload.aspects or []) if a.strip()]
    try:
        aspects = supplied_aspects if supplied_aspects else extract_aspects(text, nlp=_load_nlp())
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"{exc}. Alternatively, provide aspects explicitly in the UI.",
        ) from exc

    aspect_predictions = predict_aspect_sentiment(text, aspects=aspects)
    review_prediction = None
    review_model_available = False
    try:
        model = _load_review_model()
        if model is not None:
            review_prediction = str(model.predict([clean_text(text)])[0])
            review_model_available = True
    except Exception as exc:
        # Do not disguise a broken artifact as a valid prediction.
        raise HTTPException(status_code=500, detail=f"Review model inference failed: {type(exc).__name__}.") from exc

    return {
        "input_text": text,
        "aspects": aspect_predictions,
        "aspect_method": "lexicon-proximity baseline",
        "aspect_method_is_heuristic": True,
        "review_sentiment": review_prediction,
        "review_model_available": review_model_available,
        "review_model_kind": "trained review-level scikit-learn pipeline" if review_model_available else None,
        "message": (
            "Aspect predictions use a transparent lexicon/proximity heuristic, not a supervised aspect-polarity model."
            if aspect_predictions
            else "No aspects were detected. Try entering aspect terms manually."
        ),
    }
