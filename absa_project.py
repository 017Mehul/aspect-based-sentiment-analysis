"""Reproducible sentiment-analysis baseline and aspect-level heuristic demo.

The Amazon Fine Food Reviews dataset has review-level ratings, not aspect-level
annotations. Therefore the classical classifiers below are document-level
baselines; aspect sentiment is estimated separately with an explicit,
interpretable lexicon/proximity heuristic and must not be reported as a
supervised ABSA score.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import string
from collections import Counter
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

LOG = logging.getLogger("absa")
RANDOM_STATE = 42
POSITIVE_WORDS = {
    "amazing", "awesome", "best", "delicious", "easy", "excellent", "fantastic",
    "fresh", "good", "great", "happy", "love", "loved", "perfect", "pleasant",
    "tasty", "wonderful", "worth",
}
NEGATIVE_WORDS = {
    "awful", "bad", "boring", "broken", "disappointing", "disgusting", "hate",
    "hated", "horrible", "poor", "rotten", "stale", "terrible", "unfortunately",
    "waste", "worst",
}
NEGATIONS = {"not", "no", "never", "neither", "hardly", "isn't", "wasn't", "don't", "didn't", "cannot"}


def clean_text(text: str) -> str:
    """Normalize text while preserving negation and apostrophes."""
    if not isinstance(text, str):
        return ""
    text = text.lower().replace("’", "'")
    # Keep letters, digits, whitespace, and apostrophes; don't erase "isn't".
    text = re.sub(r"[^\w\s']", " ", text)
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", text)
    # Deliberately do not remove stopwords: negation and function words carry context.
    return " ".join(tokens)


def load_data(filepath: str | Path, sample_size: int | None = 5000) -> pd.DataFrame:
    """Load ratings and create binary review labels; validate schema and content."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. Download Reviews.csv from the Kaggle dataset and pass --data.")
    df = pd.read_csv(path)
    missing = {"Text", "Score"} - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {sorted(missing)}")
    df = df[["Text", "Score"]].dropna().copy()
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    df = df.dropna(subset=["Score", "Text"])
    df = df[df["Score"].isin([1, 2, 4, 5])].copy()
    df["sentiment"] = np.where(df["Score"] >= 4, "positive", "negative")
    df = df.rename(columns={"Text": "review"})
    df["review"] = df["review"].astype(str).str.strip()
    df = df[df["review"].ne("")]
    if sample_size is not None and sample_size > 0 and len(df) > sample_size:
        # Stratified sampling avoids a sample with an accidental class shift.
        df = (df.groupby("sentiment", group_keys=False)
                .apply(lambda part: part.sample(
                    n=max(1, round(sample_size * len(part) / len(df))),
                    random_state=RANDOM_STATE))
                .sample(frac=1, random_state=RANDOM_STATE)
                .head(sample_size).reset_index(drop=True))
    if df.empty or df["sentiment"].nunique() < 2:
        raise ValueError("Dataset must contain non-empty positive and negative review classes.")
    return df[["review", "sentiment"]].reset_index(drop=True)


def load_spacy_model(model_name: str = "en_core_web_sm"):
    """Load spaCy model without silently downloading packages at runtime."""
    try:
        return spacy.load(model_name, disable=["ner"])
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model {model_name!r} is missing. Install it with: "
            f"python -m spacy download {model_name}"
        ) from exc


def extract_aspects(text: str, nlp=None, max_chars: int = 2000) -> list[str]:
    """Extract candidate noun chunks when possible; fall back to noun lemmas."""
    if not isinstance(text, str) or not text.strip():
        return []
    if nlp is None:
        nlp = load_spacy_model()
    doc = nlp(text[:max_chars])
    candidates = []
    try:
        for chunk in doc.noun_chunks:
            phrase = " ".join(t.lemma_.lower() for t in chunk
                              if not t.is_stop and not t.is_punct and t.pos_ != "PRON")
            if phrase and any(t.pos_ in {"NOUN", "PROPN"} for t in chunk):
                candidates.append(phrase)
    except (ValueError, NotImplementedError):
        pass
    if not candidates:
        candidates = [t.lemma_.lower() for t in doc
                      if t.pos_ in {"NOUN", "PROPN"} and not t.is_stop and not t.is_punct]
    # Stable deduplication
    return list(dict.fromkeys(a for a in candidates if a))


def predict_aspect_sentiment(text: str, aspects: Iterable[str] | None = None) -> list[dict]:
    """Transparent proximity-lexicon baseline; not a trained ABSA model.

    Scores sentiment words within a small context window around each aspect.
    Returns 'unknown' when no sentiment evidence is found instead of guessing.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower().replace("’", "'"))
    if aspects is None:
        nlp = load_spacy_model()
        aspects = extract_aspects(text, nlp=nlp)
    results = []
    for aspect in aspects:
        phrase_tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", aspect.lower())
        positions = []
        for i in range(len(tokens)):
            if phrase_tokens and tokens[i:i + len(phrase_tokens)] == phrase_tokens:
                positions.append(i)
        if not positions:
            # Lemmatized extracted terms may differ from the surface form.
            positions = [i for i, token in enumerate(tokens) if token in phrase_tokens]
        score = 0
        evidence = []
        for pos in positions:
            lo, hi = max(0, pos - 4), min(len(tokens), pos + len(phrase_tokens) + 5)
            context = tokens[lo:hi]
            for j, word in enumerate(context):
                if word in POSITIVE_WORDS or word in NEGATIVE_WORDS:
                    polarity = 1 if word in POSITIVE_WORDS else -1
                    if any(w in NEGATIONS for w in context[max(0, j - 3):j]):
                        polarity *= -1
                    score += polarity
                    evidence.append(word)
        label = "positive" if score > 0 else "negative" if score < 0 else "unknown"
        results.append({"aspect": aspect, "sentiment": label, "score": score,
                        "evidence": evidence, "method": "lexicon-proximity baseline"})
    return results


def build_pipeline(model_name: str, feature_name: str, max_features: int = 5000) -> Pipeline:
    vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, 2)) if feature_name == "BoW" else TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Naive Bayes": MultinomialNB(),
        "SVM (Linear)": LinearSVC(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("vectorizer", vectorizer), ("classifier", models[model_name])])


def evaluate_models(df: pd.DataFrame, output_dir: Path, max_features: int = 5000) -> pd.DataFrame:
    """Train six baselines with train-only vectorizer fitting and full-class metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(
        df["review"].map(clean_text), df["sentiment"], test_size=0.2,
        random_state=RANDOM_STATE, stratify=df["sentiment"])
    rows = []
    best_pipeline = None
    best_f1 = -1.0
    for feature_name in ("BoW", "TF-IDF"):
        for model_name in ("Logistic Regression", "Naive Bayes", "SVM (Linear)"):
            pipe = build_pipeline(model_name, feature_name, max_features)
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            report = classification_report(y_test, pred, labels=["negative", "positive"],
                                           output_dict=True, zero_division=0)
            row = {
                "model": model_name, "features": feature_name,
                "accuracy": accuracy_score(y_test, pred),
                "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
                "weighted_f1": f1_score(y_test, pred, average="weighted", zero_division=0),
                "negative_precision": precision_score(y_test, pred, pos_label="negative", zero_division=0),
                "negative_recall": recall_score(y_test, pred, pos_label="negative", zero_division=0),
                "negative_f1": report["negative"]["f1-score"],
                "positive_precision": precision_score(y_test, pred, pos_label="positive", zero_division=0),
                "positive_recall": recall_score(y_test, pred, pos_label="positive", zero_division=0),
                "positive_f1": report["positive"]["f1-score"],
            }
            rows.append(row)
            safe_name = feature_name.lower().replace("-", "_").replace(" ", "_")
            model_slug = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            pd.DataFrame(confusion_matrix(y_test, pred, labels=["negative", "positive"]),
                         index=["actual_negative", "actual_positive"],
                         columns=["predicted_negative", "predicted_positive"]).to_csv(
                             output_dir / f"confusion_{safe_name}_{model_slug}.csv")
            with (output_dir / f"report_{safe_name}_{model_slug}.json").open("w", encoding="utf-8") as f:
                json.dump(classification_report(y_test, pred, output_dict=True, zero_division=0), f, indent=2)
            if row["macro_f1"] > best_f1:
                best_f1, best_pipeline = row["macro_f1"], pipe
    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    results.to_csv(output_dir / "model_results.csv", index=False)
    results.to_json(output_dir / "model_results.json", orient="records", indent=2)
    if best_pipeline is not None:
        joblib.dump(best_pipeline, output_dir / "best_review_sentiment_pipeline.joblib")
    plot_results(results, output_dir)
    return results


def plot_results(results: pd.DataFrame, output_dir: Path) -> None:
    """Create reproducible comparison plots from the freshly computed results."""
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = results["model"] + " / " + results["features"]
    ax.barh(labels[::-1], results["accuracy"][::-1])
    ax.set_xlabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_title("Review-level sentiment accuracy")
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_comparison.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels[::-1], results["macro_f1"][::-1])
    ax.set_xlabel("Macro F1")
    ax.set_xlim(0, 1)
    ax.set_title("Review-level sentiment macro F1")
    fig.tight_layout()
    fig.savefig(output_dir / "macro_f1_comparison.png", dpi=150)
    plt.close(fig)


def run_demo(nlp=None) -> list[dict]:
    examples = [
        ("The coffee is excellent but the packaging is terrible.", ["coffee", "packaging"]),
        ("The taste is not good and the delivery was great.", ["taste", "delivery"]),
    ]
    output = []
    for text, aspects in examples:
        output.append({"text": text, "predictions": predict_aspect_sentiment(text, aspects)})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="Reviews.csv", help="Path to Amazon Fine Food Reviews CSV")
    parser.add_argument("--sample-size", type=int, default=5000, help="Number of reviews; use 0 for all rows")
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--demo-only", action="store_true", help="Run the lexicon-based aspect demo only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.demo_only:
        print(json.dumps(run_demo(), indent=2))
        return
    df = load_data(args.data, None if args.sample_size == 0 else args.sample_size)
    LOG.info("Loaded %d reviews; class counts: %s", len(df), df["sentiment"].value_counts().to_dict())
    results = evaluate_models(df, Path(args.output_dir), args.max_features)
    print(results.to_string(index=False))
    print(f"\nSaved metrics, reports, plots and best review-level model to {args.output_dir}/")
    print("Note: these metrics evaluate review-level sentiment, not aspect-level sentiment.")


if __name__ == "__main__":
    main()
