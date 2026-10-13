"""Supervised two-stage ABSA evaluation on SemEval-2014 Restaurants XML.

Stage A: token-level BIO aspect-term extraction, scored with exact-span P/R/F1.
Stage B: aspect polarity classification from sentence + marked aspect, scored on
gold aspect terms. An additional end-to-end joint score requires both exact span
and correct polarity. This is distinct from the Amazon review-level baseline.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
VALID_POLARITIES = {"positive", "negative", "neutral", "conflict"}


def tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def parse_semeval_xml(path: str | Path) -> list[dict[str, Any]]:
    """Parse SemEval ABSA XML; offsets are the gold character offsets."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"SemEval XML not found: {file_path}")
    root = ET.parse(file_path).getroot()
    examples: list[dict[str, Any]] = []
    for sentence_node in root.iter("sentence"):
        text_node = sentence_node.find("text")
        if text_node is None or not (text_node.text or "").strip():
            continue
        text = text_node.text or ""
        aspects = []
        terms_node = sentence_node.find("aspectTerms")
        if terms_node is not None:
            for item in terms_node.findall("aspectTerm"):
                term = item.get("term", "")
                polarity = item.get("polarity", "").lower()
                try:
                    start, end = int(item.attrib["from"]), int(item.attrib["to"])
                except (KeyError, ValueError):
                    # Offset-less terms cannot support extraction evaluation.
                    continue
                if not term or start < 0 or end <= start or end > len(text):
                    continue
                if text[start:end].casefold() != term.casefold():
                    # Reject misaligned annotations rather than silently training
                    # on incorrect offsets.
                    continue
                aspects.append({"term": term, "polarity": polarity,
                                "start": start, "end": end})
        examples.append({"text": text, "aspects": aspects})
    if not examples:
        raise ValueError(f"No sentence records found in {file_path}")
    return examples


def token_labels(example: dict[str, Any]) -> tuple[list[tuple[str, int, int]], list[str]]:
    tokens = tokenize_with_offsets(example["text"])
    labels = ["O"] * len(tokens)
    for aspect in example["aspects"]:
        indices = [i for i, (_, start, end) in enumerate(tokens)
                   if start < aspect["end"] and end > aspect["start"]]
        if not indices:
            continue
        labels[indices[0]] = "B"
        for idx in indices[1:]:
            labels[idx] = "I"
    return tokens, labels


def _shape(token: str) -> str:
    return "".join("X" if c.isupper() else "x" if c.islower()
                   else "d" if c.isdigit() else c for c in token)[:12]


def token_features(tokens: list[tuple[str, int, int]], i: int,
                   text_length: int) -> dict[str, Any]:
    token = tokens[i][0]
    lower = token.lower()
    features: dict[str, Any] = {
        "bias": 1.0,
        "word": lower,
        "prefix1": lower[:1],
        "prefix2": lower[:2],
        "suffix2": lower[-2:],
        "suffix3": lower[-3:],
        "shape": _shape(token),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "is_digit": token.isdigit(),
        "position_bucket": min(9, int(10 * tokens[i][1] / max(1, text_length))),
    }
    for offset in (-2, -1, 1, 2):
        j = i + offset
        if 0 <= j < len(tokens):
            neighbor = tokens[j][0].lower()
            features[f"word{offset}"] = neighbor
            features[f"pair{min(offset, 0)}_{max(offset, 0)}"] = (
                f"{neighbor}|{lower}" if offset < 0 else f"{lower}|{neighbor}"
            )
    return features


def _int32_csr(matrix):
    """Normalize sparse indices for scikit-learn estimators that require int32."""
    matrix = matrix.tocsr()
    matrix.indices = matrix.indices.astype(np.int32, copy=False)
    matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
    return matrix


def examples_to_token_rows(examples: list[dict[str, Any]]):
    feature_rows, labels, metadata = [], [], []
    for sentence_idx, example in enumerate(examples):
        tokens, token_y = token_labels(example)
        for i, token in enumerate(tokens):
            feature_rows.append(token_features(tokens, i, len(example["text"])))
            labels.append(token_y[i])
            metadata.append((sentence_idx, token[1], token[2]))
    return feature_rows, labels, metadata


def bio_to_spans(tags: list[str], tokens: list[tuple[str, int, int]]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = None
    end = None
    for tag, (_, token_start, token_end) in zip(tags, tokens):
        if tag == "B":
            if start is not None:
                spans.append((start, end))
            start, end = token_start, token_end
        elif tag == "I":
            if start is None:
                start = token_start
            end = token_end
        else:
            if start is not None:
                spans.append((start, end))
                start = end = None
    if start is not None:
        spans.append((start, end))
    return spans


def marked_aspect_text(text: str, start: int, end: int) -> str:
    return f"{text[:start]} [ASPECT] {text[start:end]} [/ASPECT] {text[end:]}"


def polarity_input(text: str, aspect: dict[str, Any]) -> str:
    return marked_aspect_text(text, aspect["start"], aspect["end"])


def score_spans(gold: set[tuple[int, int, int]],
                predicted: set[tuple[int, int, int]]) -> dict[str, float]:
    tp = len(gold & predicted)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "gold_spans": len(gold), "predicted_spans": len(predicted),
            "true_positive_spans": tp}


def evaluate(train_xml: str | Path, test_xml: str | Path,
             output_dir: str | Path) -> dict[str, Any]:
    train = parse_semeval_xml(train_xml)
    test = parse_semeval_xml(test_xml)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Stage A: supervised token classification followed by exact-span decoding.
    x_train, y_train, _ = examples_to_token_rows(train)
    vectorizer = DictVectorizer(sparse=True)
    X_train = _int32_csr(vectorizer.fit_transform(x_train))
    extractor = LogisticRegression(max_iter=1500, class_weight="balanced",
                                   random_state=42, solver="lbfgs")
    extractor.fit(X_train, y_train)

    gold_spans: set[tuple[int, int, int]] = set()
    predicted_spans: set[tuple[int, int, int]] = set()
    for sentence_idx, example in enumerate(test):
        tokens = tokenize_with_offsets(example["text"])
        features = [token_features(tokens, i, len(example["text"]))
                    for i in range(len(tokens))]
        if not features:
            continue
        predicted_tags = extractor.predict(_int32_csr(vectorizer.transform(features))).tolist()
        for aspect in example["aspects"]:
            gold_spans.add((sentence_idx, aspect["start"], aspect["end"]))
        for start, end in bio_to_spans(predicted_tags, tokens):
            predicted_spans.add((sentence_idx, start, end))
    extraction_metrics = score_spans(gold_spans, predicted_spans)

    # Stage B: supervised polarity model using gold aspect spans (official
    # conditional-polarity protocol), then an end-to-end joint exact-span score.
    polarity_train = [
        (polarity_input(ex["text"], aspect), aspect["polarity"])
        for ex in train for aspect in ex["aspects"]
        if aspect["polarity"] in VALID_POLARITIES
    ]
    polarity_test = [
        (sentence_idx, ex, aspect)
        for sentence_idx, ex in enumerate(test) for aspect in ex["aspects"]
        if aspect["polarity"] in VALID_POLARITIES
    ]
    if len({label for _, label in polarity_train}) < 2:
        raise ValueError("Training XML must contain at least two aspect polarity classes.")
    polarity_model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                                  max_features=50000, sublinear_tf=True,
                                  strip_accents="unicode")),
        ("classifier", LinearSVC(class_weight="balanced", random_state=42)),
    ])
    polarity_model.fit([x for x, _ in polarity_train],
                       [y for _, y in polarity_train])
    gold_labels, predicted_labels = [], []
    predicted_joint: set[tuple[int, int, int, str]] = set()
    gold_joint: set[tuple[int, int, int, str]] = set()
    predicted_polarities = []
    for sentence_idx, ex, aspect in polarity_test:
        pred = polarity_model.predict([polarity_input(ex["text"], aspect)])[0]
        gold_labels.append(aspect["polarity"])
        predicted_labels.append(pred)
        predicted_polarities.append({
            "sentence_index": sentence_idx, "text": ex["text"],
            "aspect": aspect["term"], "gold": aspect["polarity"],
            "predicted": pred, "start": aspect["start"], "end": aspect["end"],
        })
        gold_joint.add((sentence_idx, aspect["start"], aspect["end"], aspect["polarity"]))
        if pred == aspect["polarity"]:
            # This set captures gold spans with correct polarity. Full joint
            # extraction+polarity is computed below using predicted spans.
            pass

    # Joint extraction + polarity: assign each predicted span a polarity and
    # require exact offsets and correct sentiment to count as a true positive.
    end_to_end_predictions = []
    for sentence_idx, start, end in sorted(predicted_spans):
        text = test[sentence_idx]["text"]
        fake_aspect = {"start": start, "end": end, "term": text[start:end]}
        pred = polarity_model.predict([polarity_input(text, fake_aspect)])[0]
        end_to_end_predictions.append((sentence_idx, start, end, pred))
    predicted_joint = set(end_to_end_predictions)
    joint_tp = len(gold_joint & predicted_joint)
    joint_p = joint_tp / len(predicted_joint) if predicted_joint else 0.0
    joint_r = joint_tp / len(gold_joint) if gold_joint else 0.0
    joint_f1 = 2 * joint_p * joint_r / (joint_p + joint_r) if joint_p + joint_r else 0.0

    polarity_p, polarity_r, polarity_f1, _ = precision_recall_fscore_support(
        gold_labels, predicted_labels, labels=sorted(VALID_POLARITIES),
        average="macro", zero_division=0,
    )
    results = {
        "dataset": "SemEval-2014 Task 4 Restaurants",
        "train_file": str(train_xml),
        "test_file": str(test_xml),
        "train_sentences": len(train),
        "test_sentences": len(test),
        "train_aspect_terms": len(polarity_train),
        "test_aspect_terms": len(polarity_test),
        "aspect_term_extraction_exact_span": extraction_metrics,
        "aspect_polarity_given_gold_aspects": {
            "accuracy": float(accuracy_score(gold_labels, predicted_labels)),
            "macro_precision": float(polarity_p),
            "macro_recall": float(polarity_r),
            "macro_f1": float(polarity_f1),
            "classification_report": classification_report(
                gold_labels, predicted_labels, labels=sorted(VALID_POLARITIES),
                output_dict=True, zero_division=0),
        },
        "end_to_end_joint_exact_span_and_polarity": {
            "precision": joint_p, "recall": joint_r, "f1": joint_f1,
            "gold_aspect_terms": len(gold_joint),
            "predicted_aspect_terms": len(predicted_joint),
            "true_positive": joint_tp,
        },
    }
    (output / "semeval_absa_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    with (output / "semeval_polarity_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in predicted_polarities:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    joblib.dump({"vectorizer": vectorizer, "aspect_extractor": extractor,
                 "polarity_model": polarity_model},
                output / "supervised_absa_models.joblib")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="SemEval Restaurants training XML")
    parser.add_argument("--test", required=True, help="SemEval Restaurants gold test XML")
    parser.add_argument("--output-dir", default="outputs/semeval14")
    args = parser.parse_args()
    try:
        results = evaluate(args.train, args.test, args.output_dir)
    except (FileNotFoundError, ValueError, ET.ParseError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
