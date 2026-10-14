# Aspectly — Aspect-Based Sentiment Analysis

<p align="center">
  <strong>Go beyond a single sentiment score.</strong><br />
  Explore review sentiment, inspect aspect-level signals, and see how a transparent NLP pipeline turns customer feedback into useful insights.
</p>

<p align="center">
  <a href="https://aspect-based-sentiment-analysis-xi.vercel.app/"><strong>Live Demo</strong></a> ·
  <a href="https://github.com/017Mehul/aspect-based-sentiment-analysis/actions"><strong>CI Workflows</strong></a> ·
  <a href="README_UI.md"><strong>Web UI & Deployment Guide</strong></a>
</p>

---

## Overview

**Aspectly** is an end-to-end NLP project for exploring sentiment in customer reviews. It combines a deployed Next.js interface with a Python inference service and a classical machine-learning pipeline for review-level sentiment prediction.

The project deliberately separates two related tasks:

- **Review-level sentiment:** a trained scikit-learn pipeline predicts sentiment for the complete review.
- **Aspect-level polarity:** a transparent lexicon/proximity heuristic estimates sentiment near a supplied or extracted aspect.

> **Research note:** The Amazon Fine Food Reviews dataset provides review ratings, not gold sentiment labels for individual aspects. The aspect-level output is therefore a heuristic demonstration—not a supervised, benchmark-validated aspect-polarity model. Review-level metrics must not be presented as aspect-level accuracy.

## Live application

Try the deployed app: **[Aspectly on Vercel](https://aspect-based-sentiment-analysis-xi.vercel.app/)**

The interface includes:
- A responsive review-analysis workspace.
- An interactive Three.js hero section.
- Optional comma-separated aspect terms, or automatic aspect extraction when the required spaCy model is installed.
- Results from the connected Python service, with the heuristic aspect method identified honestly.
- A trained review-level prediction when the model artifact is available.
- Clear handling of unavailable models and inference errors instead of fabricated predictions.

### Architecture

```text
Browser
  │
  ▼
Next.js / React on Vercel
  │  POST /api/analyze
  ▼
FastAPI inference service on Render
  │
  ├── Aspect extraction (spaCy / fallback)
  ├── Aspect polarity (lexicon/proximity heuristic)
  └── Review-level prediction (trained scikit-learn pipeline)
```

- **Frontend:** Next.js, React, TypeScript, CSS, Three.js.
- **API proxy:** Next.js route validates the request and forwards it to Python.
- **Inference:** FastAPI calls the project’s Python NLP functions.
- **Model artifact:** `outputs/best_review_sentiment_pipeline.joblib`.
- **Deployment:** Vercel for the web app and Render for the Python service.

For the detailed local setup and environment variables, see **[README_UI.md](README_UI.md)**.

## Features

- Dataset loading and reproducible sampling for Amazon Fine Food Reviews.
- Text normalization that preserves negation (for example, “not good”).
- Binary review labels: ratings 1–2 are negative, ratings 4–5 are positive; neutral 3-star reviews are excluded.
- Six classical baseline configurations: Logistic Regression, Multinomial Naive Bayes, and Linear SVM with Bag-of-Words and TF-IDF features.
- Stratified holdout evaluation using accuracy, macro/weighted F1, and per-class precision, recall, and F1.
- Confusion-matrix outputs, plots, CSV/JSON metrics, and a serialized review-level model.
- Candidate aspect extraction using spaCy noun phrases and fallback noun lemmas.
- Explainable aspect polarity estimates based on nearby opinion words.
- Unit tests and GitHub Actions workflows for tests and model-artifact integration.

## Repository layout

| Path | Purpose |
|---|---|
| `absa_project.py` | Review-level training/evaluation, aspect extraction, and heuristic demo |
| `absa_supervised.py` | Separate supervised SemEval ABSA experiment |
| `api.py` | FastAPI inference service |
| `app/` | Next.js web application and API route |
| `tests/` | Python regression tests |
| `requirements.txt` | Python dependencies |
| `README_UI.md` | Web UI, API, and deployment instructions |
| `EXPERIMENT_RESULTS.md` | Experiment results and provenance |
| `outputs/` | Generated reports, plots, metrics, and model artifact when produced |

## Quick start: Python pipeline

Python 3.11 is recommended.

### 1. Create an environment and install dependencies

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Get the dataset

Download `Reviews.csv` from the [Amazon Fine Food Reviews dataset on Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews). Place it in the repository root or provide its location with `--data`.

The dataset is not committed to this repository.

### 3. Train and evaluate

Run the reproducible 5,000-review sample:

```bash
python absa_project.py
```

Run all eligible rows:

```bash
python absa_project.py --sample-size 0
```

Choose a dataset and output directory:

```bash
python absa_project.py --data /path/to/Reviews.csv --sample-size 5000 --output-dir outputs
```

Run only the aspect-heuristic demo:

```bash
python absa_project.py --demo-only
```

The pipeline writes metrics, confusion matrices, plots, reports, and the trained review-level model to the output directory.

### 4. Run tests

```bash
pytest -q
```

## Supervised ABSA experiment

The separate `absa_supervised.py` experiment supports a two-stage supervised workflow using SemEval-2014 Task 4 Restaurants data:

1. A token-level BIO model extracts aspect terms.
2. A polarity classifier predicts sentiment for a marked aspect.
3. Evaluation reports exact-span extraction precision/recall/F1, conditional polarity metrics using gold aspects, and end-to-end joint metrics.

Obtain the corrected SemEval training and gold test XML files from the official task distribution and follow its access and usage terms. Keep the XML files local; they are not committed.

Example:

```bash
python absa_supervised.py \
  --train data/Restaurants_Train_v2.xml \
  --test data/Restaurants_Test_Gold.xml \
  --output-dir outputs/semeval14
```

The SemEval experiment is separate from the deployed Amazon review-level classifier. Do not treat Amazon star ratings as aspect-level ground truth.

## Evaluation and interpretation

- The review-level test split is stratified and uses a fixed random seed.
- Vectorizers are fitted only on training data through scikit-learn pipelines.
- Class weighting is used for Logistic Regression and Linear SVM; Naive Bayes remains an unweighted baseline.
- Accuracy and macro-F1 describe the review-level classification task only.
- Aspect extraction and aspect polarity require aspect-annotated data for rigorous evaluation.
- Use the generated outputs and [experiment notes](EXPERIMENT_RESULTS.md) for reported metrics. Record the dataset version, sample size, random seed, dependency versions, and split strategy when reproducing results.

## Known limitations

- **Aspect polarity is heuristic.** A small lexicon and local context window can miss sarcasm, contrastive clauses, long-distance dependencies, domain-specific wording, and multiple opinions.
- **Aspect extraction can miss implicit or domain-specific aspects.** Automatic extraction depends on the installed spaCy language model and fallback logic.
- **Classical BoW/TF-IDF models have limited contextual understanding.**
- **The Amazon dataset is imbalanced and labels whole reviews.** Class weighting and macro-F1 help analysis but do not eliminate bias.
- **Live predictions are not benchmark scores.** The app does not display experiment metrics as if they were predictions.

## Documentation

- [Web UI and deployment guide](README_UI.md)
- [Experiment results and provenance](EXPERIMENT_RESULTS.md)
- [MIT License](LICENSE)
- [Live application](https://aspect-based-sentiment-analysis-xi.vercel.app/)
- [GitHub Actions](https://github.com/017Mehul/aspect-based-sentiment-analysis/actions)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
