# Aspect-Based Sentiment Analysis (ABSA) — Review Sentiment Baseline

A reproducible NLP project for Amazon Fine Food Reviews. It compares classical **review-level** sentiment classifiers and includes a separate, interpretable **lexicon/proximity aspect-sentiment demo**.

> **Important scope note:** Amazon Fine Food Reviews provides review ratings, not gold labels for individual aspects. The BoW/TF-IDF models therefore predict sentiment for an entire review. The aspect-level demo is a heuristic baseline, not a supervised or benchmark-validated ABSA model. Do not describe its outputs as proven aspect-level accuracy.

## Features

- Validated dataset loading and reproducible sampling.
- Text normalization that preserves negation (for example, “not good”).
- Review-level binary labels: ratings 1–2 = negative, 4–5 = positive; neutral 3-star reviews are excluded.
- Six baseline configurations: Logistic Regression, Multinomial Naive Bayes, and Linear SVM using BoW and TF-IDF.
- Stratified holdout evaluation with accuracy, macro/weighted F1, and per-class precision, recall and F1.
- Fresh CSV/JSON metrics, confusion-matrix CSVs, plots, and a serialized best review-level pipeline.
- Candidate noun-phrase extraction with spaCy.
- A transparent aspect sentiment heuristic that assigns nearby opinion words to the nearest aspect and returns “unknown” when no lexicon evidence is found.
- Unit tests and GitHub Actions CI.

## Repository structure

- `absa_project.py` — training, evaluation, aspect extraction and demo.
- `tests/test_absa_project.py` — regression tests.
- `requirements.txt` — Python dependencies.
- `ABSA_Academic_Report.md` — report draft; update its numerical results after running the current pipeline.
- `outputs/` — generated metrics, reports, confusion matrices, plots and model artifact (created locally; not committed).
- `LICENSE` — MIT License.

## Setup

Python 3.11 is recommended.

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Dataset

Download `Reviews.csv` from the [Amazon Fine Food Reviews dataset on Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) and place it in the repository root, or pass its location with `--data`. The dataset is intentionally not committed.

## Run

Run the baseline on a reproducible sample of 5,000 reviews:

```bash
python absa_project.py
```

Use all eligible rows:

```bash
python absa_project.py --sample-size 0
```

Choose a dataset and output directory:

```bash
python absa_project.py --data /path/to/Reviews.csv --sample-size 5000 --output-dir outputs
```

Run the aspect-heuristic demo without training:

```bash
python absa_project.py --demo-only
```

Run tests:

```bash
pytest -q
```

The first full run writes results to `outputs/`. Metrics are generated from that run; old reported numbers should not be assumed to match the current code until reproduced.

## Evaluation interpretation

The review-level test split is stratified and uses a fixed random seed. Vectorizers are fitted only on training text via scikit-learn pipelines. Class weighting is used for Logistic Regression and Linear SVM; Naive Bayes remains a baseline without class weights. Review-level metrics do **not** measure aspect extraction or aspect-level sentiment.

For research-grade ABSA evaluation, use a dataset with explicit aspect spans/terms and sentiment labels (for example, a suitable SemEval ABSA dataset). Evaluate aspect extraction separately from sentiment polarity classification and report exact dataset splits, label mapping, per-class results and macro-F1. Do not use review-level star ratings as if they were aspect-level ground truth.

## Known limitations

- Aspect extraction is based on spaCy noun chunks/fallback noun lemmas and can miss implicit or domain-specific aspects.
- The aspect polarity demo uses a small lexicon and a local context window. It can fail on sarcasm, long-distance dependencies, contrastive clauses, domain-specific vocabulary and multiple opinions. It is intended for transparent demonstration, not production use.
- The Amazon Fine Food Reviews labels are review-level and the dataset is imbalanced; class weighting and macro-F1 help analysis but do not remove all bias.
- Classical BoW/TF-IDF models do not understand full context.
- No web interface or API is included; the model artifact is a local inference artifact.

## Research/report hygiene

After running the current code, copy the actual metrics from `outputs/model_results.csv` into the academic report and README if publishing numeric results. Record the dataset version, sample size, seed, dependency versions and split strategy. Never present previously committed scores as freshly reproduced unless they match the current run.


## Supervised ABSA evaluation (SemEval-2014 Restaurants)

The repository now includes `absa_supervised.py`, a separate two-stage supervised experiment. It trains a token-level BIO aspect-term extractor and a polarity classifier that receives the sentence with the target aspect marked. It reports:
- exact-span aspect extraction precision, recall and F1;
- aspect polarity accuracy and macro-F1 when supplied gold aspect spans (conditional polarity evaluation);
- end-to-end joint precision, recall and F1 requiring both exact span and correct polarity.

Obtain the corrected SemEval-2014 Task 4 Restaurants training XML and gold test XML from the official task distribution and follow its access/usage terms. Keep the XML files local; they are not committed here. Save them, for example, as `data/Restaurants_Train_v2.xml` and `data/Restaurants_Test_Gold.xml`, then run:

```bash
python absa_supervised.py --train data/Restaurants_Train_v2.xml --test data/Restaurants_Test_Gold.xml --output-dir outputs/semeval14
```

The metrics are saved to `outputs/semeval14/semeval_absa_metrics.json`, polarity predictions to JSONL, and the fitted models to a local joblib artifact. Do not compare results across different SemEval releases/splits without documenting the difference. The polarity task uses gold aspects for its conditional score; the end-to-end score includes extraction errors.

The SemEval run is independent of Amazon review-level sentiment. Amazon results cannot validate aspect-term extraction or aspect polarity because Amazon's `Score` labels are for whole reviews only.

## License

MIT. See [LICENSE](LICENSE).
