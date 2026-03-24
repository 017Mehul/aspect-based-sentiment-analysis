"""
Aspect-Based Sentiment Analysis (ABSA) - Product Reviews
=========================================================
Dataset: Amazon Fine Food Reviews (Reviews.csv)
Columns used: Text (review), Score (1-5 mapped to sentiment)
"""

import pandas as pd
import numpy as np
import re
import string
import warnings
warnings.filterwarnings('ignore')

import nltk
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# ─────────────────────────────────────────────
# 0. Download NLTK resources
# ─────────────────────────────────────────────
print("Downloading NLTK resources...")


def ensure_nltk_resource(resource_path: str, download_name: str):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name, quiet=True)


ensure_nltk_resource('corpora/stopwords', 'stopwords')
ensure_nltk_resource('tokenizers/punkt', 'punkt')
ensure_nltk_resource('tokenizers/punkt_tab', 'punkt_tab')

STOP_WORDS = set(stopwords.words('english'))

# Load spaCy model
print("Loading spaCy model...")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")


# ─────────────────────────────────────────────
# 1. Data Loading
# ─────────────────────────────────────────────
def load_data(filepath: str, sample_size: int = 5000) -> pd.DataFrame:
    """Load CSV and map Score → binary sentiment label."""
    print(f"\n{'='*55}")
    print("STEP 1: Loading Data")
    print('='*55)

    df = pd.read_csv(filepath)
    print(f"Total records loaded : {len(df)}")
    print(f"Columns              : {list(df.columns)}")

    # Keep only needed columns
    df = df[['Text', 'Score']].dropna()

    # Map Score: 1-2 → negative, 4-5 → positive  (drop neutral 3)
    df = df[df['Score'] != 3].copy()
    df['sentiment'] = df['Score'].apply(lambda s: 'positive' if s >= 4 else 'negative')
    df.rename(columns={'Text': 'review'}, inplace=True)

    # Sample for speed (reproducible)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    print(f"Records after filtering (no neutral): {len(df)}")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    return df[['review', 'sentiment']]


# ─────────────────────────────────────────────
# 2. Text Preprocessing
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lowercase → remove punctuation → remove stopwords → tokenize."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and t.isalpha()]
    return ' '.join(tokens)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print("STEP 2: Preprocessing")
    print('='*55)

    # Show before/after example
    sample = df['review'].iloc[0]
    cleaned = clean_text(sample)
    print(f"\nOriginal  : {sample[:200]}")
    print(f"\nCleaned   : {cleaned[:200]}")

    df['clean_review'] = df['review'].apply(clean_text)
    return df


# ─────────────────────────────────────────────
# 3. Aspect Extraction (spaCy nouns)
# ─────────────────────────────────────────────
def extract_aspects(text: str) -> list:
    """Return list of noun lemmas from the raw (uncleaned) text."""
    doc = nlp(text[:500])  # cap length for speed
    return [token.lemma_.lower() for token in doc
            if token.pos_ in ('NOUN', 'PROPN') and not token.is_stop]


def add_aspects(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print("STEP 3: Aspect Extraction")
    print('='*55)

    df['aspects'] = df['review'].apply(extract_aspects)

    # Show a few examples
    for i in range(3):
        print(f"\nReview    : {df['review'].iloc[i][:120]}")
        print(f"Aspects   : {df['aspects'].iloc[i]}")
    return df


# ─────────────────────────────────────────────
# 4. Feature Extraction
# ─────────────────────────────────────────────
def extract_features(df: pd.DataFrame, max_features: int = 5000):
    print(f"\n{'='*55}")
    print("STEP 4: Feature Extraction (BoW & TF-IDF)")
    print('='*55)

    X = df['clean_review']
    y = df['sentiment']

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    bow_vec  = CountVectorizer(max_features=max_features)
    tfidf_vec = TfidfVectorizer(max_features=max_features)

    # Fit only on the training split to avoid leaking test-set vocabulary.
    X_bow_train = bow_vec.fit_transform(X_train_text)
    X_bow_test = bow_vec.transform(X_test_text)
    X_tfidf_train = tfidf_vec.fit_transform(X_train_text)
    X_tfidf_test = tfidf_vec.transform(X_test_text)

    print(f"BoW train/test shapes   : {X_bow_train.shape} / {X_bow_test.shape}")
    print(f"TF-IDF train/test shapes: {X_tfidf_train.shape} / {X_tfidf_test.shape}")
    return (
        X_bow_train,
        X_bow_test,
        X_tfidf_train,
        X_tfidf_test,
        y_train,
        y_test,
        bow_vec,
        tfidf_vec,
    )


# ─────────────────────────────────────────────
# 5 & 6. Model Training + Evaluation
# ─────────────────────────────────────────────
def train_evaluate(X_train, X_test, y_train, y_test, feature_name: str) -> dict:
    """Train 3 models on pre-split features and evaluate on the holdout set."""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes':         MultinomialNB(),
        'SVM (Linear)':        LinearSVC(max_iter=2000, random_state=42),
    }

    results = {}
    conf_matrices = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label='positive', zero_division=0)
        rec  = recall_score(y_test, y_pred, pos_label='positive', zero_division=0)
        f1   = f1_score(y_test, y_pred, pos_label='positive', zero_division=0)
        cm   = confusion_matrix(y_test, y_pred, labels=['positive', 'negative'])

        results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}
        conf_matrices[name] = cm

        print(f"\n--- {name} [{feature_name}] ---")
        print(classification_report(y_test, y_pred, zero_division=0))

    return results, conf_matrices, y_test, models


# ─────────────────────────────────────────────
# 7. Result Comparison Table
# ─────────────────────────────────────────────
def print_comparison(results_bow: dict, results_tfidf: dict):
    print(f"\n{'='*55}")
    print("STEP 7: Model Comparison Table")
    print('='*55)

    rows = []
    for name, metrics in results_bow.items():
        rows.append({'Model': name, 'Features': 'BoW', **metrics})
    for name, metrics in results_tfidf.items():
        rows.append({'Model': name, 'Features': 'TF-IDF', **metrics})

    cmp_df = pd.DataFrame(rows).set_index(['Model', 'Features'])
    cmp_df = cmp_df.round(4)
    print(cmp_df.to_string())
    return cmp_df


# ─────────────────────────────────────────────
# 8. Visualizations
# ─────────────────────────────────────────────
def plot_confusion_matrices(conf_matrices: dict, feature_name: str):
    """Plot confusion matrices for all models side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Confusion Matrices — {feature_name}', fontsize=14)

    for ax, (name, cm) in zip(axes, conf_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Positive', 'Negative'],
                    yticklabels=['Positive', 'Negative'])
        ax.set_title(name)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.tight_layout()
    fname = f"confusion_matrix_{feature_name.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=120)
    print(f"Saved: {fname}")
    plt.show()


def plot_accuracy_comparison(results_bow: dict, results_tfidf: dict):
    """Bar chart comparing model accuracies across feature types."""
    models = list(results_bow.keys())
    bow_acc   = [results_bow[m]['Accuracy']   for m in models]
    tfidf_acc = [results_tfidf[m]['Accuracy'] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, bow_acc,   width, label='BoW',    color='steelblue')
    bars2 = ax.bar(x + width/2, tfidf_acc, width, label='TF-IDF', color='coral')

    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison: BoW vs TF-IDF')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=10)
    ax.set_ylim(0, 1.05)
    ax.legend()

    for bar in bars1 + bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig("accuracy_comparison.png", dpi=120)
    print("Saved: accuracy_comparison.png")
    plt.show()


# ─────────────────────────────────────────────
# BONUS: Most Frequent Aspects
# ─────────────────────────────────────────────
def plot_top_aspects(df: pd.DataFrame, top_n: int = 20):
    """Bar chart of the most frequently extracted aspect terms."""
    from collections import Counter
    all_aspects = [a for aspects in df['aspects'] for a in aspects]
    counter = Counter(all_aspects)
    common = counter.most_common(top_n)

    terms, counts = zip(*common)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(terms[::-1], counts[::-1], color='mediumseagreen')
    ax.set_xlabel('Frequency')
    ax.set_title(f'Top {top_n} Most Frequent Aspects')
    plt.tight_layout()
    plt.savefig("top_aspects.png", dpi=120)
    print("Saved: top_aspects.png")
    plt.show()


# ─────────────────────────────────────────────
# 9. Sample Prediction Demo
# ─────────────────────────────────────────────
def demo_prediction(sample_review: str, tfidf_vec, models_tfidf: dict):
    """Show aspects + predicted sentiment for a sample review."""
    print(f"\n{'='*55}")
    print("DEMO: Sample Review Prediction")
    print('='*55)
    print(f"Review   : {sample_review}")

    aspects = extract_aspects(sample_review)
    print(f"Aspects  : {aspects}")

    cleaned = clean_text(sample_review)
    X_sample = tfidf_vec.transform([cleaned])

    for name, model in models_tfidf.items():
        pred = model.predict(X_sample)[0]
        print(f"{name:25s} -> {pred}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    # 1. Load
    df = load_data('Reviews.csv', sample_size=5000)

    # 2. Preprocess
    df = preprocess(df)

    # 3. Aspect extraction
    df = add_aspects(df)

    # 4. Feature extraction
    X_bow_train, X_bow_test, X_tfidf_train, X_tfidf_test, y_train, y_test, bow_vec, tfidf_vec = extract_features(df)

    # 5 & 6. Train + evaluate
    print(f"\n{'='*55}")
    print("STEP 5 & 6: Training & Evaluation — BoW")
    print('='*55)
    results_bow, cm_bow, _, _ = train_evaluate(
        X_bow_train, X_bow_test, y_train, y_test, 'BoW'
    )

    print(f"\n{'='*55}")
    print("STEP 5 & 6: Training & Evaluation — TF-IDF")
    print('='*55)
    results_tfidf, cm_tfidf, _, models_tfidf = train_evaluate(
        X_tfidf_train, X_tfidf_test, y_train, y_test, 'TF-IDF'
    )

    # 7. Comparison table
    cmp_df = print_comparison(results_bow, results_tfidf)

    # 8. Visualizations
    plot_confusion_matrices(cm_bow,   'BoW')
    plot_confusion_matrices(cm_tfidf, 'TF-IDF')
    plot_accuracy_comparison(results_bow, results_tfidf)

    # BONUS: top aspects
    plot_top_aspects(df, top_n=20)

    # 9. Demo prediction
    demo_prediction(
        "The camera is great but battery is poor",
        tfidf_vec, models_tfidf
    )

    print("\nAll done. Check the saved PNG files for plots.")
