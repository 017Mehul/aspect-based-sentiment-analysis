# Aspect-Based Sentiment Analysis: Reproducible Baseline and Evaluation Plan

**Course area:** Natural Language Processing  
**Dataset:** Amazon Fine Food Reviews  
**Status:** Updated methodology draft — regenerate quantitative results from the current code before submission.

## 1. Abstract

This project compares classical machine-learning baselines for review-level sentiment classification and demonstrates a separate, interpretable heuristic for assigning sentiment evidence to candidate aspects. The dataset contains star ratings for entire reviews; it does not provide gold sentiment labels for each aspect. Consequently, the review-level models cannot be presented as supervised aspect-based sentiment analysis (ABSA), and the aspect heuristic must not be described as benchmark-validated.

## 2. Problem definition

Document-level sentiment classification predicts one polarity for an entire review. ABSA instead identifies aspects (explicit product features or targets) and predicts sentiment toward each aspect. For example, “The coffee is excellent, but the packaging is terrible” should yield positive sentiment for coffee and negative sentiment for packaging.

The Amazon Fine Food Reviews rating is only a review-level label. It cannot establish whether the author likes every feature mentioned. Proper supervised ABSA evaluation requires a dataset with aspect terms/spans and their sentiment annotations.

## 3. Data and labels

The script expects `Reviews.csv` with `Text` and `Score` columns. It removes missing rows and neutral 3-star reviews, mapping scores 1–2 to negative and 4–5 to positive. It supports reproducible sampling and validates required columns and class presence.

The data file is not stored in the repository. Record the exact dataset version, number of rows after filtering, class distribution and sample size when reporting an experiment.

## 4. Preprocessing

Text normalization lowercases input and removes punctuation while preserving apostrophes and negation terms. Stopwords are not indiscriminately removed because words such as “not” can reverse sentiment. The review-level vectorizer uses unigrams and bigrams, which retain some local phrase information.

## 5. Review-level baseline models

The implementation compares six configurations:

| Classifier | Representation |
|---|---|
| Logistic Regression with class weighting | Bag of Words |
| Multinomial Naive Bayes | Bag of Words |
| Linear SVM with class weighting | Bag of Words |
| Logistic Regression with class weighting | TF-IDF |
| Multinomial Naive Bayes | TF-IDF |
| Linear SVM with class weighting | TF-IDF |

A stratified 80/20 train/test split and fixed random seed are used. Each vectorizer is fitted only on the training portion via a scikit-learn pipeline. The test set is held out for evaluation.

## 6. Metrics and outputs

The script writes fresh metrics to `outputs/model_results.csv` and `outputs/model_results.json`, per-model classification reports, confusion matrices, comparison plots and a serialized best review-level pipeline. Report accuracy alongside macro-F1, weighted-F1 and per-class precision, recall and F1. In an imbalanced dataset, accuracy alone can conceal poor minority-class performance.

**Quantitative results are intentionally not hard-coded in this report.** Run the current code and transfer the actual generated metrics into the submitted report. Previously documented figures may not match the refactored pipeline and must not be claimed as reproduced without verification.

## 7. Aspect extraction and heuristic sentiment demo

spaCy noun chunks are used to extract candidate aspect phrases, with noun-lemma fallback. This is a lightweight candidate extractor, not a complete aspect-term extraction model.

The demo assigns lexicon opinion words to the nearest candidate aspect in a limited token window, flips polarity for nearby negation, and returns “unknown” when it finds no evidence. It is interpretable and useful for illustrating the task, but it is not a learned ABSA classifier. It can fail on sarcasm, implicit aspects, domain-specific expressions, long-distance relations, complex negation and contrastive syntax.

## 8. Correct evaluation plan for true ABSA

To make a research-grade ABSA claim:

1. Select an annotated ABSA corpus with aspect terms/spans and sentiment labels, such as an appropriate SemEval ABSA dataset.
2. Document the exact dataset release, license, splits and label mapping.
3. Evaluate aspect extraction separately using span/term precision, recall and F1.
4. Evaluate aspect sentiment classification against gold aspect annotations using per-class precision, recall, macro-F1 and a confusion matrix.
5. Compare a documented baseline against a stronger contextual model, using the same splits.
6. Inspect errors on multiple aspects, mixed polarity, negation, implicit targets and unseen terms.
7. Keep review-level Amazon results separate from aspect-level results; do not substitute star ratings for aspect-level labels.

## 9. Limitations

- Amazon Fine Food Reviews provides review-level, not aspect-level, ground truth.
- The candidate extractor can miss implicit aspects and may include irrelevant nouns.
- The small sentiment lexicon is a demonstration baseline and is not suitable for production decisions.
- BoW/TF-IDF cannot fully represent contextual semantics, sarcasm or long-distance dependencies.
- Class weighting does not eliminate class imbalance.
- Results depend on dataset version, sample size, dependencies and train/test split; publish those details with the actual results.

## 10. Reproducibility

Install dependencies from `requirements.txt`, download the spaCy English model as described in the README, place `Reviews.csv` in the project root, and run:

```bash
python absa_project.py
pytest -q
```

The workflow runs unit tests in GitHub Actions. A successful CI run verifies the tests, not the full dataset experiment. Record actual run output before asserting final metrics.

## 11. References

1. Liu, B. (2012). *Sentiment Analysis and Opinion Mining*. Morgan & Claypool Publishers. https://doi.org/10.2200/S00416ED1V01Y201204HLT016
2. Pang, B., & Lee, L. (2008). Opinion mining and sentiment analysis. *Foundations and Trends in Information Retrieval*, 2(1–2), 1–135. https://doi.org/10.1561/1500000011
3. Pontiki, M., et al. (2014). SemEval-2014 Task 4: Aspect Based Sentiment Analysis. *Proceedings of SemEval 2014*, 27–35.
4. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. https://arxiv.org/abs/1810.04805
5. Scikit-learn developers. *scikit-learn documentation*. https://scikit-learn.org/stable/
6. spaCy developers. *spaCy documentation*. https://spacy.io/
