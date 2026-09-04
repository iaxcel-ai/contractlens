"""
Train the clause classifier.

WHAT THIS DOES: reads the labeled clause dataset, turns each clause into
numbers a machine learning model can use (TF-IDF), trains a simple
classifier (logistic regression) to predict the category, and saves both
the vectorizer and the trained model to disk so the app doesn't need to
retrain every time it starts.

WHY TF-IDF + LOGISTIC REGRESSION instead of a transformer model: with only
~85 training examples, a big pretrained language model would be massive
overkill (slow to run, hard to explain, and not actually more accurate on a
dataset this small). TF-IDF + logistic regression is:
  - fast to train (seconds, not GPU-hours)
  - easy to understand (each word gets a weight per category)
  - a well-established, reliable baseline for small text classification
    problems like this one

WHAT IS TF-IDF: "Term Frequency - Inverse Document Frequency". It turns each
clause into a vector of numbers, one per word in the vocabulary. A word gets
a higher score in a clause if it appears often in THAT clause (term
frequency) but is rare across ALL clauses (inverse document frequency) -
so common words like "the" or "agreement" get low scores, while distinctive
words like "arbitration" or "auto-renew" get high scores. That's exactly
the kind of signal a classifier needs to tell categories apart.

WHAT IS LOGISTIC REGRESSION: a simple, well-understood statistical model
that learns a weight for each input feature (here, each TF-IDF word score)
per category, then picks the category whose weighted sum is highest. It's
the standard "first model to try" for text classification.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "clauses_dataset.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "clause_classifier.joblib")


def train_and_save_model():
    # 1. Load the labeled dataset (clause text + category label per row).
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} labeled clauses across {df['category'].nunique()} categories.")

    # 2. Split into train/test sets so we can check how well the model
    #    generalizes to clauses it hasn't seen before, instead of just
    #    memorizing the training data.
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["category"], test_size=0.2, random_state=42, stratify=df["category"]
    )

    # 3. Convert clause text into TF-IDF vectors.
    #    ngram_range=(1, 2) means we look at both single words ("cancel")
    #    and two-word phrases ("without notice"), which helps catch
    #    contract-specific phrasing that a single word would miss.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
    x_train_vectors = vectorizer.fit_transform(x_train)
    x_test_vectors = vectorizer.transform(x_test)

    # 4. Train the logistic regression classifier on the vectorized text.
    #    class_weight="balanced" tells it to pay equal attention to every
    #    category even though we don't have exactly the same number of
    #    examples per category.
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    classifier.fit(x_train_vectors, y_train)

    # 5. Check accuracy on the held-out test clauses and print a report.
    predictions = classifier.predict(x_test_vectors)
    print("\nClassification report on held-out test clauses:")
    print(classification_report(y_test, predictions, zero_division=0))

    # 6. Re-train on the FULL dataset (train + test) before saving, so the
    #    final saved model benefits from every labeled example we have.
    #    The test split above was only to measure accuracy honestly.
    full_vectors = vectorizer.fit_transform(df["text"])
    classifier.fit(full_vectors, df["category"])

    # 7. Save the vectorizer and classifier to disk with joblib, so the
    #    Streamlit app can load them instantly instead of retraining.
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(classifier, CLASSIFIER_PATH)
    print(f"\nSaved vectorizer to {VECTORIZER_PATH}")
    print(f"Saved classifier to {CLASSIFIER_PATH}")


if __name__ == "__main__":
    train_and_save_model()
