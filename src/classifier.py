"""
Load the trained classifier and use it to categorize new clauses.

This is separate from train_classifier.py so the Streamlit app can just
load an already-trained model (fast) instead of retraining every time it
starts (slow, and pointless to repeat).
"""

import os
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "clause_classifier.joblib")

CATEGORIES = [
    "fees",
    "auto-renewal",
    "cancellation",
    "data sharing",
    "liability",
    "dispute resolution",
    "other",
]


class ClauseClassifier:
    """Thin wrapper around the saved TF-IDF vectorizer + logistic regression model."""

    def __init__(self):
        if not (os.path.exists(VECTORIZER_PATH) and os.path.exists(CLASSIFIER_PATH)):
            raise FileNotFoundError(
                "Trained model files not found. Run `python src/train_classifier.py` first."
            )
        self.vectorizer = joblib.load(VECTORIZER_PATH)
        self.classifier = joblib.load(CLASSIFIER_PATH)

    def predict(self, clause_text: str) -> str:
        """Return the predicted category for a single clause."""
        vector = self.vectorizer.transform([clause_text])
        return self.classifier.predict(vector)[0]

    def predict_with_confidence(self, clause_text: str) -> tuple[str, float]:
        """Return (predicted category, confidence 0-1) using the model's probabilities."""
        vector = self.vectorizer.transform([clause_text])
        probabilities = self.classifier.predict_proba(vector)[0]
        best_index = probabilities.argmax()
        category = self.classifier.classes_[best_index]
        confidence = float(probabilities[best_index])
        return category, confidence
