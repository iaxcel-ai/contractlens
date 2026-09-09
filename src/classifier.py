"""
Load the trained classifier and use it to categorize new clauses.

The classifier provides a primary category, but risk analysis can also use
secondary plausible categories. This prevents one uncertain classification
from completely blocking relevant red-flag checks.
"""

import os
import joblib


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "clause_classifier.joblib")


class ClauseClassifier:
    """Wrapper around the saved TF-IDF vectorizer and logistic regression model."""

    def __init__(self):
        if not (
            os.path.exists(VECTORIZER_PATH)
            and os.path.exists(CLASSIFIER_PATH)
        ):
            raise FileNotFoundError(
                "Trained model files not found. "
                "Run `python src/train_classifier.py` first."
            )

        self.vectorizer = joblib.load(VECTORIZER_PATH)
        self.classifier = joblib.load(CLASSIFIER_PATH)

    @property
    def categories(self) -> list[str]:
        """Return the categories learned by the trained model."""
        return list(self.classifier.classes_)

    def predict(self, clause_text: str) -> str:
        """Return the single most likely category."""
        vector = self.vectorizer.transform([clause_text])
        return self.classifier.predict(vector)[0]

    def predict_with_confidence(
        self,
        clause_text: str
    ) -> tuple[str, float]:
        """Return the most likely category and its probability."""

        predictions = self.predict_probabilities(clause_text)

        category, confidence = predictions[0]

        return category, confidence

    def predict_probabilities(
        self,
        clause_text: str
    ) -> list[tuple[str, float]]:
        """
        Return every category with its probability, sorted from
        most likely to least likely.
        """

        vector = self.vectorizer.transform([clause_text])

        probabilities = self.classifier.predict_proba(vector)[0]

        results = [
            (category, float(probability))
            for category, probability in zip(
                self.classifier.classes_,
                probabilities
            )
        ]

        return sorted(
            results,
            key=lambda item: item[1],
            reverse=True
        )

    def predict_plausible_categories(
        self,
        clause_text: str,
        threshold: float = 0.20,
        max_categories: int = 3
    ) -> list[tuple[str, float]]:
        """
        Return the most plausible categories.

        The highest prediction is always included. Secondary categories are
        included when their probability passes the threshold.

        This is useful for risk analysis because a clause can contain ideas
        related to multiple categories.
        """

        predictions = self.predict_probabilities(clause_text)

        plausible = [
            (category, probability)
            for category, probability in predictions
            if probability >= threshold
        ]

        if not plausible:
            plausible = [predictions[0]]

        return plausible[:max_categories]