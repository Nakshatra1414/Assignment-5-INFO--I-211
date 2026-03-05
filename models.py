"""
This module contains the machine learning models used in the project.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class CancerModels:
    """
    A class responsible for initializing and returning
    the classification models used in the project.
    """

    def __init__(self):
        """Initialize the models."""

        self.models = {

            "Logistic Regression": LogisticRegression(
                max_iter=5000
            ),

            "Random Forest": RandomForestClassifier(),

            "Support Vector Machine": SVC(
                probability=True
            )
        }

    def get_models(self):
        """Return dictionary of models."""
        return self.models