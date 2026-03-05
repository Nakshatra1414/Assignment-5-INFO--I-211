"""
This module evaluates machine learning models using
multiple performance metrics.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


class ModelEvaluator:
    """
    A class responsible for evaluating classification models.
    """

    def evaluate(self, model, X_test, y_test):
        """
        Evaluate a trained model using several metrics.
        """

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        results = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions),
            "Recall": recall_score(y_test, predictions),
            "F1 Score": f1_score(y_test, predictions),
            "ROC AUC": roc_auc_score(y_test, probabilities),
            "Confusion Matrix": confusion_matrix(y_test, predictions)
        }

        return results