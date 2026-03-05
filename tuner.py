"""
This module performs hyperparameter tuning
using GridSearchCV.
"""

from sklearn.model_selection import GridSearchCV


class ModelTuner:
    """
    A class responsible for hyperparameter tuning
    of machine learning models.
    """

    def tune(self, model_name, model, X_train, y_train):
        """
        Tune model hyperparameters using GridSearchCV.
        """

        param_grids = {

            "Logistic Regression": {
                "C": [0.01, 0.1, 1, 10]
            },

            "Random Forest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 5, 10]
            },

            "Support Vector Machine": {
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto"]
            }
        }

        grid = GridSearchCV(
            model,
            param_grids[model_name],
            cv=5,
            scoring="roc_auc"
        )

        grid.fit(X_train, y_train)

        print("Best Parameters:", grid.best_params_)

        return grid.best_estimator_