"""
Main execution file for the Breast Cancer Classification Project.

This script loads the dataset, splits the data into training and testing sets,
scales the features, tunes the models, trains them, and evaluates their performance.
"""

import sys

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models import CancerModels
from evaluator import ModelEvaluator
from tuner import ModelTuner


def main():
    """Main program execution."""

    try:
        # Load dataset
        data = load_breast_cancer()
        X = data.data
        y = data.target

        print("Dataset loaded successfully.")
        print(f"Total samples: {X.shape[0]}")
        print(f"Total features: {X.shape[1]}")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        print("\nData successfully split into training and testing sets.")

        # Feature scaling
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        print("Feature scaling completed.")

        # Load models
        models = CancerModels().get_models()

        # Initialize evaluator and tuner
        evaluator = ModelEvaluator()
        tuner = ModelTuner()

        tuned_models = {}

        print("\n----- Hyperparameter Tuning -----")

        for name, model in models.items():
            print(f"Tuning {name}...")
            best_model = tuner.tune(name, model, X_train, y_train)
            tuned_models[name] = best_model

        print("\n----- Model Evaluation -----")

        for name, model in tuned_models.items():
            print(f"\n{name}")
            print("-" * 40)

            model.fit(X_train, y_train)

            results = evaluator.evaluate(model, X_test, y_test)

            for metric, value in results.items():
                print(f"{metric}: {value}")

    except Exception as error:
        print("An error occurred:", error)
        sys.exit(1)


if __name__ == "__main__":
    main()