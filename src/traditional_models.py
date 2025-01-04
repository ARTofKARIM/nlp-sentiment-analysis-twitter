"""Traditional ML classifiers for sentiment analysis."""

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import numpy as np


class SentimentClassifierTraditional:
    """Wrapper for traditional ML sentiment classifiers."""

    def __init__(self):
        self.models = {}
        self.best_params = {}

    def train_naive_bayes(self, X_train, y_train, alpha=1.0):
        """Train Multinomial Naive Bayes classifier."""
        model = MultinomialNB(alpha=alpha)
        model.fit(X_train, y_train)
        self.models["naive_bayes"] = model
        train_acc = model.score(X_train, y_train)
        print(f"Naive Bayes - Train accuracy: {train_acc:.4f}")
        return model

    def train_svm(self, X_train, y_train, C=1.0):
        """Train Linear SVM classifier."""
        model = LinearSVC(C=C, max_iter=5000, random_state=42)
        model.fit(X_train, y_train)
        self.models["svm"] = model
        train_acc = accuracy_score(y_train, model.predict(X_train))
        print(f"SVM - Train accuracy: {train_acc:.4f}")
        return model

    def train_logistic_regression(self, X_train, y_train, C=1.0):
        """Train Logistic Regression classifier."""
        model = LogisticRegression(C=C, max_iter=1000, random_state=42, solver="lbfgs", multi_class="multinomial")
        model.fit(X_train, y_train)
        self.models["logistic_regression"] = model
        train_acc = model.score(X_train, y_train)
        print(f"Logistic Regression - Train accuracy: {train_acc:.4f}")
        return model

    def hyperparameter_search(self, X_train, y_train, model_name="svm"):
        """Perform grid search for hyperparameter tuning."""
        param_grids = {
            "naive_bayes": {"alpha": [0.01, 0.1, 0.5, 1.0, 2.0]},
            "svm": {"C": [0.01, 0.1, 1.0, 10.0]},
            "logistic_regression": {"C": [0.01, 0.1, 1.0, 10.0]},
        }
        estimators = {
            "naive_bayes": MultinomialNB(),
            "svm": LinearSVC(max_iter=5000, random_state=42),
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        }

        grid = GridSearchCV(
            estimators[model_name], param_grids[model_name],
            cv=5, scoring="f1_weighted", n_jobs=-1, verbose=0,
        )
        grid.fit(X_train, y_train)
        self.best_params[model_name] = grid.best_params_
        self.models[model_name] = grid.best_estimator_
        print(f"{model_name} best params: {grid.best_params_} (F1: {grid.best_score_:.4f})")
        return grid.best_estimator_

    def predict(self, model_name, X):
        """Get predictions from a trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained")
        return self.models[model_name].predict(X)

    def evaluate(self, model_name, X_test, y_test, labels=None):
        """Evaluate a model on test data."""
        y_pred = self.predict(model_name, X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
        print(f"\n{model_name} - Test accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=labels))
        return {"accuracy": acc, "report": report, "predictions": y_pred}
