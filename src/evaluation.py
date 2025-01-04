"""Model evaluation and comparison for sentiment analysis."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, roc_auc_score,
)


class ModelEvaluator:
    """Evaluates and compares sentiment classification models."""

    def __init__(self):
        self.results = {}

    def evaluate(self, y_true, y_pred, model_name, labels=None):
        """Compute classification metrics."""
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)

        self.results[model_name] = {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "confusion_matrix": cm,
            "report": report,
        }
        return self.results[model_name]

    def comparison_table(self):
        """Generate comparison DataFrame of all models."""
        rows = []
        for name, m in self.results.items():
            rows.append({
                "Model": name,
                "Accuracy": f"{m['accuracy']:.4f}",
                "F1 (macro)": f"{m['f1_macro']:.4f}",
                "F1 (weighted)": f"{m['f1_weighted']:.4f}",
            })
        return pd.DataFrame(rows)
