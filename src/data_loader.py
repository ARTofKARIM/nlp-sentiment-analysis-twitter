"""Data loading and exploration for Twitter sentiment dataset."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml


class TwitterDataLoader:
    """Loads and prepares Twitter sentiment data for analysis."""

    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.data = None
        self.label_mapping = {"negative": 0, "neutral": 1, "positive": 2}

    def load(self, filepath=None):
        """Load the Twitter sentiment dataset."""
        path = filepath or self.config["data"]["path"]
        self.data = pd.read_csv(path, encoding="utf-8")
        print(f"Loaded {len(self.data)} tweets")
        return self.data

    def describe(self):
        """Generate dataset summary statistics."""
        if self.data is None:
            raise ValueError("Data not loaded")
        text_col = self.config["data"]["text_column"]
        label_col = self.config["data"]["label_column"]

        summary = {
            "total_samples": len(self.data),
            "class_distribution": self.data[label_col].value_counts().to_dict(),
            "avg_text_length": self.data[text_col].str.len().mean(),
            "max_text_length": self.data[text_col].str.len().max(),
            "missing_values": self.data.isnull().sum().to_dict(),
        }
        for key, val in summary.items():
            print(f"  {key}: {val}")
        return summary

    def encode_labels(self):
        """Convert string labels to numeric values."""
        label_col = self.config["data"]["label_column"]
        self.data["label_encoded"] = self.data[label_col].map(self.label_mapping)
        return self.data

    def split(self):
        """Split into train and test sets."""
        text_col = self.config["data"]["text_column"]
        X = self.data[text_col].values
        y = self.data["label_encoded"].values if "label_encoded" in self.data.columns else self.data[self.config["data"]["label_column"]].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config["data"]["test_size"],
            random_state=self.config["data"]["random_state"],
            stratify=y,
        )
        print(f"Train: {len(X_train)} | Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test
