"""Visualization module for sentiment analysis results."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud


class SentimentVisualizer:
    """Generates plots for sentiment analysis."""

    def __init__(self, output_dir="results/"):
        self.output_dir = output_dir

    def plot_class_distribution(self, labels, save=True):
        """Plot the distribution of sentiment classes."""
        fig, ax = plt.subplots(figsize=(8, 5))
        unique, counts = np.unique(labels, return_counts=True)
        colors = ["#e74c3c", "#95a5a6", "#2ecc71"]
        ax.bar(unique, counts, color=colors[:len(unique)])
        ax.set_xlabel("Sentiment Class")
        ax.set_ylabel("Count")
        ax.set_title("Sentiment Distribution")
        if save:
            fig.savefig(f"{self.output_dir}class_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_wordcloud(self, texts, title="Word Cloud", save=True):
        """Generate word cloud from texts."""
        text = " ".join(texts)
        wc = WordCloud(width=800, height=400, background_color="white", max_words=200).generate(text)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=16)
        if save:
            safe = title.lower().replace(" ", "_")
            fig.savefig(f"{self.output_dir}wordcloud_{safe}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_confusion_matrix(self, y_true, y_pred, labels, model_name, save=True):
        """Plot confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        ax.set_title(f"Confusion Matrix - {model_name}")
        if save:
            fig.savefig(f"{self.output_dir}cm_{model_name.lower()}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_model_comparison(self, results_dict, save=True):
        """Bar chart comparing model accuracies."""
        models = list(results_dict.keys())
        accs = [results_dict[m]["accuracy"] for m in models]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(models, accs, color="steelblue")
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Comparison")
        ax.set_ylim(0, 1)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{acc:.3f}", ha="center", fontsize=10)
        if save:
            fig.savefig(f"{self.output_dir}model_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_training_history(self, history, save=True):
        """Plot LSTM training loss and accuracy curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(history.history["loss"], label="Train Loss")
        if "val_loss" in history.history:
            ax1.plot(history.history["val_loss"], label="Val Loss")
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Epoch")
        ax1.legend()

        ax2.plot(history.history["accuracy"], label="Train Acc")
        if "val_accuracy" in history.history:
            ax2.plot(history.history["val_accuracy"], label="Val Acc")
        ax2.set_title("Training Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.legend()
        if save:
            fig.savefig(f"{self.output_dir}training_history.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
