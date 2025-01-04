"""Main pipeline for Twitter sentiment analysis."""

import argparse
import yaml
import numpy as np
from src.data_loader import TwitterDataLoader
from src.preprocessing import TextPreprocessor
from src.feature_extraction import TfidfFeatureExtractor, SequenceEncoder
from src.traditional_models import SentimentClassifierTraditional
from src.lstm_model import SentimentLSTM


def main():
    parser = argparse.ArgumentParser(description="Twitter Sentiment Analysis")
    parser.add_argument("--data", type=str, help="Path to dataset")
    parser.add_argument("--model", choices=["all", "nb", "svm", "lr", "lstm"], default="all")
    args = parser.parse_args()

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    # Load and preprocess
    loader = TwitterDataLoader()
    loader.load(args.data)
    loader.describe()
    loader.encode_labels()
    X_train, X_test, y_train, y_test = loader.split()

    preprocessor = TextPreprocessor(config["preprocessing"])
    X_train_clean = preprocessor.preprocess_batch(X_train.tolist())
    X_test_clean = preprocessor.preprocess_batch(X_test.tolist())

    labels = ["negative", "neutral", "positive"]

    # Traditional models with TF-IDF
    if args.model in ["all", "nb", "svm", "lr"]:
        tfidf = TfidfFeatureExtractor(
            max_features=config["models"]["tfidf"]["max_features"],
            ngram_range=tuple(config["models"]["tfidf"]["ngram_range"]),
        )
        X_train_tfidf = tfidf.fit_transform(X_train_clean)
        X_test_tfidf = tfidf.transform(X_test_clean)

        classifier = SentimentClassifierTraditional()
        if args.model in ["all", "nb"]:
            classifier.train_naive_bayes(X_train_tfidf, y_train)
            classifier.evaluate("naive_bayes", X_test_tfidf, y_test, labels)
        if args.model in ["all", "svm"]:
            classifier.train_svm(X_train_tfidf, y_train)
            classifier.evaluate("svm", X_test_tfidf, y_test, labels)
        if args.model in ["all", "lr"]:
            classifier.train_logistic_regression(X_train_tfidf, y_train)
            classifier.evaluate("logistic_regression", X_test_tfidf, y_test, labels)

    # LSTM model
    if args.model in ["all", "lstm"]:
        lstm_config = config["models"]["lstm"]
        encoder = SequenceEncoder(max_words=lstm_config["max_words"], max_len=lstm_config["max_len"])
        X_train_seq = encoder.fit_encode(X_train_clean)
        X_test_seq = encoder.encode(X_test_clean)

        lstm = SentimentLSTM(
            vocab_size=encoder.vocab_size,
            max_len=lstm_config["max_len"],
            embedding_dim=lstm_config["embedding_dim"],
            lstm_units=lstm_config["lstm_units"],
            dropout=lstm_config["dropout"],
        )
        lstm.build()
        lstm.train(X_train_seq, y_train, X_test_seq, y_test,
                   epochs=lstm_config["epochs"], batch_size=lstm_config["batch_size"])
        lstm.evaluate(X_test_seq, y_test)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
