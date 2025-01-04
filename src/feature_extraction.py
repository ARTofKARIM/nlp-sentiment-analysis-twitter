"""Text vectorization and feature extraction for sentiment analysis."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import Tuple


class TfidfFeatureExtractor:
    """TF-IDF based feature extraction."""

    def __init__(self, max_features=10000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
        )

    def fit_transform(self, texts):
        """Fit on training texts and return TF-IDF matrix."""
        X = self.vectorizer.fit_transform(texts)
        print(f"TF-IDF matrix: {X.shape}")
        return X

    def transform(self, texts):
        """Transform new texts using fitted vectorizer."""
        return self.vectorizer.transform(texts)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()


class BowFeatureExtractor:
    """Bag of Words feature extraction."""

    def __init__(self, max_features=10000):
        self.vectorizer = CountVectorizer(max_features=max_features)

    def fit_transform(self, texts):
        X = self.vectorizer.fit_transform(texts)
        print(f"BoW matrix: {X.shape}")
        return X

    def transform(self, texts):
        return self.vectorizer.transform(texts)


class SequenceEncoder:
    """Encodes texts as padded integer sequences for deep learning models."""

    def __init__(self, max_words=20000, max_len=128):
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")

    def fit(self, texts):
        """Fit tokenizer on training texts."""
        self.tokenizer.fit_on_texts(texts)
        vocab_size = min(len(self.tokenizer.word_index), self.max_words)
        print(f"Vocabulary size: {vocab_size}")
        return self

    def encode(self, texts) -> np.ndarray:
        """Convert texts to padded sequences."""
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len, padding="post", truncating="post")
        return padded

    def fit_encode(self, texts) -> np.ndarray:
        """Fit and encode in one step."""
        self.fit(texts)
        return self.encode(texts)

    @property
    def vocab_size(self):
        return min(len(self.tokenizer.word_index) + 1, self.max_words)
