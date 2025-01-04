"""LSTM-based deep learning model for sentiment classification."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks


class SentimentLSTM:
    """LSTM model for multi-class sentiment classification."""

    def __init__(self, vocab_size, max_len, embedding_dim=128, lstm_units=64,
                 num_classes=3, dropout=0.3):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.num_classes = num_classes
        self.dropout = dropout
        self.model = None
        self.history = None

    def build(self):
        """Build the LSTM architecture."""
        self.model = keras.Sequential([
            layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_len),
            layers.SpatialDropout1D(0.2),
            layers.Bidirectional(layers.LSTM(self.lstm_units, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(self.lstm_units // 2)),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout),
            layers.Dense(32, activation="relu"),
            layers.Dropout(self.dropout),
            layers.Dense(self.num_classes, activation="softmax"),
        ])
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        print(self.model.summary())
        return self

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=64):
        """Train the LSTM model with callbacks."""
        cb = [
            callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss"),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6),
        ]
        val_data = (X_val, y_val) if X_val is not None else None
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            callbacks=cb,
            verbose=1,
        )
        return self.history

    def predict(self, X):
        """Predict class labels."""
        probs = self.model.predict(X, verbose=0)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        """Get class probabilities."""
        return self.model.predict(X, verbose=0)

    def evaluate(self, X_test, y_test):
        """Evaluate model on test data."""
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"LSTM - Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
        return {"loss": loss, "accuracy": accuracy}

    def save(self, path="models/lstm_sentiment.h5"):
        """Save model weights."""
        self.model.save(path)
        print(f"Model saved to {path}")

    def load(self, path="models/lstm_sentiment.h5"):
        """Load model weights."""
        self.model = keras.models.load_model(path)
        print(f"Model loaded from {path}")
