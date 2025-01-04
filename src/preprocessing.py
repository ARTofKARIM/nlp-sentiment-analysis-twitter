"""Text preprocessing pipeline for Twitter data."""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List


class TextPreprocessor:
    """Comprehensive text cleaning pipeline for tweets."""

    def __init__(self, config=None):
        self.config = config or {}
        try:
            self.stop_words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            nltk.download("wordnet", quiet=True)
            nltk.download("punkt", quiet=True)
            self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def to_lowercase(self, text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()

    def remove_urls(self, text: str) -> str:
        """Remove HTTP/HTTPS URLs from text."""
        return re.sub(r"https?://\S+|www\.\S+", "", text)

    def remove_mentions(self, text: str) -> str:
        """Remove @username mentions."""
        return re.sub(r"@\w+", "", text)

    def remove_hashtags(self, text: str) -> str:
        """Remove hashtag symbols (keep the word)."""
        return re.sub(r"#(\w+)", r"\1", text)

    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation characters."""
        return text.translate(str.maketrans("", "", string.punctuation))

    def remove_numbers(self, text: str) -> str:
        """Remove numeric characters."""
        return re.sub(r"\d+", "", text)

    def remove_extra_whitespace(self, text: str) -> str:
        """Collapse multiple whitespace into single space."""
        return re.sub(r"\s+", " ", text).strip()

    def remove_stopwords(self, text: str) -> str:
        """Remove English stopwords."""
        words = text.split()
        filtered = [w for w in words if w not in self.stop_words]
        return " ".join(filtered)

    def lemmatize(self, text: str) -> str:
        """Apply WordNet lemmatization."""
        words = text.split()
        lemmatized = [self.lemmatizer.lemmatize(w) for w in words]
        return " ".join(lemmatized)

    def remove_emojis(self, text: str) -> str:
        """Remove emoji characters."""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub("", text)

    def preprocess(self, text: str) -> str:
        """Run the full preprocessing pipeline on a single text."""
        if not isinstance(text, str):
            return ""
        text = self.to_lowercase(text)
        text = self.remove_urls(text)
        text = self.remove_mentions(text)
        text = self.remove_hashtags(text)
        text = self.remove_emojis(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize(text)
        text = self.remove_extra_whitespace(text)
        return text

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts."""
        processed = [self.preprocess(t) for t in texts]
        empty_count = sum(1 for t in processed if len(t) == 0)
        print(f"Preprocessed {len(texts)} texts ({empty_count} empty after cleaning)")
        return processed
