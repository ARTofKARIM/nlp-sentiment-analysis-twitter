# Twitter Sentiment Analysis

A comprehensive NLP pipeline for sentiment analysis on Twitter data using both traditional machine learning (Naive Bayes, SVM, Logistic Regression) and deep learning (Bidirectional LSTM) approaches.

## Overview

This project classifies tweets into three sentiment categories: **positive**, **neutral**, and **negative**. It implements a full NLP pipeline from text preprocessing to model evaluation and comparison.

## Architecture

```
nlp-sentiment-analysis-twitter/
├── src/
│   ├── data_loader.py          # Dataset loading and splitting
│   ├── preprocessing.py        # Tweet text cleaning pipeline
│   ├── feature_extraction.py   # TF-IDF, BoW, sequence encoding
│   ├── traditional_models.py   # Naive Bayes, SVM, Logistic Regression
│   ├── lstm_model.py           # Bidirectional LSTM with embeddings
│   ├── evaluation.py           # Metrics and model comparison
│   └── visualization.py        # Word clouds, confusion matrices, plots
├── config/
│   └── config.yaml
├── tests/
│   └── test_preprocessing.py
└── main.py
```

## Models

| Model | Type | Features |
|-------|------|----------|
| Multinomial NB | Traditional | TF-IDF |
| Linear SVM | Traditional | TF-IDF |
| Logistic Regression | Traditional | TF-IDF |
| Bidirectional LSTM | Deep Learning | Word Embeddings |

## Text Preprocessing Pipeline

1. Lowercase conversion
2. URL removal
3. @mention removal
4. Hashtag normalization
5. Emoji removal
6. Punctuation and number removal
7. Stopword removal
8. Lemmatization

## Installation

```bash
git clone https://github.com/mouachiqab/nlp-sentiment-analysis-twitter.git
cd nlp-sentiment-analysis-twitter
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

## Usage

```bash
# Run all models
python main.py --data data/twitter_sentiment.csv --model all

# Run specific model
python main.py --data data/twitter_sentiment.csv --model lstm
python main.py --data data/twitter_sentiment.csv --model svm
```

## Technologies

- Python 3.9+
- NLTK (text preprocessing)
- scikit-learn (traditional ML)
- TensorFlow/Keras (LSTM)
- matplotlib, seaborn, wordcloud (visualization)









