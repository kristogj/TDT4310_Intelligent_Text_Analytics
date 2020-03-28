import pandas as pd
from transformer import TextNormalizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

from sklearn.externals import joblib
from utils import get_tagged_tweets, shuffle_data
import matplotlib.pyplot as plt
from nltk import word_tokenize

import tensorflow as tf
import numpy as np
from utils import plot


class TweetCorpusReader:
    def __init__(self, df, labels):
        self.df = df
        self.tweets = get_tagged_tweets(df)
        self.categories = [labels.index(c) for c in df["airline_sentiment"].tolist()]


def split_data(corpus, split_idx=0.1):
    """
    Split the tweets from the corpus into trainig and test data.
    :param corpus: TweetCorpusReader
    :param split_idx: double
    :return:
    """
    # Load the corpus data and labels for classification
    X, y = corpus.tweets, corpus.categories

    # Shuffle the tweets
    X, y = shuffle_data(X, y)
    split_idx = int(len(X) * split_idx)
    X_test, y_test = X[:split_idx], y[:split_idx]
    X_train, y_train = X[split_idx:], y[split_idx:]
    return X_train, y_train, X_test, y_test


def fit_model(X, y, model, saveto=None):
    """
    Fit the sklearn pipeline model and save it to file
    :param X: List of tagged tweets
    :param y: 0,1,2
    :param model: Pipeline
    :param saveto: string
    :return:
    """
    print("Starting Training")
    # Fit the model on entire data set
    model.fit(X, y)
    print("Saving Model")
    # Write to disk if specified
    if saveto:
        joblib.dump(model, saveto)
    return model


def test_model(X, y, model):
    print("Testing model")
    predictions = model.predict(X)
    acc = sum(predictions == y) / len(y)
    print("Test accuracy: {}".format(acc))


def preprocess(X, config, vocab_table=None):
    """
    Turn tweets from TweetCorpus into sequences of word ids. Needed for LSTM task.
    :param X: Tagged tweets
    :param vocab_table: dict() mapping from word to word id
    :return: X, vocab_table
    """
    print("Preprocessing data")
    normalizer = TextNormalizer()

    # Normalize text
    X = normalizer.transform(X)

    # Build vocabulary
    if not vocab_table:
        vocab = set()
        for tweet in X:
            words = set(word_tokenize(tweet))
            vocab = vocab.union(words)

        vocab = ["unk"] + list(vocab)
        vocab_table = {word: vocab.index(word) + 1 for word in vocab}

    # Convert each tweet to a sequence of word ids
    tweets = []
    for tweet in X:
        sequence = []
        for word in word_tokenize(tweet):
            if word in vocab_table.keys():
                sequence.append(vocab_table[word])
            else:
                sequence.append(vocab_table["unk"])
        tweets.append(sequence)
    X = tweets

    # Preprocess into equal sized sequence length
    X = tf.keras.preprocessing.sequence.pad_sequences(
        X, maxlen=config["max_length"], dtype='int32', padding='post', truncating='post', value=0)

    return X, vocab_table


def task_5a(corpus):
    """
    Convert the tweets in vectors and feed them to a MLP Classifier. Report accuracy.
    :param corpus: TweetCorpusReader
    :return:
    """
    # Set up the pipeline
    pipeline = Pipeline([
        ('norm', TextNormalizer()),  # can use KeyphraseExtractor() instead
        ('tfidf', TfidfVectorizer()),
        ('ann', MLPClassifier(hidden_layer_sizes=[500, 150], verbose=True, max_iter=10))
    ])

    # Split into training and test
    X_train, y_train, X_test, y_test = split_data(corpus, split_idx=0.1)

    # Train model
    model = fit_model(X_train, y_train, pipeline, saveto="./model")

    # Graph loss during training
    training_loss = pipeline.steps[2][1].loss_curve_
    plot(training_loss)

    # Test model on unseen data
    test_model(X_test, y_test, model)
    print()


def task5bc(corpus):
    """
    Train a LSTM model and reports it accuracy on a 90/10 data split
    :param corpus: TweetCorpusReader
    :return:
    """
    print("Exercise 5b, 5c: ")
    config = {
        "max_length": 26,  # MAX sequence length of all tweets
        "embedding_dim": 512,
        "hidden_dim": 256
    }

    # Split into training and test
    X_train, y_train, X_test, y_test = split_data(corpus, split_idx=0.1)

    # Preprocess all tweets
    X_train, vocab_table = preprocess(X_train, config)
    X_test, _ = preprocess(X_test, config, vocab_table)
    y_train, y_test = np.array(y_train), np.array(y_test)
    config["vocab_size"] = len(vocab_table.keys()) + 1

    # Initialize model
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Embedding(config["vocab_size"], config["embedding_dim"], input_length=config["max_length"]))
    model.add(tf.keras.layers.LSTM(config["hidden_dim"]))
    model.add(tf.keras.layers.Dense(3, activation="softmax"))

    # Compile and fit
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=1)

    # Evaluate prediction on test data
    loss, acc = model.evaluate(X_test, y_test, verbose=1)


if __name__ == '__main__':
    path = "./twitter-airline-sentiment/Tweets.csv"
    headers = ['airline', 'text', 'airline_sentiment']

    # Load data from csv file
    df = pd.read_csv(path)[headers]

    # Turn DataFrame into a corpus
    labels = ["positive", "negative", "neutral"]
    corpus = TweetCorpusReader(df, labels)

    # task_5a(corpus)
    task5bc(corpus)
