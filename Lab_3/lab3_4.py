import tweepy
from nltk.corpus import PlaintextCorpusReader, stopwords
import os
from nltk.tokenize import TweetTokenizer
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import random
import numpy as np

import sys
import numpy

numpy.set_printoptions(threshold=sys.maxsize)


def init_twitter():
    # Enter your keys/secrets as strings in the following fields
    consumer_key = "mmDUOjKIlRFQDyfudDFKCWOaF"
    consumer_secret = "SeHG9RZdjtJbkil7Xt5oJxwWG3gAv6g8KwigYcx5nHXyfaadHp"
    access_token = "272648500-1yB7TIbESfaHCygZugjxu16dStI23XYvAWHqRUhu"
    access_token_secret = "XSqjatjFTHIeddD75zwcUhGDfAuIsLz25sxTsO1GJKw3q"

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)
    return api


API = init_twitter()


def get_corpus_reader(screen_names):
    if not os.path.exists("./twitter-files"):
        os.mkdir("./twitter-files")
        for s_name in screen_names:
            latest = API.user_timeline(screen_name=s_name, count=1000, tweet_mode="extended", include_rts=False,
                                       exclude_replies=True)
            # Only keep some features from each tweet
            file = open("./twitter-files/{}.txt".format(s_name), 'w', encoding="utf8")
            for tweet in latest:
                file.write(tweet.full_text + "\t")
            file.close()

    return PlaintextCorpusReader(root="./twitter-files", fileids=".*\.txt", word_tokenizer=TweetTokenizer())


def fit_vectorizer(corpus_reader):
    file_ids = corpus_reader.fileids()
    full_corpus = corpus_reader.sents(file_ids)
    full_corpus = [" ".join(sent) for sent in full_corpus]

    stop_words = stopwords.words("english")
    vectorizer = CountVectorizer(stop_words=stop_words)
    vectorizer.fit(full_corpus)
    return vectorizer


def shuffle_data(X, y):
    Xy = list(zip(X, y))
    random.shuffle(Xy)
    X, y = zip(*Xy)
    return np.array(X), np.array(y)


def get_data(corpus_reader, vectorizer):
    """
    Return a training-test split
    :return:
    """
    # First process training data
    X, y = [], []
    ids = corpus_reader.fileids()
    for id in ids:
        print("id: {}, #tweets: {}".format(id, len(corpus_reader.sents(id))))
        for sent in corpus_reader.sents(id):
            sent = " ".join(sent)
            X.append(sent)
            y.append(ids.index(id))

    # Shuffle data
    X, y = shuffle_data(X, y)

    test_X, test_y = X[:100], y[:100]
    train_X, train_y = X[100:], y[100:]

    return train_X, train_y, test_X, test_y


def fit_model(X, y):
    X = vectorizer.transform(X)
    X = [x.toarray()[0] for x in X]
    return MultinomialNB().fit(X, y)


def test_model(X, y, model):
    # Transform tweets to vectors
    X = vectorizer.transform(X)
    X = [x.toarray()[0] for x in X]

    predictions = model.predict(X)
    acc = sum(predictions == y) / len(y)
    print("Test accuracy: {}".format(acc))


def show_some_probs(X, y, model, screen_names, n=5):
    X_vec = vectorizer.transform(X)
    X_vec = [x.toarray()[0] for x in X_vec]

    X_vec, y = X_vec[:n], y[:n]
    predications = model.predict_proba(X_vec)

    for i in range(n):
        max_prob_index = np.argmax(predications[i])
        print("\n")
        print("Tweet: {}".format(X[i]))
        print("True Label: {}".format(screen_names[y[i]]))
        print("Prediction {}({}%)".format(screen_names[max_prob_index], max(predications[i])))


if __name__ == '__main__':
    print("Exercise 4")
    screen_names = ["BillGates", "LoganPaul"]
    for i, name in enumerate(screen_names):
        print("{}: {}".format(i, name), end=" ")
    print("\n")

    corpus_reader = get_corpus_reader(screen_names)
    vectorizer = fit_vectorizer(corpus_reader)

    # Split data into traing and testing
    train_X, train_y, test_X, test_y = get_data(corpus_reader, vectorizer)

    # Initialize the classifier
    # Transform tweets to vectors
    model = fit_model(train_X, train_y)

    # Report test results
    test_model(test_X, test_y, model)

    # Show some examples
    show_some_probs(test_X, test_y, model, screen_names)
