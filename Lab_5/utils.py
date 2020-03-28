from nltk import word_tokenize, sent_tokenize, pos_tag_sents
import random
import numpy as np


def get_tagged_tweets(df):
    tweets = []
    for tweet in df["text"].tolist():
        sents = [word_tokenize(sent) for sent in sent_tokenize(tweet)]
        tagged_sents = pos_tag_sents(sents)
        tweets.append(tagged_sents)
    return tweets


def shuffle_data(X, y):
    Xy = list(zip(X, y))
    random.shuffle(Xy)
    X, y = zip(*Xy)
    return list(X), list(y)


def plot(y):
    x = list(range(1, len(y) + 1))
    plt.plot(x, y)
    plt.title("Loss over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("./loss_graph.png")
    plt.show()
