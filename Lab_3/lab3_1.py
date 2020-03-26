from nltk.corpus import movie_reviews, wordnet, stopwords
from nltk import FreqDist, NaiveBayesClassifier
from nltk.classify import accuracy, apply_features
import random


def define_word_features():
    """
    Return the word_features
    :return: list of top 2000 most frequent words
    """
    stop_words = stopwords.words("english")
    # Find the 2000 most frequent words, and use those as the word_feature for a movie review.
    all_words = FreqDist(w.lower() for w in movie_reviews.words() if w not in stop_words)
    word_features = sorted(all_words.items(), key=lambda x: x[1], reverse=True)[:2000]
    word_features = list(map(lambda x: x[0], word_features))
    return word_features


WORD_FEATURES = define_word_features()


def get_documents():
    """
    Return a list of all movie reviews in from the nltk corpus with their assigned category
    :return: list of (document, category)
    """
    # Retrieve all movie reviews and their assigned category.
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]
    random.shuffle(documents)
    return documents


def document_features(document):
    """
    Convert a document into a word_feature vector
    :param document: List of words
    :return: a word feature vector represented as a dictionary (word, boolean word present in document)
    """
    document_words = set(document)
    features = {}
    for word in WORD_FEATURES:
        features[word] = word in document_words
        # Extend by also using WordNet
        for synset in wordnet.synsets(word):
            for lemma_name in synset.lemma_names():
                features[lemma_name] = lemma_name in document_words
    return features


if __name__ == '__main__':
    print("Lab 3 - Exercise 1")
    # Get all movie reviews and their category
    documents = get_documents()

    # Convert all documents to feature vectors, and split into training and test dataset with apply_features.
    # Apply features makes sure that it does not store all feature sets in memory, but it still acts as a list.
    train_set = apply_features(document_features, documents[100:])
    test_set = apply_features(document_features, documents[:100])

    # Train and test a classifier
    classifier = NaiveBayesClassifier.train(train_set)
    acc = accuracy(classifier, test_set)
    print("TestAccuracy: {}".format(acc))
    classifier.show_most_informative_features(5)
