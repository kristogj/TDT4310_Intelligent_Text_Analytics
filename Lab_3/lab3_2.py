from nltk.corpus import names
from nltk.classify import apply_features, accuracy
from nltk import NaiveBayesClassifier, DecisionTreeClassifier, MaxentClassifier
import random


def get_data():
    data = [(name, 'male') for name in names.words('male.txt')] + \
           [(name, 'female') for name in names.words('female.txt')]
    random.shuffle(data)
    return data


def gender_features(name):
    """
    A gender feature that only looks at the last letter of a name.
    :param name:
    :return:
    """
    return {'last_letter': name[-1]}


def gender_features2(name):
    """
    A feature extractor that overfits gender features.
    The featuresets returned by this feature extractor contain a large number of specific features,
    leading to overfitting for the relatively small Names Corpus.
    :param name:
    :return:
    """
    features = {}
    features["firstletter"] = name[0].lower()
    features["lastletter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count(%s)" % letter] = name.lower().count(letter)
        features["has(%s)" % letter] = (letter in name.lower())
    return features


def gender_features3(word):
    return {'suffix1': word[-1:],
            'suffix2': word[-2:]}


if __name__ == '__main__':
    print("Lab 3 - Exercise 2")
    data = get_data()
    train_set = apply_features(gender_features3, data[500:])
    test_set = apply_features(gender_features3, data[:500])

    print("Training classifiers")
    # Train the different classifiers on the training set
    classifier = [(NaiveBayesClassifier.train(train_set), "NaiveBayes"),
                  (DecisionTreeClassifier.train(train_set), "DecisionTree"),
                  (MaxentClassifier.train(train_set, max_iter=10, trace=0), "MaxEntropy")]

    # Test all classifiers on the test set
    for classifier, name in classifier:
        acc = accuracy(classifier, test_set)
        print("{} classifier test accuracy: {}".format(name, acc))
