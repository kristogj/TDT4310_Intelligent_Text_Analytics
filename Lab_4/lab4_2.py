from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import TweetTokenizer
from nltk.util import ngrams
from nltk import KneserNeyProbDist, FreqDist, word_tokenize
from numpy import random


def predict_next_word(input_text, model):
    """
    Predicts the next word in a sentence
    :param input_text: a string
    :param model: LanguageModel
    :return:
    """
    tokenized = word_tokenize(input_text)
    completions = {}
    for sample in model.samples():
        if (sample[0], sample[1]) == (tokenized[-2], tokenized[-1]):
            completions[sample[2]] = model.prob(sample)
    if len(completions) == 0:
        response = "Can we talk about something else?"
    else:
        best = max(completions.keys(), key=(lambda key: completions[key]))
        tokenized += [best]
        response = " ".join(tokenized)
    return response


def complete_sentence(inputs, model):
    """
    Print the predicted sentence for all input texts
    :param inputs: list of input texts
    :param model: Language model
    :return:
    """
    for sent in inputs:
        while True:
            sent = predict_next_word(sent, model)
            if sent.split(" ")[-1] == "<END>":
                sent = " ".join(sent.split(" ")[:-1])
                break
        print(sent)


if __name__ == '__main__':
    print("Lab 4 Exercise 2")
    corpus_reader = PlaintextCorpusReader(root="./twitter-files", fileids=".*\.txt", word_tokenizer=TweetTokenizer())

    # Convert tweets to tri-grams
    tweets = [tweet for tweet in corpus_reader.sents()]
    tweet_trigrams = [list(ngrams(sequence=tweet,
                                  n=3,
                                  pad_left=True,
                                  pad_right=True,
                                  left_pad_symbol="<START>",
                                  right_pad_symbol="<END>")) for tweet in tweets]
    all_trigrams = [gram for tweet in tweet_trigrams for gram in tweet]

    # Initialize the language model
    freq_dist = FreqDist(all_trigrams)
    model = KneserNeyProbDist(freq_dist)

    # Predict sentences
    inputs = ["make America", "I am the", "China is", "The President of",
              "This election", "I love", "Fake News"]
    print("Inputs: {}".format(inputs))
    complete_sentence(inputs, model)
