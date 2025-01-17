#!/usr/bin/env python3

import nltk
from collections import Counter

MALE = 'male'
FEMALE = 'female'
UNKNOWN = 'unknown'
BOTH = 'both'

MALE_WORDS = set([
    'guy', 'spokesman', 'chairman', "men's", 'men', 'him', "he's", 'his',
    'boy', 'boyfriend', 'boyfriends', 'boys', 'brother', 'brothers', 'dad',
    'dads', 'dude', 'father', 'fathers', 'fiance', 'gentleman', 'gentlemen',
    'god', 'grandfather', 'grandpa', 'grandson', 'groom', 'he', 'himself',
    'husband', 'husbands', 'king', 'male', 'man', 'mr', 'nephew', 'nephews',
    'priest', 'prince', 'son', 'sons', 'uncle', 'uncles', 'waiter', 'widower',
    'widowers'
])

FEMALE_WORDS = set([
    'heroine', 'spokeswoman', 'chairwoman', "women's", 'actress', 'women',
    "she's", 'her', 'aunt', 'aunts', 'bride', 'daughter', 'daughters', 'female',
    'fiancee', 'girl', 'girlfriend', 'girlfriends', 'girls', 'goddess',
    'granddaughter', 'grandma', 'grandmother', 'herself', 'ladies', 'lady',
    'lady', 'mom', 'moms', 'mother', 'mothers', 'mrs', 'ms', 'niece', 'nieces',
    'priestess', 'princess', 'queens', 'she', 'sister', 'sisters', 'waitress',
    'widow', 'widows', 'wife', 'wives', 'woman'
])


def genderize(words):
    """
    Categorize a list of words to be eihter Male, Female, Both or unknown based on the match between some
    predefined words for each gender and the words in the sentence.
    :param words: List of words
    :return: String category
    """
    mwlen = len(MALE_WORDS.intersection(words))
    fwlen = len(FEMALE_WORDS.intersection(words))

    if mwlen > 0 and fwlen == 0:
        return MALE
    elif mwlen == 0 and fwlen > 0:
        return FEMALE
    elif mwlen > 0 and fwlen > 0:
        return BOTH
    else:
        return UNKNOWN


def count_gender(sentences):
    sents = Counter()
    words = Counter()

    # For each sentence, categorize its gender and save results
    for sentence in sentences:
        gender = genderize(sentence)
        sents[gender] += 1
        words[gender] += len(sentence)

    return sents, words


def parse_gender(text):
    """
    nltk.sent_tokenize(text) takes a text and returns a list of sentences from text
    nltk.word_tokenize(sentence) takes a text/sentence and returns tokenize it into a list of words
    """
    # Convert text to list of sentences where each sentence is a list of words
    sentences = [
        [word.lower() for word in nltk.word_tokenize(sentence)]
        for sentence in nltk.sent_tokenize(text)
    ]

    # Categorize words and sentences into either Male, Female, Both or unknown words
    sents, words = count_gender(sentences)

    # Total number of words
    total = sum(words.values())

    # Report percentage of word categorized in each category, as well as number of sentences in that category.
    for gender, count in words.items():
        pcent = (count / total) * 100
        nsents = sents[gender]
        print(
            "{:0.3f}% {} ({} sentences)".format(pcent, gender, nsents)
        )


if __name__ == '__main__':
    with open('sample.txt', 'r') as f:
        parse_gender(f.read())
