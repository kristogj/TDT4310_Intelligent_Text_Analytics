from nltk.corpus import brown
import nltk
import random


def most_frequent_tag(corpus):
    """
    Task a)
    Print out the most frequent tags of the corpus
    :param corpus: CorpusReader
    :return: None
    """
    word_tags = corpus.tagged_words()
    tags = [tup[1] for tup in word_tags]
    fd = nltk.FreqDist(tags)
    most_frequent = fd.max()
    print("Most frequent tag in Brown: {}".format(most_frequent))


def ambiguous_words(corpus):
    """
    Task b) c)
    Print out how many words that have at least two tags
    :param corpus: CorpusReader
    :return:None
    """
    word_tags = corpus.tagged_words()
    data = nltk.ConditionalFreqDist((word.lower(), tag) for (word, tag) in word_tags)
    ambig_words = {}
    for word in data.conditions():
        if len(data[word]) >= 2:
            ambig_words[word] = data[word]
    n_ambiguous = len(ambig_words.keys())
    total = len(word_tags)
    print("Number of ambiguous words in Brown: {}".format(n_ambiguous))
    print(
        "Percentage of ambiguous words in Brown: {}‰ ({}/{})".format(round(n_ambiguous / total, 3), n_ambiguous, total))


def distinct_tags(corpus):
    """
    Task d) e)
    Print top 10 words with the greatest number of distinct tags
    :param corpus: CorpusReader
    :return: None
    """
    word_tags = corpus.tagged_words()
    data = nltk.ConditionalFreqDist((word.lower(), tag) for (word, tag) in word_tags)
    items = sorted(list(data.items()), key=lambda t: len(t[1]), reverse=True)
    top_10 = items[:10]
    print("Top 10 words with greatest number of distinct tags:")
    for i, (word, tags) in enumerate(top_10, 1):
        print("{}. {} {}".format(i, word, list(tags.keys())))

    top_word = top_10[0][0]
    tags = list(top_10[0][1].keys())
    print_sentences(top_word, tags, corpus)


def print_sentences(word, tags, corpus):
    """
    Return a list of sentences with the word using each of the tags
    :param word: string
    :param tags: List of strings (tags)
    :return:
    """
    tags = set(tag for tag in tags)
    sentences = []
    # Pop one tag out, and search for a sentence with that tag
    while tags:
        rt = random.sample(tags, 1)[0]
        for sent in corpus.tagged_sents():
            for w, t in sent:
                if w.lower() == word and t == rt:
                    sentences.append(((word, t), " ".join(w for w, t in sent)))
                    tags.remove(rt)
                    # print("{}...".format(len(word_tag)))
                    if tags:
                        rt = random.sample(tags, 1)[0]
                    else:
                        break
    print("\nOne sentence for each tag of the most frequent word:")
    for tup, sent in sentences:
        print("{} : {}".format(tup, sent))


if __name__ == '__main__':
    most_frequent_tag(brown)
    ambiguous_words(brown)
    distinct_tags(brown)
