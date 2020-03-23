from nltk import UnigramTagger, FreqDist, ConditionalFreqDist
from nltk.corpus import brown, nps_chat
from lab2_2 import CombinedTagger, get_data, test_tagger


def task3(data, corpus):
    fd = FreqDist(corpus.words())
    cfd = ConditionalFreqDist(corpus.tagged_words())
    most_freq_words = sorted(list(fd.items()), key=lambda x: x[1], reverse=True)[:200]
    most_freq_words = list(map(lambda x: x[0], most_freq_words))
    likely_tags = dict((word, cfd[word].max()) for word in most_freq_words)
    lookup_tagger = UnigramTagger(model=likely_tags)
    for str in ["brown50", "brown90", "nps50", "nps90"]:
        tagger = CombinedTagger(train=data["train_" + str], default=lookup_tagger, name=str)
        test_tagger(tagger, data)


if __name__ == '__main__':
    data = get_data()
    task3(data, brown)
    task3(data, nps_chat)
