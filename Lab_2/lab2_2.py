from nltk.corpus import brown, nps_chat
from nltk import DefaultTagger, UnigramTagger, BigramTagger, TrigramTagger, RegexpTagger, FreqDist, ContextTagger


def train_test_split(lst, fraction):
    """
    Split the sentences into two list fraction/1-fraction
    :param lst: List of items
    :param fraction: training set size
    :return: train and test split
    """
    size = int(len(lst) * fraction)
    return lst[:size], lst[size:]


def get_data():
    """
    Split the Brown and Nps Chat corpus into 4 training sets and 4 test sets
    :return:
    """
    data = {"train_brown50": train_test_split(brown.tagged_sents(), 0.5)[0],
            "test_brown50": train_test_split(brown.tagged_sents(), 0.5)[1],
            "train_brown90": train_test_split(brown.tagged_sents(), 0.9)[0],
            "test_brown10": train_test_split(brown.tagged_sents(), 0.9)[1],
            "train_nps50": train_test_split(nps_chat.tagged_posts(), 0.9)[0],
            "test_nps50": train_test_split(nps_chat.tagged_posts(), 0.9)[1],
            "train_nps90": train_test_split(nps_chat.tagged_posts(), 0.9)[0],
            "test_nps10": train_test_split(nps_chat.tagged_posts(), 0.9)[1]}
    return data


def test_tagger(tagger, data):
    """
    Test tagger on all
    :param tagger:
    :param data:
    :return:
    """
    print(tagger)
    for test_set in ["brown50", "brown10", "nps50", "nps10"]:
        acc = tagger.evaluate(data["test_" + test_set])
        print("\tAccuracy on test set {}: {}".format(test_set, round(acc, 3)))


class CombinedTagger:

    def __init__(self, train=None, default=None, name=None):
        self.name = name
        # As found on page 199 of the nltk book
        regexps = [
            (r'.*ing$', 'VBG'),  # gerunds
            (r'.*ed$', 'VBD'),  # simple past
            (r'.*es$', 'VBZ'),  # 3rd singular present
            (r'.*ould$', 'MD'),  # modals
            (r'.*\'s$', 'NN$'),  # possessive nouns
            (r'.*s$', 'NNS'),  # plural nouns
            (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
            ]
        self.default = default
        self.regex = RegexpTagger(regexps, backoff=self.default)
        self.unigram = UnigramTagger(train=train, backoff=self.regex)
        self.bigram = BigramTagger(train=train, backoff=self.unigram)

    def evaluate(self, data):
        return self.bigram.evaluate(data)

    def __repr__(self):
        return '<Combined Tagger: train={}>'.format(self.name)


def task2a(data):
    tags = []
    for key in data.keys():
        for sentence in data[key]:
            for _, tag in sentence:
                tags.append(tag)
    fd = FreqDist(tags)
    most_frequent_tag = fd.max()
    print("Most frequent tag: {}".format(most_frequent_tag))
    default_tagger = DefaultTagger(most_frequent_tag)
    test_tagger(default_tagger, data)
    return tag


def task2b(data, tag):
    default_tagger = DefaultTagger(tag)
    for str in ["brown50", "brown90", "nps50", "nps90"]:
        tagger = CombinedTagger(train=data["train_" + str], default=default_tagger, name=str)
        test_tagger(tagger, data)


if __name__ == '__main__':
    data = get_data()
    most_frequent_tag = task2a(data)
    task2b(data, most_frequent_tag)
