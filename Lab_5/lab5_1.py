import nltk
from nltk.tree import Tree


def get_tagged_sents(path):
    file = open(path, "r")
    text = file.read()
    file.close()
    sents = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text)]
    tagged_sents = nltk.pos_tag_sents(sents)
    return tagged_sents


def get_named_entities(tagged_sents):
    named_entities = []
    for sent in tagged_sents:
        tree: Tree = nltk.ne_chunk(sent, binary=True)
        for subtree in tree.subtrees():
            if subtree.label() == "NE":
                ne = list(map(lambda x: x[0], subtree.leaves()))
                ne = " ".join(ne)
                named_entities.append(ne)
    return named_entities


def pprint(named_entities):
    print("Exercise 1a - all entity names in the file:")
    for name in named_entities:
        print(name, end=", ")

    print("\n\nExercise 1b - unique entity names:")
    for name in set(named_entities):
        print(name, end=", ")

    print("\n\nExercise 1c - unique names i lexical ascending order:")
    for name in sorted(list(set(named_entities))):
        print(name, end=", ")


if __name__ == '__main__':
    path = "./SpaceX.txt"
    tagged_sents = get_tagged_sents(path)
    named_entities = get_named_entities(tagged_sents)
    pprint(named_entities)
