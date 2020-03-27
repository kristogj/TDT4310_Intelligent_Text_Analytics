import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk.tree import Tree

def run_parser(cp, tagged_sents):
    """

    :param cp: Parser
    :param tagged_sents: List of tagged sentences
    :return:
    """
    for i, sentence in enumerate(tagged_sents, 1):
        print("{}. Sentence : ".format(i))
        # Parse the sentence using the RegexpParser
        result = cp.parse(sentence)

        # 3a) Just print the result from parser
        print(result)

        # 3b) If it is a NP, print out the text inside it
        print("Matching texts in NP chunks: ")
        for subtree in result.subtrees():
            if subtree.label() == "NP":
                subtree = list(map(lambda x: x[0], subtree.leaves()))
                subtree = " ".join(subtree)
                print(subtree)
        print()


if __name__ == '__main__':
    # Initialize Parser
    grammar = r"""
    NP: {< NNP > ∗}
    {< DT >? < JJ >? < NNS >} 
    {< NN >< NN >}
    """
    cp = nltk.RegexpParser(grammar)

    # Initalize corpus reader
    corpus_reader = PlaintextCorpusReader(root="./SpaceX", fileids=".*\.txt")

    # Tag all sentences
    sents = corpus_reader.sents("SpaceX.txt")
    tagged_sents = nltk.pos_tag_sents(sents)

    # Run parser on first five sentences
    run_parser(cp, tagged_sents[:5])
