from nltk.corpus import brown
import nltk
from nltk.tree import Tree


def get_top20(cp):
    results = []
    for sentence in brown.tagged_sents():
        # Parse the sentence using the RegexpParser
        result = cp.parse(sentence)

        # Look for (verb,prep, NP) sequences in each sentence
        for x in range(len(result) - 2):
            a, b, c = result[x], result[x + 1], result[x + 2]
            if isinstance(a, tuple) and isinstance(b, tuple) and isinstance(c, Tree):
                if a[1] == "VBD" and b[1] == "IN":
                    results.append((a[0], b[0], "NP"))

    # Sort results and slice top 20
    return sorted(results, key=lambda x:x[0])[:20]


if __name__ == '__main__':
    print("Lab 4 Exercise 1")
    grammar = r""" NP:
                {<DT>?<JJ>*<NN>} 
                }<VBD|IN>+{ # Chink sequences of VBD and IN """
    cp = nltk.RegexpParser(grammar)
    results = get_top20(cp)
    print("Top 20 in lexicographical order on verb:")
    for tup in results:
        print("{}".format(tup))