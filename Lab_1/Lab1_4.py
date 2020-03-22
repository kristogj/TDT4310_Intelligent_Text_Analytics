from nltk.corpus import brown


def get_words(k, n):
    # Slice the first n words from brown corpus
    word_list = brown.words()[:n]

    # Remove recurrent words
    word_set = set(word_list)
    result = set()
    # Count occurrences of word in word_list, and check if it is at least k
    for word in word_set:
        if word_list.count(word) >= k:
            result.add(word)
    return result


k = 4
n = 100

print(get_words(k, n))
