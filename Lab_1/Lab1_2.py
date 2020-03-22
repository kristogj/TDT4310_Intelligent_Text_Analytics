from nltk.book import text9, sent9


def words_to_senteces(words):
    sentences = []
    start, end = 0, 0
    for i in range(len(words)):
        word = text9[i]
        if word == ".":
            end = i
            sentences.append(" ".join(words[start:end]))
            start = i + 1
    return sentences


sentences = words_to_senteces(text9)

# Find and print the first sentence that contains the word "sunset"
for sent in sentences:
    if "sunset" in sent:
        print(sent)
        break


# Write a program to get a word as input, then write all the sentences in text9 that contain the
# input word
def find_sentences(word):
    for sent in sentences:
        if word in sent:
            print(sent)


find_sentences("sunset")
