import tweepy
from nltk.corpus import TwitterCorpusReader, PlaintextCorpusReader, stopwords
import os
import json
from collections import Counter
from nltk.tokenize import TweetTokenizer
from string import punctuation
punctuation = punctuation + '’' + '…' + '️'

# Enter your keys/secrets as strings in the following fields
consumer_key = "mmDUOjKIlRFQDyfudDFKCWOaF"
consumer_secret = "SeHG9RZdjtJbkil7Xt5oJxwWG3gAv6g8KwigYcx5nHXyfaadHp"
access_token = "272648500-1yB7TIbESfaHCygZugjxu16dStI23XYvAWHqRUhu"
access_token_secret = "XSqjatjFTHIeddD75zwcUhGDfAuIsLz25sxTsO1GJKw3q"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

screen_names = ["elonmusk", "realDonaldTrump", "BarackObama", "BillGates", "ylecun",
                "goodfellow_ian", "LoganPaul", "TheNotoriousMMA", "Cristiano", "lexfridman"]

if not os.path.exists("./twitter-files"):
    os.mkdir("./twitter-files")
    for s_name in screen_names:
        latest = api.user_timeline(screen_name=s_name, count=100, tweet_mode="extended")
        tweet_list = []
        # Only keep some features from each tweet
        file = open("./twitter-files/{}.txt".format(s_name), 'w', encoding="utf8")
        for tweet in latest:
            file.write(tweet.full_text + "\t")
        file.close()

# Init CropusReader

stopw = stopwords.words("english")
reader = PlaintextCorpusReader(root="./twitter-files", fileids=".*\.txt", word_tokenizer=TweetTokenizer())
ids = reader.fileids()
counter = Counter()
for word in reader.words():
    word = word.lower()
    if word not in stopw and word not in punctuation:
        counter[word] += 1

top_10 = sorted(list(counter.items()), key=lambda x: x[1], reverse=True)[:10]
print(top_10)


