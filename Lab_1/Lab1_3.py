import tweepy
from nltk.corpus import TwitterCorpusReader
import os
import json
from collections import Counter

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
        for tweet in latest:
            tweet_information = dict()
            tweet_information['full_text'] = tweet.full_text
            tweet_information['created_at'] = tweet.created_at.strftime("%Y-%m-%d %H:%M:%S")
            tweet_information['screen_name'] = s_name
            tweet_list.append(tweet_information)

        # Dump all tweets to a file
        file_des = open("./twitter-files/{}.json".format(s_name), 'w', encoding="utf8")

        # dump tweets to the file
        json.dump(tweet_list, file_des, indent=4, sort_keys=True)
        file_des.close()



# Init CropusReader
reader = TwitterCorpusReader(root="./twitter-files", fileids=".*\.json")
ids = reader.fileids()
for id in ids:
    words = reader.strings(id)


