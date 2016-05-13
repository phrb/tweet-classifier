from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import csv
import re

data_dir      = "../data"

training_data = "{0}/training.csv".format(data_dir)
test_data     = "{0}/test.csv".format(data_dir)

def clean_line(line):
    with_stop    = re.sub("[^a-zA-z@]", " ", line).lower().split()
    return [word for word in with_stop if not word in stopwords.words("english")]

def load_data(filename = training_data):
    csvfile      = open(filename)
    reader       = csv.DictReader(csvfile)
    new_data     = []
    text_samples = []

    for row in reader:
        text_sample = clean_line(row['text'])

        new_data.append({'id': row['id'],
                         'tweet_id': row['tweet_id'],
                         'retweet_count': row['retweet_count'],
                         'tweet_created': row['tweet_created'],
                         'tweet_location': row['tweet_location'],
                         'text': text_sample,
                         'airline': row['airline'],
                         'airline_sentiment': row['airline_sentiment']})

        text_samples.append(" ".join(text_sample))

    return new_data, text_samples

def vectorize(data):
    vectorizer = CountVectorizer(analyzer     = "word",
                                 tokenizer    = None,
                                 preprocessor = None,
                                 stop_words   = None,
                                 max_features = 5000)

    features   = vectorizer.fit_transform(data).toarray()
    vocabulary = vectorizer.get_feature_names()

    return features, vocabulary

def print_text(data):
    for row in data:
        print(row)

data, samples = load_data()
features, vocabulary = vectorize(samples)

distribution = np.sum(features, axis = 0)

for tag, count in zip(vocabulary, distribution):
    print(count, tag)
