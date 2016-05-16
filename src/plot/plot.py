#! /usr/bin/python

from nltk.corpus import stopwords

import csv
import re
import os
import matplotlib as mpl

mpl.use('agg')

import matplotlib.pyplot as plt
import numpy as np

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

font = {'family' : 'serif',
        'size'   : 20}

mpl.rc('font', **font)

data_dir       = "../../data"

training_data_file = "{0}/training.csv".format(data_dir)
test_data_file     = "{0}/test.csv".format(data_dir)

def clean_line(line):
    with_stop    = re.sub("[^a-zA-z@]", " ", line).lower().split()
    return [word for word in with_stop if not word in stopwords.words("english")]

def load_data(filename = training_data_file):
    csvfile      = open(filename)
    reader       = csv.DictReader(csvfile)
    new_data     = []
    text_samples = []

    for row in reader:
        text_sample = clean_line(row['text'])

        new_data.append({'id': row['id'],
                         'tweet_id': row['tweet_id'],
                         'retweet_count': int(row['retweet_count']),
                         'tweet_created': row['tweet_created'],
                         'tweet_location': row['tweet_location'],
                         'text': text_sample,
                         'airline': row['airline'],
                         'airline_sentiment': int(row['airline_sentiment'])})

    return new_data

test_data  = load_data()

airlines   = []

sentiment  = {}
retweet    = {}
good_words = {"good"     : 0,
              "great"    : 0,
              "excellent": 0,
              "amazing"  : 0,
              "awesome"  : 0,
              "happy"    : 0,
              "funny"    : 0}

bad_words = {"bad"      : 0,
             "horrible" : 0,
             "terrible" : 0,
             "sad"      : 0,
             "delay"    : 0,
             "late"     : 0,
             "mad"      : 0,
             "time"     : 0,
             "seat"     : 0,
             "angry"    : 0}


sentiments = []
retweets   = []

for example in test_data:
    airline = example['airline']
    if airline not in airlines:
        airlines.append(airline)

    if airline in sentiment.keys():
        sentiment[airline] += example['airline_sentiment']
    else:
        sentiment[airline]  = example['airline_sentiment']

    if airline in retweet.keys():
        retweet[airline]   += example['retweet_count']
    else:
        retweet[airline]    = example['retweet_count']

    for w in good_words.keys():
        if w in example['text']:
            good_words[w] += example['airline_sentiment']

    for w in bad_words.keys():
        if w in example['text']:
            bad_words[w] += example['airline_sentiment']

for airline in airlines:
    sentiments.append(sentiment[airline])
    retweets.append(retweet[airline])

fig     = plt.figure(1, figsize=(9, 6))
ax      = fig.add_subplot(111)

indexes = np.arange(len(airlines))
width   = 0.5

ax.bar(indexes, sentiments, width, color='black')

ax.set_title("Airline Sentiment")
ax.set_xlabel("")
ax.set_xticks(indexes + (width / 2))
ax.set_ylabel("Positive + Negative Sentiment")
ax.set_xticklabels(tuple(airlines), rotation = 30)

plt.tight_layout()

fig.savefig('airline_sentiment.eps', format = 'eps', dpi = 1000)

plt.clf()

fig     = plt.figure(1, figsize=(9, 6))
ax      = fig.add_subplot(111)

ax.bar(indexes, retweets, width, color='black')

ax.set_title("Airline Retweet Count")
ax.set_xlabel("")
ax.set_xticks(indexes + (width / 2))
ax.set_ylabel("Retweet Count")
ax.set_xticklabels(tuple(airlines), rotation = 30)

plt.tight_layout()

fig.savefig('airline_retweet.eps', format = 'eps', dpi = 1000)

plt.clf()

fig     = plt.figure(1, figsize=(9, 6))
ax      = fig.add_subplot(111)

indexes = np.arange(len(good_words.values()))
ax.bar(indexes, good_words.values(), width, color='black')

plt.axhline(y = 0, color = 'black')

ax.set_title("Sentiment associated with \"Good\" Words")
ax.set_xlabel("")
ax.set_xticks(indexes + (width / 2))
ax.set_ylabel("Positive + Negative Sentiment")
ax.set_xticklabels(good_words.keys(), rotation = 30)

plt.tight_layout()

fig.savefig('airline_good_words.eps', format = 'eps', dpi = 1000)

plt.clf()

fig     = plt.figure(1, figsize=(9, 6))
ax      = fig.add_subplot(111)

indexes = np.arange(len(bad_words.values()))
ax.bar(indexes, bad_words.values(), width, color='black')

plt.axhline(y = 0, color = 'black')

ax.set_title("Sentiment associated with \"Bad\" Words")
ax.set_xlabel("")
ax.set_xticks(indexes + (width / 2))
ax.set_ylabel("Positive + Negative Sentiment")
ax.set_xticklabels(bad_words.keys(), rotation = 30)

plt.tight_layout()

fig.savefig('airline_bad_words.eps', format = 'eps', dpi = 1000)
