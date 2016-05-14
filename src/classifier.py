from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import random
import csv
import re

data_dir       = "../data"

training_data  = "{0}/training.csv".format(data_dir)
test_data      = "{0}/test.csv".format(data_dir)

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
                         'airline_sentiment': int(row['airline_sentiment'])})

        text_samples.append(" ".join(text_sample))

    return new_data, text_samples

def load_test(filename = test_data):
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
                         'airline': row['airline']})

        text_samples.append(" ".join(text_sample))

    return new_data, text_samples

def vectorize(data, vocabulary = None):
    vectorizer = CountVectorizer(analyzer     = "word",
                                 tokenizer    = None,
                                 preprocessor = None,
                                 stop_words   = None,
                                 vocabulary   = vocabulary)

    features          = vectorizer.fit_transform(data).toarray()
    vocabulary        = vectorizer.get_feature_names()

    weighted_features = np.array([np.insert(feature, 0, 1) for feature in features])
    return weighted_features, vocabulary

def get_initial_weights(size):
    return np.zeros(size + 1)

def unit_step(x):
    if x < 0:
        return -1
    else:
        return 1

def update_weights(weights, sample, error, rate = 0.1):
    return weights + (sample * error * rate)

def classify(data, samples, weights):
    errors = 0

    for sample, value in zip(samples, data):
        classification = unit_step(weights.T.dot(sample))
        error          = value['airline_sentiment'] - classification
        weights        = update_weights(weights, sample, error)

        if classification != value['airline_sentiment']:
            errors += 1

    return weights, errors

def write_results(weights, test_samples, test_data):
    with open("results.csv", "w+") as output_file:
        output_file.write("airline_sentiment,id\n")

        for sample, value in zip(test_samples, test_data):
            sample_id      = value['id']
            classification = unit_step(weights.T.dot(sample))
            output_file.write("{0},{1}\n".format(classification, sample_id))

def learn(threshold = 6):
    print("Starting PLA with:\n    Threshold: {0}".format(threshold))
    print("Setting up...")

    training_data, training_samples        = load_data()
    training_features, training_vocabulary = vectorize(training_samples)

    weights                                = get_initial_weights(len(training_vocabulary))
    iterations                             = 0

    print("Setup complete.\nEntering PLA loop...")
    while True:
        iterations                     += 1
        weights, misclassified_samples  = classify(training_data, training_features, weights)

        if misclassified_samples <= threshold:
            break

        if iterations % 100 == 0:
            print("    Iteration: {0}".format(iterations))
            print("    Misclassifications: {0}".format(misclassified_samples))

    print("Training Complete.\nSetting up Test Data...")
    test_data, test_samples = load_test()
    test_features, test_vocabulary = vectorize(test_samples, vocabulary = training_vocabulary)
    print("Test Setup Complete.\nTesting...")
    write_results(weights, test_features, test_data)
    print("Testing Complete.\nWrote to \"results.csv\".")

if __name__ == '__main__':
    learn()
