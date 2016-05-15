from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import random
import csv
import re

data_dir       = "../data"

training_data_file = "{0}/training2.csv".format(data_dir)
test_data_file     = "{0}/test2.csv".format(data_dir)

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
                         'retweet_count': row['retweet_count'],
                         'tweet_created': row['tweet_created'],
                         'tweet_location': row['tweet_location'],
                         'text': text_sample,
                         'airline': row['airline'],
                         'airline_sentiment': int(row['airline_sentiment'])})

        text_samples.append(" ".join(text_sample))

    return new_data, text_samples

def load_test(filename = test_data_file):
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

def perceptron_classify(data, samples, weights, rate = 0.8):
    errors = 0

    for sample, value in zip(samples, data):
        classification = unit_step(weights.T.dot(sample))
        error          = value['airline_sentiment'] - classification
        weights       += sample * error * rate

        if classification != value['airline_sentiment']:
            errors += 1

    return weights, errors

def gradient_classify(data, samples, weights, alpha = 0.0001):
    labels = np.array([value['airline_sentiment'] for value in data])
    N      = len(labels)
    output = weights.dot(samples.T).flatten()

    errors = samples.T.dot(output - labels)
    weights -= alpha * (1./N) * errors

    return weights, len(samples)

def write_results(weights, test_samples, test_data):
    with open("results.csv", "w+") as output_file:
        output_file.write("airline_sentiment,id\n")

        for sample, value in zip(test_samples, test_data):
            sample_id      = value['id']
            classification = unit_step(weights.T.dot(sample))
            if value['airline_sentiment'] != classification:
                output_file.write("{0},{1},{2}\n".format(classification, value['airline_sentiment'], sample_id))

def write_gradient_results(weights, test_samples, test_data):
    with open("gradient_results.csv", "w+") as output_file:
        output_file.write("airline_sentiment,id\n")

        for sample, value in zip(test_samples, test_data):
            sample_id      = value['id']
            output         = np.dot(sample, weights)
            classification = unit_step(output)
            if value['airline_sentiment'] != classification:
                output_file.write("{0},{1},{2}\n".format(classification, value['airline_sentiment'], sample_id))

def learn(threshold  = 6,
          stop_after = 500,
          classifier = perceptron_classify,
          write_out  = write_results):
    print("Starting PLA with:\n    Threshold: {0}".format(threshold))
    print("Setting up...")

    training_data, training_samples        = load_data()
    training_features, training_vocabulary = vectorize(training_samples)

    weights                                = get_initial_weights(len(training_vocabulary))
    iterations                             = 0

    print("Setup complete.\nEntering PLA loop...")
    while True:
        iterations                     += 1
        weights, misclassified_samples  = classifier(training_data, training_features, weights)

        if misclassified_samples <= threshold or iterations >= stop_after:
            print("    Iteration: {0}".format(iterations))
            print("    Misclassifications: {0}".format(misclassified_samples))
            break

        if iterations % 500 == 0:
            print("    Iteration: {0}".format(iterations))
            print("    Misclassifications: {0}".format(misclassified_samples))

    print("Training Complete.\nSetting up Test Data...")
    test_data, test_samples = load_data(filename = test_data_file)
    test_features, test_vocabulary = vectorize(test_samples, vocabulary = training_vocabulary)
    print("Test Setup Complete.\nTesting...")
    write_out(weights, test_features, test_data)
    print("Testing Complete.\nWrote to \"results.csv\".")

if __name__ == '__main__':
    learn()
