import pandas as pd
from nltk.corpus import twitter_samples
import numpy as np

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

def train_test_split(pos_tweets, neg_tweets):
    pos_data = list()
    for i in range(len(pos_tweets)):
        temp = [pos_tweets[i], 1]
        pos_data.append(temp)

    neg_data = list()
    for j in range(len(neg_tweets)):
        temp = [neg_tweets[j], 0]
        neg_data.append(temp)

    train = pos_data[0:4000]
    train = train + neg_data[0:4000]

    test = pos_data[4000:]
    test = test + neg_data[4000:]

    return train, test

train, test = train_test_split(all_positive_tweets, all_negative_tweets)

m = len(train)

vocabulary = {word for tweet in train for word in tweet[0].split()}

pos_freq = dict()
neg_freq = dict()

for tweet, sentiment in train:
    for word in tweet.split():
        if sentiment == 1:
            pos_freq[word] = pos_freq.get(word, 0) + 1
        elif sentiment == 0:
            neg_freq[word] = neg_freq.get(word, 0) + 1

ex = train[2]

def feature_extraction_pos(tweet: str) -> int:
    total_sum = 0
    for word in set(tweet.split()):
        if word in pos_freq:
            total_sum += pos_freq[word]
    return total_sum

def feature_extraction_neg(tweet: str) -> int:
    total_sum = 0
    for word in set(tweet.split()):
        if word in neg_freq:
            total_sum += neg_freq[word]
    return total_sum

tweet_value = np.array([1, feature_extraction_pos(ex[0]), feature_extraction_neg(ex[0])])

def calc_dot_product(x, theta):
    return np.dot(theta.T, x)

def sigmoid(z):
    den = 1 + np.exp(-z)
    return 1 / den

z = np.array([2, 3, 4])

print(sigmoid(z))

x = np.zeros((len(train), 3), dtype=np.float128)
for i in range(len(train)):
    x[i] = [1, feature_extraction_pos(train[i][0]), feature_extraction_neg(train[i][0])]


print(x[0:5])

y = np.array([tweet[1] for tweet in train])
print(y[0:5])

theta = np.array([0.5, 0.5, 0.5])
J = 1000
learning_rate = 0.001

increase_count = 0
J_temp = None
for i in range(600):
    J_temp = J
    z = np.dot(x, theta)

    h = sigmoid(z)

    b = [np.log(a) if a > 0.0 else 0 for a in h]
    c = [np.log(1-a) if (1-a) > 0.0 else 0 for a in h]

    J = (-np.dot(y.T, b)) - np.dot((1 - y).T, c)
    J = J / m

    if J > J_temp:
        increase_count += 1
    else:
        increase_count = 0

    if increase_count >= 5:
        print("moving upward")
        break

    # if i % 5 == 0:
    print(f"Cost at iteration {i} is ", J)
    theta = theta - learning_rate / m * (np.dot(x.T, (h - y)))

predictions = list()

for tweet in test:
    # features
    x = np.array([1, feature_extraction_pos(tweet[0]), feature_extraction_neg(tweet[0])], dtype=np.float128)
    # dot
    y_pred = np.dot(x, theta)
    y_pred = sigmoid(y_pred)

    if y_pred > 0.5:
        predictions.append(1.0)
    else:
        predictions.append(0.0)

accuracy = 0
for i in range(len(predictions)):
    if predictions[i] == test[i][1]:
        accuracy += 1
accuracy = accuracy / len(test)



