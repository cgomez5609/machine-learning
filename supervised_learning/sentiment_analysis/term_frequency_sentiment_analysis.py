#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 14:07:24 2021

@author: chris
"""
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import pandas as pd
from nltk.corpus import twitter_samples
import numpy as np

NLP = spacy.load("en_core_web_sm")

# get the positive and negative tweets from NLTK
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# Custom class to preprocess data
# NOTE: I made this package. I left this in to highlight the preprocessing steps, however the implementation is not in this file
from processtxt import TextProcessing
tp = TextProcessing(nlp=NLP, stopwords=STOP_WORDS)

def process_tweets(tweets):
    processed_tweets = list()
    for i, tweet in enumerate(tweets):
        temp_text = tp.lowercase(text=tweet)
        temp_text = tp.expand_contractions(text=temp_text)
        temp_text = tp.remove_urls(text=temp_text)
        temp_text = tp.remove_punctuation(text=temp_text)
        temp_text = tp.remove_stopwords(text=temp_text)
        temp_text = tp.remove_extra_whitespace(text=temp_text)
        temp_text = tp.lemmatize(text=temp_text)
        processed_tweets.append(temp_text)
    return processed_tweets

class IncorrectType(Exception):
    pass

class IncorrectShape(Exception):
    pass

# class to get the probabilistic sentiment of a tweet
class TermFrequencyPrediction:
    def __init__(self, data):
        self.data = np.array(data)
        self.check_data()
        self.classes = set()
        self.word_freq = dict()
        self.vocabulary = set()
        self.V = 0
        self.word_sums = dict()
        self.word_probabilities = dict()
    
    def check_data(self):
        if len(self.data.shape) != 2:
            raise IncorrectShape(f"Not correct shape at {self.data.shape}. Should should be (m,2) where m is the total number of rows")
        if len(self.data[0]) != 2:
            raise Exception(f"data should contain text and label only. This contains {len(self.data[0])} elements")
        if type(self.data[0][0]) != str and type(self.data[0][0]) != np.str_:
            raise IncorrectType("First elements in matrix should be a string")
        if type(float(self.data[0][1])) != float and type(int(self.data[0][1])) != int:
            raise IncorrectType("second element should be an integer or float")
        
    def __frequency(self):
        self.classes = set(self.data[:,1])
        self.word_sums = {str(label): 0 for label in self.classes }
        for i, datum in enumerate(self.data):
            label = str(datum[1])
            for word in datum[0].split():
                self.word_sums[label] += 1
                if word in self.word_freq:
                    self.word_freq[word][label] += 1
                else:
                    self.word_freq[word] = {str(label): 0 for label in self.classes}
                    self.word_freq[word][label] += 1
                    self.vocabulary.add(word)
        self.V = len(self.vocabulary)
        
    def __probability(self, word: str, class_label: float, apply_laplacian_smoothing=True):
        if self.V > 0:
            if apply_laplacian_smoothing:
                freq = self.word_freq[word][str(class_label)] + 1
                return round(freq / (self.word_sums[str(class_label)] + self.V), 6)
            else:
                freq = self.word_freq[word][str(class_label)] 
                return round(freq / (self.word_sums[str(class_label)]), 6)
        else:
            print("Frequency of words not obtain yet. Call the frequency method first.")
            
    def __calculate_probabilites(self):
        for word in self.vocabulary:
            self.word_probabilities[word] = dict()
            for label in self.classes:
                prob = self.__probability(word=word, class_label=str(label))
                self.word_probabilities[word][str(label)] = prob
                
    def fit_data(self):
        """
        Run this to obtain the frequencies and probabilities
        """
        self.__frequency()
        self.__calculate_probabilites()
        
    def get_probability(self, word: str):
        return self.word_probabilities[word]
                
    def predict(self, text):
        """
        This gives the log likelihood, meaning that a prediction greater than 0 is 
        considered positive and below is negative. This is for binary classification.
        """
        total_sum = 0
        classes = sorted(self.classes, reverse=True)
        for word in text.split():
            if word in self.word_probabilities:
                lambda_value = np.log(self.word_probabilities[word][str(classes[0])] / self.word_probabilities[word][str(classes[1])])
                total_sum += lambda_value
            else:
                total_sum += 0
        return total_sum

# combine all out tweets
all_tweets = all_positive_tweets + all_negative_tweets
# get labels for the tweets (5000 per class)
labels = [1 if i < 5000 else 0 for i in range(len(all_tweets))]

# create a dataframe holding the tweets and labels
df = pd.DataFrame()
df["tweets"] = all_tweets
df["labels"] = labels
df = df.sample(frac=1).reset_index(drop=True) # randomize the data

# preprocess all the tweets (see function "process_tweets")
processed_tweets = process_tweets(tweets=df["tweets"].values)
df["processed_tweets"] = processed_tweets # add the processed tweets to the dataframe

# training-test split on the dataframe (80-20 split)
train_size = int(len(df) * 0.80)
train_df = df[0:train_size]
test_df = df[train_size:]

# Get the training samples
tweet_sentiment_train = train_df[["processed_tweets", "labels"]].values # Note: using the processed tweets

# Tweet Classifier fitting on the training set
tf = TermFrequencyPrediction(data=tweet_sentiment_train)
tf.fit_data()

# get the predictions of the model on the test set
y_pred = [tf.predict(test_tweet) for test_tweet in test_df["processed_tweets"].values]

# convert to binary prediction (over 0 is positive, otherwise it is negative)
y_pred = np.where(np.array(y_pred) > 0, 1, 0)
y_test = test_df["labels"].values # get the true labels from the test dataframe

# determine accuracy of the model based on sum of the common labels between the actual labels and predicted labels 
acc = sum([1 if y_pred[i] == y_test[i] else 0 for i in range(len(y_pred))]) / len(y_test)