#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 20:27:09 2021

@author: chris
"""
from nltk.corpus import twitter_samples
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from processtxt import TextProcessing
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

NLP = spacy.load("en_core_web_sm")

# Global varibales for Padding and Maximum Vocabulary
PADDING = 50
MAX_TOKENS = 10000

# Get the positive and negative tweets from NLTK
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# Combine all the tweets
all_tweets = all_positive_tweets + all_negative_tweets

# Get the labels (5000 per class)
labels = [1 if i < 5000 else 0 for i in range(10000)]

# Custom class to preprocess data
# NOTE: I made this package. I left this in to highlight the preprocessing steps, however the implementation is not in this file
tp = TextProcessing(nlp=NLP, stopwords=STOP_WORDS)

def process_tweets(tweets):
    processed_tweets = list()
    for i, tweet in enumerate(tweets):
        if type(tweet) == str:
            temp_text = tp.lowercase(text=tweet)
            temp_text = tp.expand_contractions(text=temp_text)
            temp_text = tp.remove_urls(text=temp_text)
            temp_text = tp.remove_punctuation(text=temp_text)
            temp_text = tp.remove_stopwords(text=temp_text)
            temp_text = tp.remove_extra_whitespace(text=temp_text)
            processed_tweets.append(temp_text)
        else:
            processed_tweets.append("")
    return processed_tweets

# create a dataframe holding the tweets and labels
df = pd.DataFrame()
df["tweets"] = all_tweets
df["labels"] = labels
df = df.sample(frac=1).reset_index(drop=True)

processed_tweets = process_tweets(tweets=df["tweets"].values) # randomize the data
df["processed_tweets"] = processed_tweets

# training-test split on the dataframe (80-20 split)
train_size = int(len(df) * 0.80)
train_df = df[0:train_size]
test_df = df[train_size:]

# split training and test set into their features and labels (X= features, y=labels)
X_train = train_df["processed_tweets"].values
X_test = test_df["processed_tweets"].values
y_train = train_df["labels"].values
y_test = test_df["labels"].values

# Tensorflow vectorizer to add padding and include only the top words (based on global variables)
vectorize_layer = TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode='int',
    output_sequence_length=PADDING)

# fit the training set features only
vectorize_layer.adapt(X_train)

def vectorize_text(text):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text)

# get the voab size plus 1. The 1 is for instances where a word is not found from the training set
VOCAB_SIZE = len(vectorize_layer.get_vocabulary()) + 1

# transform the features into numerical format, with the added padding
X_train = vectorize_text(X_train)
X_test = vectorize_text(X_test)

# Create a simple feed forward network
# This model has an embedding layer based on one-hot-encoding and the padding
# Word embedding weights can be used here as well - your own or pre-trained(i.e. word2vec, Glove)
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 50, input_length=PADDING, trainable=True),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
 ])

# binary cross-entropy is used here because our output is positive or negative
model.compile(optimizer='adam', loss='binary_crossentropy', 
metrics=['accuracy'])

model.summary()

# fit the data
history = model.fit(X_train, y_train,
                    epochs=100,
                    verbose=True,
                    validation_data=(X_test, y_test),
                    batch_size=10)

# check how well the model did on the test set
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print(f"Training Accuracy: {accuracy}")
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print(f"Testing Accuracy:  {accuracy}")