import pandas as pd
import numpy as np

from collections import Counter

class TF_IDF:
    def __init__(self, data):
        self.org_data = data.copy()
        self.data = [set(row) for row in self.org_data]
        self.num_of_documents = len(self.data)
        self.unique_words = self.__get_unique_words()
        self.term_freq = pd.DataFrame(0.0, index=np.arange(self.num_of_documents), columns=self.unique_words, dtype=float)
        self.tf = None
        self.idf = None

    def __get_unique_words(self):
        unique_words = {word for row in self.data for word in row}
        return sorted(list(unique_words))

    def calculate_tf(self):
        tf = dict()
        for i, doc in enumerate(self.org_data):
            tf[i] = dict()
            num_words_in_doc = len(doc)
            word_counter = Counter(doc)
            for word in word_counter:
                tf[i][word] = (word_counter[word] / num_words_in_doc)
        self.tf = tf

    def calculate_tf_idf(self):
        self.calculate_tf()
        self.calculate_idf()
        for doc, words in self.tf.items():
            for word, value in words.items():
                self.term_freq.at[doc, word] = (value * self.idf[word]["idf"])

    def calculate_idf(self):
        idf = dict()
        for word in self.unique_words:
            num_docs_containing_word = 0
            for doc in self.data:
                if word in doc:
                    num_docs_containing_word += 1
            idf[word] = {"idf":np.log10((self.num_of_documents / num_docs_containing_word))}
        self.idf = idf
