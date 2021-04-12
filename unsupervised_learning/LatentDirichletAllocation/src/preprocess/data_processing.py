import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class NewsGroupDataProcessing:
    def __init__(self, dataset):
        self.data = dataset.copy()
        self.text = dataset.data
        self.labels = list(dataset.target)
        self.class_names = {i: label for i, label in enumerate(dataset.target_names)}
        self.stopwords_english = set(stopwords.words("english"))
        self.stop_punctuation = set(string.punctuation)
        self.top_numbers = set(string.digits)
        self.letters = set(string.ascii_lowercase)

    def extract_data_based_on_class(self, classes_to_use):
        dataset = list()
        for i, label in enumerate(self.labels):
            if label in classes_to_use:
                data = (self.text[i], self.labels[i])
                dataset.append(data)
        return dataset

    def split_text_and_labels(self, dataset):
        text, labels = list(), list()
        for t, l in dataset:
            text.append(t)
            labels.append(l)
        return text, labels

    def remove_numbers_and_punctuation_from_text(self, text_data):
        new_text = list()
        for i in range(len(text_data)):
            temp = text_data[i]
            for punc in string.punctuation:
                if punc in temp:
                    temp = temp.replace(punc, " ")
            for digit in string.digits:
                if digit in temp:
                    temp = temp.replace(digit, "")
            new_text.append(temp)
        return new_text

    def remove_stopwords(self, text_data):
        new_text = list()
        for i, row in enumerate(text_data):
            temp = list()
            split_row = row.split()
            for token in split_row:
                if token.lower() not in self.stopwords_english:
                    temp.append(token.lower())
            new_text.append(temp)
        return new_text

    def remove_single_letters(self, text_data):
        new_text = list()
        for i, row in enumerate(text_data):
            temp = list()
            for token in row:
                if token.lower() not in self.letters:
                    temp.append(token.lower())
            new_text.append(temp)
        return new_text

    def lemmatize(self, text_data):
        new_text = list()
        for i, row, in enumerate(text_data):
            temp = list()
            for token in row:
                stemmed_token = self.stem_token(token)
                temp.append(stemmed_token)
            new_text.append(temp)
        return new_text

    def stem_token(self, token):
        return WordNetLemmatizer().lemmatize(token, pos='v')
