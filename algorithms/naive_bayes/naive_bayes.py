

class NaiveBayes:
    def __init__(self, alpha):
        self.alpha = alpha
        self.data = None
        self.labels = None
        self.total_rows = None
        self.unique_words = dict()
        self.class_dist = dict()
        self.word_dist = dict()

    def fit(self, data, unique_labels):
        self.data = data
        self.labels = unique_labels
        self.total_rows = len(self.data)
        self.__unique_words_and_counts()
        self.__class_distribution()
        self.__word_distribution()

    def get_unique_words(self):
        if len(self.unique_words) > 0:
            return set(self.unique_words.keys())

    def get_class_distribution(self):
        return self.class_dist

    def get_word_distribution_by_class(self, class_name):
        return self.word_dist[class_name]

    def predict(self, test_list_data: list):
        return [self.__predict_helper(text) for text in test_list_data]

    def __predict_helper(self, text):
        label_prediction = None
        highest = None
        proba = list()
        for label in self.labels:
            p_label = self.class_dist[label][f"P({label})"]
            word_proba_product = 1.0
            for word in text.split():
                if word in self.word_dist[label]["word_proba"]:
                    word_proba_product *= self.word_dist[label]["word_proba"][word]
            probability = p_label * word_proba_product
            proba.append(probability)
            if highest is None or probability > highest:
                highest = probability
                label_prediction = label
        # print(label_prediction, highest)
        label_probability = round(highest / sum(proba), 3)
        return (label_prediction, label_probability)

    def __unique_words_and_counts(self):
        for row in self.data:
            for word_token in row[0].split():
                if word_token in self.unique_words:
                    self.unique_words[word_token] += 1
                else:
                    self.unique_words[word_token] = 1

    def __class_distribution(self):
        for row in self.data:
            label = row[1]
            if label in self.class_dist:
                self.class_dist[label]["count"] += 1
            else:
                self.class_dist[label] = {"count": 1}
        for label in self.labels:
            key = f"P({label})"
            self.class_dist[label][key] = round(self.class_dist[label]["count"] / self.total_rows, 3)

    def __word_distribution(self):
        for label in self.labels:
            self.word_dist[label] = {"total_num_words": 0,
                                     "words": {word: self.alpha for word in self.unique_words.keys()},
                                     "word_proba": dict()}
        self.__get_total_word_count_for_labels()
        self.__get_word_probability()

    def __get_total_word_count_for_labels(self):
        for row in self.data:
            text = row[0].split()
            label = row[1]
            self.word_dist[label]["total_num_words"] += len(text)
            self.__get_individual_word_count(text, label)
        for label in self.labels:
            self.word_dist[label]["total_num_words"] += (self.alpha * len(self.unique_words))

    def __get_individual_word_count(self, row_text, row_label):
        for word in row_text:
            if word in self.word_dist[row_label]["words"]:
                self.word_dist[row_label]["words"][word] += 1
            else:
                self.word_dist[row_label]["words"][word] = 1

    def __get_word_probability(self):
        for key, value in self.word_dist.items():
            for word, count in value["words"].items():
                value["word_proba"][word] = round((count / value["total_num_words"]), 3)