from collections import Counter

import random

class LDA:
    def __init__(self, documents, alpha, beta, num_topics):
        self.documents = documents
        self.alpha = alpha
        self.beta = beta
        self.num_topics = num_topics
        self.topics = self.__create_topics()
        self.unique_words = self.__extract_unique_words()
        self.doc_with_topic = self.__create_doc_with_topics()
        self.word_dist = self.__extract_word_distribution()
        self.document_topic_distribution = None
        self.topic_word_dist = None
        self.computed = False

    def __extract_unique_words(self):
        unique_words = set()
        for doc in self.documents:
            for word in doc:
                unique_words.add(word)
        return unique_words

    def __create_topics(self):
        return [f"topic_{i}" for i in range(self.num_topics)]

    # Randomly assigned at first
    def __create_doc_with_topics(self):
        new_documents = list()
        for doc in self.documents:
            temp_doc = list()
            for word in doc:
                word_topic_tuple = (word, random.choice(self.topics))
                temp_doc.append(word_topic_tuple)
            new_documents.append(temp_doc)
        return new_documents

    # Number of words per topic in all documents
    def __extract_word_distribution(self):
        word_dist = {word: {topic: self.beta for topic in self.topics} for word in self.unique_words}
        for doc in self.doc_with_topic:
            for word, topic in doc:
                word_dist[word][topic] += 1
        return word_dist

    def __get_document_topic_dist(self, current_word_index, current_document):
        doc_dist = {topic: self.alpha for topic in self.topics}
        for i in range(len(current_document)):
            if i != current_word_index:
                word, topic = current_document[i]
                doc_dist[topic] += 1
        return doc_dist

    def __subtract_one_from_word_dist(self, focus_word):
        word, topic = focus_word
        self.word_dist[word][topic] -= 1

    def __topic_for_word_selection(self, doc_word_dist, focus):
        focus_word, topic = focus
        topic_prob = {topic: dict() for topic in self.topics}
        total = 0
        for topic, count in doc_word_dist.items():
            prob = count * self.word_dist[focus_word][topic]
            topic_prob[topic]["count"] = prob
            total += prob
        for topic, value in topic_prob.items():
            topic_prob[topic]["prob"] = round(topic_prob[topic]["count"] / total, 3)
        # pp.pprint(topic_prob)
        # print(sorted(topic_prob))
        random_selection = round(random.uniform(0, 1), 3)
        from_min = 0.0
        last_topic = None
        for topic, value in topic_prob.items():
            to_max = from_min + value["prob"]
            # print(from_min, to_max, random_selection)
            if from_min <= random_selection <= to_max:
                # print(topic, random_selection)
                return topic
            from_min = to_max
            last_topic = topic
        return last_topic

    def update_word_dist(self, word, new_topic):
        self.word_dist[word][new_topic] += 1

    def run(self, iterations):
        test_doc_dist = None
        for i in range(iterations):
            print(f"On iteration {i}")
            for i in range(len(self.doc_with_topic)):
                doc = self.doc_with_topic[i]
                for j in range(len(self.doc_with_topic[i])):
                    index = j
                    focus = doc[index]
                    self.__subtract_one_from_word_dist(focus_word=focus)
                    doc_topic_dist = self.__get_document_topic_dist(current_word_index=index, current_document=doc)
                    # pp.pprint(doc_topic_dist)
                    new_topic = self.__topic_for_word_selection(doc_word_dist=doc_topic_dist, focus=focus)
                    word = focus[0]
                    self.doc_with_topic[i][index] = (word, new_topic)
                    self.update_word_dist(word=word, new_topic=new_topic)
                    # print("UPDATE")
                    # pp.pprint(self.word_dist)
                    test_doc_dist = doc_topic_dist
        distribution = {i: {topic: 0 for topic in self.topics} for i in range(len(self.documents))}
        topic_word_dist = {topic: {word: 0 for word in self.unique_words} for topic in self.topics}
        for i, doc in enumerate(self.doc_with_topic):
            total = len(doc)
            for word, topic in doc:
                distribution[i][topic] += 1
                topic_word_dist[topic][word] += 1
            for topic in distribution[i]:
                distribution[i][topic] = (distribution[i][topic] / total)
        self.document_topic_distribution = distribution
        self.topic_word_dist = topic_word_dist
        self.computed = True

    def get_top_words_per_topic(self, num_top_words=10):
        if self.computed :
            for topic, words, in self.topic_word_dist.items():
                s = sorted(words.items(), key=lambda x: x[1], reverse=True)
                print(topic, s[0:num_top_words])

    def display_document_to_topic_distribution(self):
        if self.computed:
            best_topics = list()
            for doc, topics in self.document_topic_distribution.items():
                max_topic_value = 0
                best_topic = None
                for topic, value in topics.items():
                    if value > max_topic_value:
                        max_topic_value = value
                        best_topic = topic
                best_topics.append(best_topic)
            print(Counter(best_topics))