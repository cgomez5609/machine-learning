"""
Naive Bayes implementation for text classification.
I used dictionaries to display probabilities for the labels and words.
This can be optimized by using matrices, but for the sake of instruction I used dictionaries.
Preprocessing was not considered here, but in real applications there will be a great deal
of preprocessing.
"""

from pprint import PrettyPrinter

from algorithms.naive_bayes.naive_bayes import NaiveBayes

training_data = [
    ["you are a jerk", "neg"],
    ["love you", "pos"],
    ["welcome home", "pos"],
    ["ass jerk", "neg"],
    ["ass licker", "neg"],
    ["you love ass", "neg"],
    ["a wonderful day love", "pos"],
    ["you are the best person buddy", "pos"]
]
test = ["love you friend", "fuck you jerk face", "ass ass ass"]

LABELS = {"pos", "neg"}
ALPHA = 1

def main():
    pp = PrettyPrinter()
    nb = NaiveBayes(alpha=ALPHA)
    nb.fit(data=training_data, unique_labels=LABELS)
    pp.pprint(nb.get_unique_words())
    pp.pprint(nb.get_class_distribution())
    pp.pprint(nb.get_word_distribution_by_class("pos"))
    pp.pprint(nb.get_word_distribution_by_class("neg"))
    y_pred = nb.predict(test_list_data=test)
    print(y_pred)


if __name__ == '__main__':
    main()
