import pandas as pd
from collections import Counter

from unsupervised_learning.LatentDirichletAllocation.src.preprocess.data_processing import NewsGroupDataProcessing
from unsupervised_learning.LatentDirichletAllocation.src.lda.lda import LDA
from unsupervised_learning.LatentDirichletAllocation.src.tf_idf.tf_idf import TF_IDF
from unsupervised_learning.LatentDirichletAllocation.src.helper_functions.helpers import final_words_for_lda, final_dataset_creation, removing_short_documents

from sklearn.datasets import fetch_20newsgroups

def main(ALPHA, BETA):
    newsgroups_train = fetch_20newsgroups(subset='train')

    # Preprocess Dataset
    print("Preprocessing")
    ng = NewsGroupDataProcessing(newsgroups_train)
    dataset = ng.extract_data_based_on_class(classes_to_use=[14, 16])
    text, labels = ng.split_text_and_labels(dataset=dataset)
    text = ng.remove_numbers_and_punctuation_from_text(text_data=text)
    text = ng.remove_stopwords(text_data=text)
    text = ng.remove_single_letters(text_data=text)
    text = ng.lemmatize(text_data=text)

    # Apply Tf-Idf
    print("Applying Tf-Idf")
    tfidf = TF_IDF(data=text)
    tfidf.calculate_tf_idf()
    unique_words = tfidf.unique_words
    tfidf_df = tfidf.term_freq

    # Deciding which words to use for LDA
    print(f"Selecting our top words. Documents with less {15} words are removed")
    words_to_keep = final_words_for_lda(unique_words=unique_words, tfidf_df=tfidf_df, threshold=0.05)
    final_dataset = final_dataset_creation(text, words_to_keep)
    removing_short_documents(final_dataset, labels, less_than=15)

    df = pd.DataFrame()
    df["text"] = final_dataset
    df["label"] = labels

    print("LDA")
    lda = LDA(documents=final_dataset, alpha=ALPHA, beta=BETA, num_topics=2)
    lda.run(iterations=3)

    for key, value in Counter(labels).items():
        print(f"Label {key} should have {value} documents")
    for key, value in ng.class_names.items():
        if key in labels:
            print(key, ng.class_names[key])
    lda.display_document_to_topic_distribution()
    lda.get_top_words_per_topic(num_top_words=10)

if __name__ == '__main__':
    # LDA hyperparameters
    ALPHA = 1
    BETA = 0.1
    main(ALPHA=ALPHA, BETA=BETA)


