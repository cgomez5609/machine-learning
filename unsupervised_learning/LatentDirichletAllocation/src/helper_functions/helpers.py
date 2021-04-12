# Decide which words to keep for topic modeling
def final_words_for_lda(unique_words, tfidf_df, threshold=0.05):
    words_to_keep = list()
    for word in unique_words:
        if max(tfidf_df[word].values) >= threshold:
            words_to_keep.append(word)
    return words_to_keep

def final_dataset_creation(text, words_to_keep):
    final_dataset = list()
    for doc in text:
        temp = doc.copy()
        new_row = list()
        for word in words_to_keep:
            if word in temp:
                new_row.append(word)
        final_dataset.append(new_row)
    return final_dataset

def removing_short_documents(final_dataset, labels, less_than=10):
    indices_to_remove = list()
    count = 0
    for i, row in enumerate(final_dataset):
        if len(row) < less_than:
            # print("here", i)
            count += 1
            indices_to_remove.append(i)
    print(count)
    final_dataset = [value for i, value in enumerate(final_dataset) if i not in indices_to_remove]
    labels = [value for i, value in enumerate(labels) if i not in indices_to_remove]
    return final_dataset, labels