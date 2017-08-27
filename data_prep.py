from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 100000000


# Responsible to create a list with all the words that matter already lemmatized
def create_lexicon(pos, neg):
    lex = []
    for fi in [pos, neg]:
        with open(fi, 'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l)
                lex += list(all_words)
    lex = [lemmatizer.lemmatize(i) for i in lex]
    w_counts = Counter(lex)

    filtered_list = [d for d in w_counts if 1000 > w_counts[d] > 50]

    return filtered_list


# Prepares the sentences changing them into the hot vector
def sentence_to_vector(sample, lexicon, classification):
    featureset = []

    with open(sample,'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1

            features = list(features)
            featureset.append([features, classification])

    return featureset


# Actually do the work by receiving the datasets and doing all the data preparation
def generate_content(pos, neg, test_size=0.1):
    current_lexicon = create_lexicon(pos, neg)
    features = []
    features += sentence_to_vector('pos.txt', current_lexicon, [1, 0])
    features += sentence_to_vector('neg.txt', current_lexicon, [0, 1])
    random.shuffle(features)
    features = np.array(features)

    testing_size = int(test_size*len(features))

    train_x = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])
    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    # Saves the lexicon and the hot vector into pickle files to be read in the coming codes
    final_lexicon = create_lexicon('pos.txt', 'neg.txt')
    train_x, train_y, test_x, test_y = generate_content('pos.txt', 'neg.txt')

    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)

    with open('lexicon.pickle', 'wb') as f2:
        pickle.dump(final_lexicon, f2)
