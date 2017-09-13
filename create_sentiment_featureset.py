import nltk
from nltk.tokenize import word_tokenize
# Stemming the words but creates a stem
# that is a valid dictionary word
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
from collections import Counter
import random
import io

lemmatizer = WordNetLemmatizer()

# To avoid the exhaution of memmory
hm_lines = 10000000


# Creating lexicons or the feature words
# that define the classification
def create_lexicons(pos,neg):
    lexicon = []

    for file in [pos,neg]:
        with io.open(file,'r', encoding='cp437') as f:
            content = f.readlines()
            for l in content[:hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    l2 = []

    for w in w_counts :
        # Not to include stop words and words
        # that occur very few no. of times
        if 1000 >  w_counts[w] > 50:
            l2.append(w)

    print len(l2)
    return l2


# creating a neat featureset of the sample texts
def sample_handling(sample, lexicon, classification):
    featureset = []

    with io.open(sample, 'r', encoding = 'cp437') as f:
        content = f.readlines()

        # Same operations as applied for lexicons
        for l in content[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words :
                if word.lower() in lexicon :
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            # Storing in the format [feature , label]
            featureset.append([features, classification])


    return featureset

# Creation of the training and testing sets
# for classification and testing

def create_feature_sets_and_labels(pos, neg, test_size =0.1):
    lexicon = create_lexicons(pos,neg)
    features = []
    features += sample_handling('pos.txt', lexicon, [1,0])
    features += sample_handling('neg.txt', lexicon, [0,1])

    random.shuffle(features)
    # Creating a testing oF 10%  of the total corpus
    testing_size = int(test_size*len(features))

    features = np.array(features)

    # name[:,0] --> selcts all the 0th index for the list of lists, here
    # that is features from the featureset
    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])

    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    return train_x, train_y, test_x, test_y

if __name__ == '__main__' :
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x,train_y, test_x, test_y], f)
