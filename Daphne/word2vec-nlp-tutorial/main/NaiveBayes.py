import os
os.chdir("C:/Users/daphn/Documents/Kaggle/word2vec-nlp-tutorial")

import pandas as pd
train = pd.read_csv("data/labeledTrainData.tsv", header=0,
                    delimiter="\t", quoting=3)

# Inspection of the training set
# for sentiment:
#   - 1 is  positive
#   - 0 is negative
positive = train[train["sentiment"] == 1]

# The sample is equally distributed in positive and
# negative reviews
len(positive) / len(train)

train["characters"] = train["review"].str.len()
# Longest review has 13 710 characters
max(train["characters"])

# Shortest review has 54 characters
min(train["characters"])

# The shortest review is:
short = train[train["characters"] == min(train["characters"])]["review"]
print(list(short))

# Average number of letters is 1 329 characters
train["characters"].mean()

##
import numpy as np
import pandas as pd
import re
import nltk
import spacy
import string

import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

import num2words
from emot.emo_unicode import UNICODE_EMO, EMOTICONS

####### CLEANING ########
# To lower case
train["review_cleaned"] = train["review"].str.lower()

# Remove HTML
from bs4 import BeautifulSoup
def remove_html(review):
    review = BeautifulSoup(review).get_text()
    return review
train["review_cleaned"] = train["review_cleaned"].apply(
    lambda review: remove_html(review)
)

# Removing the punctuation
train["review_cleaned"] = train["review_cleaned"].str.translate(
    str.maketrans("", "", string.punctuation)
)

# Removing stop words
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split()
                     if word not in STOPWORDS])
train["review_cleaned"] = train["review_cleaned"].apply(
    lambda review: remove_stopwords(review))

# Removal of the too frequent words
from collections import Counter

cnt = Counter()
for review in train["review_cleaned"].values:
    for word in review.split():
        cnt[word] += 1

# 142 478 unique words in the dictionnary
# I will rearrange a little b/c good and bad are in there
FREQWORDS = set([w for (w, wc) in cnt.most_common(50)
                 if w not in ["good", "bad", "like", "worst",
                              "great", "love", "best"]])
def remove_freqwords(review):
    """custom function to remove the frequent words"""
    return " ".join([word for word in str(review).split()
                     if word not in FREQWORDS])
train["review_cleaned"] = train["review_cleaned"].apply(
    lambda text: remove_freqwords(text)
)

# Removal of rare words
n_rare_words = 50000
RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])

def remove_rarewords(review):
    """custom function to remove the rare words"""
    return " ".join([word for word in str(review).split()
                     if word not in RAREWORDS])

train["review_cleaned"] = train["review_cleaned"] \
    .apply(lambda review: remove_rarewords(review))

# Stemming
# Stemming is the process of reducing inflected
# (or sometimes derived) words to their word stem, base or root form
# from nltk.stem.porter import PorterStemmer
# stemmer = PorterStemmer()
# def stem_words(review):
#     return " ".join([stemmer.stem(word)
#                      for word in review.split()])
#
# train["review_cleaned"] = train["review_cleaned"] \
#     .apply(lambda review: stem_words(review))


# Lemmatization
# Lemmatization is similar to stemming in reducing
# inflected words to their word stem but differs
# in the way that it makes sure the root word
# (also called as lemma) belongs to the language.

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet # we need a corpus to
# know what is the type of the word

lemmatizer = WordNetLemmatizer()
wordnet_map = {"N": wordnet.NOUN,
               "V": wordnet.VERB,
               "J": wordnet.ADJ,
               "R": wordnet.ADV}

def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word,
                                          wordnet_map.get(pos[0],
                                                          wordnet.NOUN))
                     for word, pos in pos_tagged_text])


train["review_cleaned"] = train["review_cleaned"] \
    .apply(lambda review: lemmatize_words(review))


# Remove the emojis
def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

train["review_cleaned"] = train["review_cleaned"] \
    .apply(lambda review: remove_emoji(review))

# Exchange emoticons for their meaning
def convert_emoticons(text):
    for emot in EMOTICONS:
        text = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), text)
    return text

train["review_cleaned"] = train["review_cleaned"] \
    .apply(lambda review: convert_emoticons(review))

# Replace numbers
# import num2words
# ...

# Extra step could be to replace abbreviatiobs (like AFK)


# A bag-of-words representation of a document
# does not only contain specific words
# but all the unique words in a document
# and their frequencies of occurrences.
## BAG == set

# doc   w1    w2 ... wn   sentiment
# d1    2      1 ... 2    positive
# This is Corpus representation

# Naïve Bayes classification
# P(class|doc) = [P(doc|class)*P(class)]/P(doc)
# Naive Bayes Independence Assumptions:
    # 1. Bag of words assumption: Assume position does not matter.
    # 2. Conditional independence assumption (so that we can multiply)
    # and reduce combinations from X^n to X*n

# Hence:
# c_naiveB = argmax{c in C} P(c) PRODUCT{over X} OF P(X|class)

# Advantages of naiveB:
    # 1. Reduced nb of param
    # 2. Linear time complexity instead of exponential

# Note: when applied to text classification it is referred as
# Multinomial Naive Bayes classification

import nltk

# Define the feature extractor

# Reshaping into a list of set (list(words in review), sentiment)
list_train = []
for i in range(0, len(train)):
    list_train.append((train["review_cleaned"][i].split(),
                       train["sentiment"][i]))
del i

from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

fdist = FreqDist()
for i in range(0, len(train)):
    review = train["review_cleaned"][i]
    for word in word_tokenize(review):
        fdist[word.lower()] += 1
del i

word_features = list(fdist)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

# Train Naive Bayes classifier
featuresets = [(document_features(d), c) for (d,c) in list_train]
train_set, test_set = featuresets[0:int(len(list_train)/2)], \
                      featuresets[int(len(list_train)/2)+1::]

classifier = nltk.NaiveBayesClassifier.train(train_set)

# Test the classifier
print(nltk.classify.accuracy(classifier, test_set))

# Show the most important features as interpreted by Naive Bayes
classifier.show_most_informative_features(5)

# Trying to score
test = pd.read_csv("data/testData.tsv", header=0,
                    delimiter="\t", quoting=3)

# Train Naive Bayes classifier
def features(sentence):
    words = sentence.lower().split()
    return dict(('contains(%s)' % w, True) for w in words)

test["sentiment"] = None
for i in range(0, len(test)):
    review = features(test["review"][i])
    result = classifier.classify(review)
    test["sentiment"][i] = result

# Submission file
submission = test[["id", "sentiment"]]
submission.to_csv("data/submission1.csv", index=False, quoting=3)



# OLD
# from nltk.probability import FreqDist
# from nltk.tokenize import word_tokenize
#
# fdist = FreqDist()
# for i in range(0, len(train)):
#     review = train["review_cleaned"][i]
#     for word in word_tokenize(review):
#         fdist[word] += 1
# del i
#
# # Following the approach listed in "On Stopwords, Filtering and Data Sparsity for Sentiment Analysis of Twitter" of
# # Saif et al. We apply the Z-method which consists of plotting the frequency and cut-off the lower and higher part.
# # "The size of the stoplist corresponds to where an “elbow” appears in the plot."
# # The lower part looks a bit arbitrary
# #fdist.plot()
# freqs = [fdist[sample] for sample in fdist]
# freqs = sorted(freqs, reverse= True)
# ordered_filtered = [freq for freq in freqs if freq != 1]
# len(ordered_filtered)
# freqs[0:5]
# # from matplotlib import pyplot as plt
# # plt.plot(ordered_filtered)
# #
# word_features = list(fdist)[60:3500]
