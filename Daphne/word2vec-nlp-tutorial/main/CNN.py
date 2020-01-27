import os
os.chdir("C:/Users/daphn/Documents/Kaggle/word2vec-nlp-tutorial")

import pandas as pd
train = pd.read_csv("data/labeledTrainData.tsv", header=0,
                    delimiter="\t", quoting=3)
# Add the extra data
extra = pd.read_csv('data/extradata.csv',
                    encoding="latin-1")

extra = extra.drop(['Unnamed: 0', 'type', 'file'],
                   axis=1)
extra.columns = ["review", "sentiment"]

#remove half of it, unsupervised learning
extra = extra[extra.sentiment != 'unsup']
extra['sentiment'] = extra['sentiment'].map({'pos': 1,
                                             'neg': 0})
# MERGE
train = pd.concat([train, extra]).reset_index(drop=True)

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

# Shortest review has 32 characters
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

train["review"] = train["review"] \
    .apply(lambda review: remove_emoji(review))

# Exchange emoticons for their meaning
def convert_emoticons(text):
    for emot in EMOTICONS:
        text = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), text)
    return text

train["review"] = train["review"] \
    .apply(lambda review: convert_emoticons(review))

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
# probably 50 000 is a bit too much
n_rare_words = 230000
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

from nltk import word_tokenize
tokens = [word_tokenize(sen) for sen in train.review_cleaned]
train['tokens'] = tokens

# Replace numbers
# import num2words
# ...

# Reshaping into a list of set (list(words in review), sentiment)
docs = train["review_cleaned"].to_list()

# CNN tutorial
#train["review_final"] = train["review_cleaned"].str.split()

train['Pos'] = np.where(train["sentiment"] == 1, 1, 0)
train['Neg'] = np.where(train["sentiment"] == 0, 1, 0)

from sklearn.model_selection import train_test_split
data_train, data_test = train_test_split(train,
                                         test_size=0.10,
                                         random_state=42)

all_training_words = [word for tokens in data_train["tokens"]
                      for word in tokens]
training_sentence_lengths = [len(tokens) for tokens
                             in data_train["tokens"]]

TRAINING_VOCAB = sorted(list(set(all_training_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_training_words), len(TRAINING_VOCAB)))
print("Max sentence length is %s" % max(training_sentence_lengths))

all_test_words = [word for tokens in data_test["tokens"]
                  for word in tokens]
test_sentence_lengths = [len(tokens) for tokens
                         in data_test["tokens"]]
TEST_VOCAB = sorted(list(set(all_test_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_test_words), len(TEST_VOCAB)))
print("Max sentence length is %s" % max(test_sentence_lengths))


from gensim import models
# https://github.com/mmihaltz/word2vec-GoogleNews-vectors
word2vec_path = './data/word2vec-GoogleNews-vectors/GoogleNews-vectors-negative300.bin.gz'
# Some explanation: https://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

# Keras works with tensorflow
# for it to work you will need to dl the lastest version of virtual studio c++
# https://support.microsoft.com/fr-fr/help/2977003/the-latest-supported-visual-c-downloads
# Then for NVIDIA you will definitely need this one
# https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781
# and this
# https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-windows
# and this (here you are interested in the VCS root definition)
# https://www.quora.com/How-does-one-install-TensorFlow-to-use-with-PyCharm

from keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D, Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model


MAX_SEQUENCE_LENGTH = 1289
EMBEDDING_DIM = 300

# Tokenize and Pad sequences
tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB),
                      lower=True, char_level=False)

tokenizer.fit_on_texts(data_train["review_cleaned"].tolist())

training_sequences = tokenizer.texts_to_sequences(
    data_train["review_cleaned"].tolist())

train_word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(train_word_index))

# This function transforms a list of num_samples sequences (lists of integers) into a 2D Numpy array
train_cnn_data = pad_sequences(training_sequences,
                               maxlen=MAX_SEQUENCE_LENGTH)

train_embedding_weights = np.zeros(
    (len(train_word_index)+1, EMBEDDING_DIM))
for word,index in train_word_index.items():
    train_embedding_weights[index, :] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
print(train_embedding_weights.shape)

test_sequences = tokenizer.texts_to_sequences(
    data_test["review_cleaned"].tolist())
test_cnn_data = pad_sequences(test_sequences,
                              maxlen=MAX_SEQUENCE_LENGTH)

# Convutional Neural Networks
# https://www.youtube.com/watch?v=9aYuQmMJvjA
# Historically for Image processing but it has been out-performing
# the Recurrent Neural Network on sequence tasks.
# High level explanation: accepts 2D and 3D input
# Image => 2D array of pixels => then convulutions on this array
# i.e try to locate features on window x by x (also called kernel)
# finding shapes and curves and corners and etc.
# once done slide the window
# then condensing the image by keeping the results of the convultions
# then pooling (complex algo) by taking the max value
# Each layer will try to identify patterns in the convultions from
# before.

# The tuto is about imagery and gives the steps for preprocessing those

# You have to be careful of the balance of the training data otherwise
# the NN will optimize for the over-represented class and get stuck.

# YOU NEED TO SHUFFLE THE DATA before training!


def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index):
    embedding_layer = Embedding(num_words,
                                embedding_dim,
                                weights=[embeddings],
                                input_length=max_sequence_length,
                                trainable=False)

    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    convs = []
    filter_sizes = [2, 3, 4, 5, 6]

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=200,
                        kernel_size=filter_size,
                        activation='relu')(embedded_sequences)
        l_pool = GlobalMaxPooling1D()(l_conv)
        convs.append(l_pool)

    l_merge = concatenate(convs, axis=1)

    x = Dropout(0.1)(l_merge)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    preds = Dense(labels_index, activation='sigmoid')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    model.summary()
    return model

label_names = ['Pos', 'Neg']
y_train = data_train[label_names].values

x_train = train_cnn_data
y_tr = y_train

model = ConvNet(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index)+1, EMBEDDING_DIM,
                len(list(label_names)))

# OUTPUT
# Model: "model_1"
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to
# ==================================================================================================
# input_1 (InputLayer)            (None, 50)           0
# __________________________________________________________________________________________________
# embedding_1 (Embedding)         (None, 50, 300)      24237600    input_1[0][0]
# __________________________________________________________________________________________________
# conv1d_1 (Conv1D)               (None, 49, 200)      120200      embedding_1[0][0]
# __________________________________________________________________________________________________
# conv1d_2 (Conv1D)               (None, 48, 200)      180200      embedding_1[0][0]
# __________________________________________________________________________________________________
# conv1d_3 (Conv1D)               (None, 47, 200)      240200      embedding_1[0][0]
# __________________________________________________________________________________________________
# conv1d_4 (Conv1D)               (None, 46, 200)      300200      embedding_1[0][0]
# __________________________________________________________________________________________________
# conv1d_5 (Conv1D)               (None, 45, 200)      360200      embedding_1[0][0]
# __________________________________________________________________________________________________
# global_max_pooling1d_1 (GlobalM (None, 200)          0           conv1d_1[0][0]
# __________________________________________________________________________________________________
# global_max_pooling1d_2 (GlobalM (None, 200)          0           conv1d_2[0][0]
# __________________________________________________________________________________________________
# global_max_pooling1d_3 (GlobalM (None, 200)          0           conv1d_3[0][0]
# __________________________________________________________________________________________________
# global_max_pooling1d_4 (GlobalM (None, 200)          0           conv1d_4[0][0]
# __________________________________________________________________________________________________
# global_max_pooling1d_5 (GlobalM (None, 200)          0           conv1d_5[0][0]
# __________________________________________________________________________________________________
# concatenate_1 (Concatenate)     (None, 1000)         0           global_max_pooling1d_1[0][0]
#                                                                  global_max_pooling1d_2[0][0]
#                                                                  global_max_pooling1d_3[0][0]
#                                                                  global_max_pooling1d_4[0][0]
#                                                                  global_max_pooling1d_5[0][0]
# __________________________________________________________________________________________________
# dropout_1 (Dropout)             (None, 1000)         0           concatenate_1[0][0]
# __________________________________________________________________________________________________
# dense_1 (Dense)                 (None, 128)          128128      dropout_1[0][0]
# __________________________________________________________________________________________________
# dropout_2 (Dropout)             (None, 128)          0           dense_1[0][0]
# __________________________________________________________________________________________________
# dense_2 (Dense)                 (None, 2)            258         dropout_2[0][0]
# ==================================================================================================
# Total params: 25,566,986
# Trainable params: 1,329,386
# Non-trainable params: 24,237,600
# __________________________________________________________________________________________________

# Train CNN
num_epochs = 3
batch_size = 34

hist = model.fit(x_train, y_tr, epochs=num_epochs, validation_split=0.1,
                 shuffle=True, batch_size=batch_size)
import matplotlib.pyplot as plt

# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.title('model train vs validation loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper right')
# plt.show()



# Test CNN
predictions = model.predict(test_cnn_data, batch_size=1024, verbose=1)

labels = [1, 0]
prediction_labels=[]
for p in predictions:
    prediction_labels.append(labels[np.argmax(p)])

sum(data_test.sentiment == prediction_labels)/len(prediction_labels)
# By reducing the number of words in the vocabulary (removing the 150 000
# rarest words), we actually gained: 0.01 in accuracy
data_test.sentiment.value_counts()

labels = ["Pos_cnn", "Neg_cnn"]
df_predictions = pd.DataFrame(data=predictions, columns=labels)
data_test.reset_index(drop=True, inplace=True)
df_predictions.reset_index(drop=True, inplace=True)
hh = pd.concat([data_test, df_predictions], axis=1)

hh["threshold"] = np.where(
    (hh["Pos_cnn"] < 0.6) & (hh["Pos_cnn"] > 0.4), True, False)

# Some manual corrections could be added
sum(hh["threshold"] == True)
cut = hh[hh["threshold"] == True]
#################################################################################
# Trying to score
test = pd.read_csv("data/testData.tsv", header=0,
                    delimiter="\t", quoting=3)

test["review_cleaned"] = test["review"].str.lower()

# Remove the emojis
test["review_cleaned"] = test["review_cleaned"] \
    .apply(lambda review: remove_emoji(review))

# Exchange emoticons for their meaning
test["review_cleaned"] = test["review_cleaned"] \
    .apply(lambda review: convert_emoticons(review))

# Remove HTML
test["review_cleaned"] = test["review_cleaned"].apply(
    lambda review: remove_html(review)
)

# Removing the punctuation
test["review_cleaned"] = test["review_cleaned"].str.translate(
    str.maketrans("", "", string.punctuation)
)

# Removing stop words
STOPWORDS = set(stopwords.words('english'))
test["review_cleaned"] = test["review_cleaned"].apply(
    lambda review: remove_stopwords(review))

# Removal of the too frequent words
test["review_cleaned"] = test["review_cleaned"].apply(
    lambda text: remove_freqwords(text)
)

# Removal of rare words
test["review_cleaned"] = test["review_cleaned"] \
    .apply(lambda review: remove_rarewords(review))


# Lemmatization
test["review_cleaned"] = test["review_cleaned"] \
    .apply(lambda review: lemmatize_words(review))

#test["review_final"] = test["review_cleaned"].str.split()

# Apply the model
test_sequences = tokenizer.texts_to_sequences(
    test["review_cleaned"].tolist())
test_cnn_data = pad_sequences(test_sequences,
                              maxlen=MAX_SEQUENCE_LENGTH)

predictions = model.predict(test_cnn_data, batch_size=1024, verbose=1)

labels = ["Pos", "Neg"]
prediction_labels=[]
for p in predictions:
    prediction_labels.append(labels[np.argmax(p)])

df_predictions = pd.DataFrame(data=predictions, columns=labels)

essai = pd.concat([test, df_predictions], axis=1)
essai["threshold"] = np.where((essai["Pos"] < 0.6) & (essai["Pos"] > 0.4), True, False)

# Finding the review in test and train
sub_train = train[["review", "sentiment"]]
sub_train = sub_train.rename(columns={"sentiment": "true_sentiment"})
mergedStuff = pd.merge(essai, sub_train,  on=['review'], how='left')

len(mergedStuff)
sum(mergedStuff["true_sentiment"] == 1)
sum(mergedStuff["true_sentiment"] == 0)

mergedStuff["Pos"] = np.where(mergedStuff["true_sentiment"] == 1, 1, mergedStuff["Pos"])
mergedStuff["Pos"] = np.where(mergedStuff["true_sentiment"] == 0, 0, mergedStuff["Pos"])

mergedStuff["threshold"] = np.where((mergedStuff["Pos"] < 0.6) & (mergedStuff["Pos"] > 0.4), True, False)

mergedStuff = mergedStuff[["id", "review", "Pos", "Neg", "threshold"]]
#mergedStuff.to_excel("data/manual_classification.xlsx", index=False)

# Some manual corrections could be added
sum(mergedStuff["threshold"] == True)

# Read back
# Actually lost a bit of accuracy. My understanding of positive and negative is not strong enough
# The gem of this boring reading:
# "I've heard a lot about Porno Holocaust and its twin film Erotic Nights Of The Living Dead.
# Both films are interchangeable and were filmed at the same time on the same location with
# the same actors changing clothes for each film (and taking them off).
# If you are expecting the D'Amato genius displayed in films like Buio Omega
# or Death Smiles on Murder, you won't find it here. Nonetheless this film has a charm
# that exploitation fans will not be able to resist. Where else will you see hardcore sex mixed
# with a zombie/monster and his enormous penis that strangles and chokes women to death? Only from D'Amato.
# There is some amount of gore in which many of the men are bludgeoned to death.
# The film is set on a beautiful tropical island. As far as I know there is no subtitled version,
# so if you don't speak Italian you wont know what is going on...but who cares right?
# In all honesty, Gore fans will probably fast forward through the hardcore sex.
# And if anyone is actually watching this for the sex only, will for sure be offended instantly.
# I can just imagine modern day porn fans tracing back through D'Amato's output and coming across this atrocity!
# Out of the two I find Erotic Nights Of The Living Dead far superior.
# But, don't bother watching either if they are cut. Porno Holocaust is extremely low budget as expected.
# Even the monster looks no where as good as George Eastman's character in Anthropophagus.
# The film is worth watching for laughs and to complete your D'Amato film quest."
essai = pd.read_excel("data/manual_classification.xlsx")

essai["Pos"] = np.where(essai["true_sentiment"] == 1, 1, essai["Pos"])
essai["Pos"] = np.where(essai["true_sentiment"] == -1, 0, essai["Pos"])

#essai["sentiment"] = np.where(essai["Pos"] > essai["Neg"], 1, 0)
#essai["sentiment"] = essai["Pos"]

# This methods carry no goods because the missclassified drag the score down more than the well-classified
#essai["sentiment"] = essai["Pos"]
#essai["sentiment"] = np.where(essai["sentiment"] > 0.80, 1, essai["sentiment"])
#essai["sentiment"] = np.where(essai["sentiment"] < 0.20, 0, essai["sentiment"])

# Submission file
submission = essai[["id", "sentiment"]]
submission.to_csv("data/submission_cnn_padded_rounding.csv", index=False, quoting=3)


