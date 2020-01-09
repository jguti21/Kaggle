# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 18:22:32 2020

@author: gutierj
"""

#Word2Vec
import os



os.chdir('C:/Users/jordi/Desktop/Work/Kaggle/Word2Vec')

os.chdir('C:/Users/gutierj/Desktop/Programming/Kaggle/Word2Vec')


#To do:
#    stemming
#    tokenization
#    lemmatization
#    voting models (gpu?)
#    TF-IDF
#   Master's project stuff

#notebook of the tutorial
#https://nbviewer.jupyter.org/github/MatthieuBizien/Bag-popcorn/blob/master/Kaggle-Word2Vec.ipynb

# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
import pandas as pd       
train = pd.read_csv("labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)

train.shape

#print(train["review"][0])

#$ sudo pip install BeautifulSoup4

# Import BeautifulSoup into your workspace
from bs4 import BeautifulSoup             

# Initialize the BeautifulSoup object on a single movie review     
example1 = BeautifulSoup(train["review"][0])  

# Print the raw review and then the output of get_text(), for 
# comparison
#print(train["review"][0])
print(example1.get_text())


import re
# Use regular expressions to do a find-and-replace
letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      example1.get_text() )  # The text to search
print(letters_only)

lower_case = letters_only.lower()        # Convert to lower case
words = lower_case.split()               # Split into words


import nltk
nltk.download()  # Download text data sets, including stop words

from nltk.corpus import stopwords # Import the stop word list
print(stopwords.words("english"))


# Remove stop words from "words"
words = [w for w in words if not w in stopwords.words("english")]
print(words)


def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))  
    

clean_review = review_to_words( train["review"][0] )
print(clean_review)



# Get the number of reviews based on the dataframe column size
num_reviews = train["review"].size

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

"""
# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for i in range( 0, num_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_train_reviews.append( review_to_words( train["review"][i] ) )
"""    
    
print("Cleaning and parsing the training set movie reviews...\n")
clean_train_reviews = []
for i in range( 0, num_reviews ):
    # If the index is evenly divisible by 1000, print a message
    if( (i+1)%1000 == 0 ):
        print("Review %d of %d\n" % ( i+1, num_reviews )  )                                                             
    clean_train_reviews.append( review_to_words( train["review"][i] ))


print ("Creating the bag of words...\n")
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()    


# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
print(vocab)


import numpy as np

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print(count, tag)
    
    
print("Training the random forest...")
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, train["sentiment"] )



# Read the test data
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", \
                   quoting=3 )

# Verify that there are 25,000 rows and 2 columns
print (test.shape)

# Create an empty list and append the clean reviews one by one
num_reviews = len(test["review"])
clean_test_reviews = [] 

print ("Cleaning and parsing the test set movie reviews...\n")
for i in range(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print ("Review %d of %d\n" % (i+1, num_reviews))
    clean_review = review_to_words( test["review"][i] )
    clean_test_reviews.append( clean_review )

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )    





############################################
# Tutorial part 2
############################################

import pandas as pd

# Read data from files 
train = pd.read_csv( "labeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3 )
test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3 )
unlabeled_train = pd.read_csv( "unlabeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3 )

# Verify the number of reviews that were read (100,000 in total)
print ("Read %d labeled train reviews, %d labeled test reviews, " \
 "and %d unlabeled reviews\n" % (train["review"].size,  
 test["review"].size, unlabeled_train["review"].size ))
 
 
 # Import various modules for string cleaning
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)
 
# Download the punkt tokenizer for sentence splitting
import nltk.data
nltk.download()   

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


sentences = []  # Initialize an empty list of sentences

print( "Parsing sentences from training set")
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print ("Parsing sentences from unlabeled set")
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)
    
    
# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print ("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)    


model.doesnt_match("man woman child kitchen".split())

model.doesnt_match("france england germany berlin".split())   

model.doesnt_match("paris berlin london austria".split())

model.most_similar("man")

model.most_similar("awful")




###########################################
# Part 3
###########################################

# Load a pre-trained model
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import numpy as np
import os
from KaggleWord2VecUtility import KaggleWord2VecUtility


# Define a function to create bags of centroids
#
def create_bag_of_centroids( wordlist, word_centroid_map ):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max( word_centroid_map.values() ) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids


if __name__ == '__main__':

    model = Word2Vec.load("300features_40minwords_10context")


    # ****** Run k-means on the word vectors and print a few clusters
    #

    start = time.time() # Start time

    # Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
    # average of 5 words per cluster
    word_vectors = model.wv.syn0
    num_clusters = word_vectors.shape[0] / 5

    # Initalize a k-means object and use it to extract centroids
    print "Running K means"
    kmeans_clustering = KMeans( n_clusters = num_clusters )
    idx = kmeans_clustering.fit_predict( word_vectors )

    # Get the end time and print how long the process took
    end = time.time()
    elapsed = end - start
    print "Time taken for K Means clustering: ", elapsed, "seconds."


    # Create a Word / Index dictionary, mapping each vocabulary word to
    # a cluster number
    word_centroid_map = dict(zip( model.wv.index2word, idx ))

    # Print the first ten clusters
    for cluster in xrange(0,10):
        #
        # Print the cluster number
        print "\nCluster %d" % cluster
        #
        # Find all of the words for that cluster number, and print them out
        words = []
        for i in xrange(0,len(word_centroid_map.values())):
            if( word_centroid_map.values()[i] == cluster ):
                words.append(word_centroid_map.keys()[i])
        print words




    # Create clean_train_reviews and clean_test_reviews as we did before
    #

    # Read data from files
    train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3 )
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3 )


    print "Cleaning training reviews"
    clean_train_reviews = []
    for review in train["review"]:
        clean_train_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, \
            remove_stopwords=True ))

    print "Cleaning test reviews"
    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, \
            remove_stopwords=True ))


    # ****** Create bags of centroids
    #
    # Pre-allocate an array for the training set bags of centroids (for speed)
    train_centroids = np.zeros( (train["review"].size, num_clusters), \
        dtype="float32" )

    # Transform the training set reviews into bags of centroids
    counter = 0
    for review in clean_train_reviews:
        train_centroids[counter] = create_bag_of_centroids( review, \
            word_centroid_map )
        counter += 1

    # Repeat for test reviews
    test_centroids = np.zeros(( test["review"].size, num_clusters), \
        dtype="float32" )

    counter = 0
    for review in clean_test_reviews:
        test_centroids[counter] = create_bag_of_centroids( review, \
            word_centroid_map )
        counter += 1


    # ****** Fit a random forest and extract predictions
    #
    forest = RandomForestClassifier(n_estimators = 100)

    # Fitting the forest may take a few minutes
    print "Fitting a random forest to labeled training data..."
    forest = forest.fit(train_centroids,train["sentiment"])
    result = forest.predict(test_centroids)

    # Write the test results
    output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
    output.to_csv("BagOfCentroids.csv", index=False, quoting=3)
    print "Wrote BagOfCentroids.csv"