# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 18:22:32 2020

@author: gutierj
"""

#Word2Vec


import os
import pandas as pd    
import numpy as np
from bs4 import BeautifulSoup             
import re
import nltk
from nltk.corpus import stopwords # Import the stop word list
from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer



from sklearn.ensemble import RandomForestClassifier


os.chdir('C:/Users/jordi/Desktop/Work/Kaggle/Word2Vec')

os.chdir('C:/Users/gutierj/Desktop/Programming/Kaggle/Word2Vec')


#To do:
#    tokenization
#    stemming
#    lemmatization
#    voting models (gpu?)
#    TF-IDF
#   Master's project stuff



#Tokenize and stemming

# Tokenize = to lower and split to words
# stemming = root of the word
# lemmatization: like stemming, but the lemma is an actual language word



#notebook of the tutorial
#https://nbviewer.jupyter.org/github/MatthieuBizien/Bag-popcorn/blob/master/Kaggle-Word2Vec.ipynb

# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
train = pd.read_csv("labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)




# Initialize the BeautifulSoup object on a single movie review     
example1 = BeautifulSoup(train["review"][0])  


# Use regular expressions to do a find-and-replace
letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      example1.get_text() )  # The text to search


lower_case = letters_only.lower()        # Convert to lower case
words = lower_case.split()               # Split into words



porter=PorterStemmer()

#porter.stem(word)



# Remove stop words from "words"
words = [w for w in words if not w in stopwords.words("english")]


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
    
    # Steps 2 and 3 are tokenizing - replace with function?
    #    word_tokenize(sentence)
    
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    
    
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   

    
    # 6. Stem
    
    meaningful_words = [porter.stem(w) for w in meaningful_words]
    
    # Insert 
    
    # 7. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))  
    

clean_review = review_to_words( train["review"][0] )



def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)



# Get the number of reviews based on the dataframe column size
num_reviews = train["review"].size


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



# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print(count, tag)
    
    
print("Training the random forest...")

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



