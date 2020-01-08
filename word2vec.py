# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 18:22:32 2020

@author: gutierj
"""

#Word2Vec

# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
import pandas as pd       
train = pd.read_csv("labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)

train.shape

print(train["review"][0])

#$ sudo pip install BeautifulSoup4

# Import BeautifulSoup into your workspace
from bs4 import BeautifulSoup             

# Initialize the BeautifulSoup object on a single movie review     
example1 = BeautifulSoup(train["review"][0])  

# Print the raw review and then the output of get_text(), for 
# comparison
print(train["review"][0])
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


