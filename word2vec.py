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
from nltk.corpus import stopwords # Import the stop word list
from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

import num2words
from emot.emo_unicode import UNICODE_EMO, EMOTICONS

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
wordnet_lemmatizer = WordNetLemmatizer()

#porter.stem(word)

# Function for converting emoticons into word
def convert_emoticons(text):
    for emot in EMOTICONS:
        text = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), text)
    return text


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
    #   1a. replace numbers with words
    
    #letters_only = re.sub(r"(\d+)", lambda x: num2words.num2words(int(x.group(0))), review_text) 
    
    #   1b. replace emoticons with words
    
    #    convert_emoticons(review_text)
    
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
    
    
    #6b
#    meaningful_words = [wordnet_lemmatizer.lemmatize(w, pos="v") for w in meaningful_words]
    # Insert 
    
    # 7. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))  
    

clean_review = review_to_words( train["review"][0] )

print(clean_review)

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


#vectorizer = TfidfVectorizer()


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


############################################

# Modeling
###########################################

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier




# Initialize models

lgr = LogisticRegression()

knn = KNeighborsClassifier()

rf = RandomForestClassifier() 

abc = AdaBoostClassifier()

gbc = GradientBoostingClassifier()



################################################## Model fitting

# split in train and test

# For each model:
    # prepare a parameter grid
    # Grid Search CV
    # Obtain best parameters
    # Input model with best parameters into voting

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer

from joblib import dump, load

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    train_data_features, train["sentiment"], test_size=0.3, random_state=0)

#set up ROC as scorer
roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True,
                             needs_threshold=True)

############################## lgr

# Grid Search CV


# Create the parameter grid based on the results of random search 
param_grid_lgr = {
    'max_iter': [100, 80, 120],
    'solver': ['lbfgs', 'newton-cg', 'liblinear'],
    'multi_class': ['auto', 'ovr'],
}




# Create a based model
# Instantiate the grid search model
grid_search_lgr = GridSearchCV(estimator = lgr, param_grid = param_grid_lgr, 
                          cv = 5, n_jobs = -2, scoring='roc_auc')


grid_search_lgr.fit(X_train, y_train)

#lgr = LogisticRegression(grid_search_lgr.best_params_) 




dump(lgr, 'lgr.joblib')

#dump(grid_search_lgr, 'grid_search_lrg.joblib' )
#
#lgr = load('lgr.joblib')

"""
############################## knn

# Grid Search CV


# Create the parameter grid based on the results of random search 
param_grid_knn = {
    'n_neighbors': [5],
    'algorithm': ['auto'],
    'leaf_size': [30],
}


# Create a based model
# Instantiate the grid search model
grid_search_knn = GridSearchCV(estimator = knn, param_grid = param_grid_knn, 
                          cv = 5, n_jobs = -2, scoring='roc_auc')


grid_search_knn.fit(X_train, y_train)

knn =  KNeighborsClassifier(grid_search_knn.best_params_) 

grid_search_knn.best_score_ = 0.663629683998995

dump(knn, 'knn.joblib')

dump(grid_search_knn, 'grid_search_knn.joblib' )
"""

############################## rf

# Grid Search CV


# Create the parameter grid based on the results of random search 
param_grid_rf = {
    'max_depth': [80, 90],
    'min_samples_split': [8],
    'n_estimators': [100, 200]
}


# Create a based model
# Instantiate the grid search model
grid_search_rf = GridSearchCV(estimator = rf, param_grid = param_grid_rf, 
                          cv = 5, n_jobs = -2, scoring='roc_auc')


grid_search_rf.fit(X_train, y_train)

#rf = RandomForestClassifier(grid_search_rf.best_params_) 



dump(rf, 'rf.joblib')

#grid_search_rf.best_score_ = 1


#dump(grid_search_rf, 'grid_search_rf.joblib' )

############################## abc

# Grid Search CV


# Create the parameter grid based on the results of random search 
param_grid_abc = {
    'algorithm': ['SAMME', 'SAMME.R'],
    'learning_rate': [1, 1.2],
    'n_estimators': [50, 100]
}


# Create a based model
# Instantiate the grid search model
grid_search_abc = GridSearchCV(estimator = abc, param_grid = param_grid_abc, 
                          cv = 5, n_jobs = -2, scoring='roc_auc')


grid_search_abc.fit(X_train, y_train)

#abc = AdaBoostClassifier(grid_search_abc.best_params_) 



dump(abc, 'abc.joblib')



#dump(grid_search_abc, 'grid_search_abc.joblib' )

############################## gbc

# Grid Search CV


# Create the parameter grid based on the results of random search 
param_grid_gbc = {
    'subsample': [1],
    'learning_rate': [0.1],
    'n_estimators': [100, 120]
}


# Create a based model
# Instantiate the grid search model
grid_search_gbc = GridSearchCV(estimator = gbc, param_grid = param_grid_gbc, 
                          cv = 5, n_jobs = -2, scoring='roc_auc')


grid_search_gbc.fit(X_train, y_train)

#gbc = GradientBoostingClassifier(grid_search_gbc.best_params_) 



dump(gbc, 'gbc.joblib')


#dump(grid_search_gbc, 'grid_search_gbc.joblib' )




# Fit the grid search to the data
#grid_search.fit(train_data_features, train["sentiment"])

#Models

lgr = LogisticRegression(max_iter = 80) 

rf = RandomForestClassifier(max_depth = 90, min_samples_split = 8, n_estimators = 200) 

abc = AdaBoostClassifier() 

gbc = GradientBoostingClassifier(learning_rate = 0.1, n_estimators = 120, subsample = 1) 


#Voting


grid_search_lgr.best_score_ = 0.9226668788029424

grid_search_rf.best_score_ = 0.920650480793587

grid_search_abc.best_score_ = 0.9023515074908777

grid_search_gbc.best_score_ = 0.8987871540149742


weights=[grid_search_lgr.best_score_,
         grid_search_rf.best_score_, grid_search_gbc.best_score_, grid_search_abc.best_score_]

# Voting model
vc = VotingClassifier(estimators=[('lgr', lgr), ('rf', rf), ('abc', abc),
                                     ('gbc', gbc)], voting='soft', n_jobs = -2, weights = weights)

#n_jobs = -1 uses all CPU's, -2 uses all CPUs -1

     
vc = vc.fit(train_data_features, train["sentiment"] )



# Read the test data
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", \
                   quoting=3 )


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
#result = forest.predict(test_data_features)


result = vc.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "voting.csv", index=False, quoting=3 )    