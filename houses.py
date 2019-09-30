# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 12:48:44 2019

@author: Jordi
"""

import sklearn
import seaborn
import numpy as np
import pandas as pd
import os

os.chdir('C:/Users/Jordi/Desktop/Economics/Kaggle/')


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sample_submission.csv')


train.head(6)

#Select the X´s by removing Id and SalePrice
x_train = train.loc[:, train.columns != 'Id']
x_train = x_train.loc[:, x_train.columns != 'SalePrice']

#NaNs
x_train.isnull().sum() == 0

#Select y train, SalePrice
y_train = train.loc[:, train.columns == 'SalePrice']


# Initialize linear regression object
from sklearn import datasets, linear_model

regr  = linear_model.LinearRegression()


# Train the model


#get dummies for categorical variables
x_traindummy = pd.get_dummies(x_train)

# fill missing values with mean column values
x_traindummy.fillna(x_traindummy.mean(), inplace=True)


regr.fit(x_traindummy, y_train)

#numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

#newdf = x_train.select_dtypes(include=numerics)



regr.fit(newdf, y_train)

print ('Coefficients: \n', regr.coef_)

#plt.scatter(x_train, y_train, color='black')

#Ridge Regression
from sklearn.linear_model import Ridge
rreg = Ridge(alpha = 0.001)
rreg.fit(x_traindummy, y_train)

print ('Coefficients: \n', rreg.coef_)

#Lasso
from sklearn.linear_model import Lasso

lreg = Lasso(alpha = 0.1)

lreg.fit(x_traindummy, y_train)

print ('Coefficients: \n', lreg.coef_)  #many coefs set to 0

# Training Logistic regression

from sklearn.model_selection import train_test_split

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2,random_state=0)


logistic_classifier = linear_model.LogisticRegression(C=100.0)
logistic_classifier.fit(x_train, y_train.values[:,1])

#Cross-Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(logistic_classifier, x_train, y_train.values[:,1], cv=10)
scores

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#Download again data - Alvaro style
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

os.chdir('C:/Users/Jordi/Google Drive/BGSE/2nd Term/CML/Kaggle/')

data_X = pd.read_csv('x_train.dat', header=0,)
data_y = pd.read_csv('y_train.dat', header=None, names=['Particle', 'Type'])
data_X.rename(columns={'Unnamed: 0':'Particle'}, inplace=True)
data_all = pd.merge(data_X, data_y, on='Particle', how='right')
df = pd.DataFrame(data_all)
df = df[df.Particle != 27610]
df = df[df.Particle != 14910]
df = df.drop(df.columns[[18,21,22,44,45,47,48,49,50,51]],axis=1) #Pandas starts counting at 0

data = np.array(df)
target = data[:,69]
X = data[:,1:68]

#Lasso
from sklearn.linear_model import Lasso

lreg = Lasso(alpha = 0.001)

lreg.fit(X, target)

print ('Coefficients: \n', lreg.coef_)  #many coefs set to 0

x_train, X_test, y_train, y_test= train_test_split(X, target, test_size=0.2,random_state=0)


logistic_binary = linear_model.LogisticRegression(C=100.0)


# Random Forest
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

random_forest = RandomForestClassifier(max_depth=100, n_estimators=20)
random_forest.fit(x_train, y_train)

scores_rf = cross_val_score(random_forest, x_train, y_train, cv=10)
scores_rf

# Kneighbors
from sklearn.neighbors import KNeighborsClassifier
k_neighbor_class = KNeighborsClassifier(10)
scores_k = cross_val_score(k_neighbor_class, x_train, y_train, cv=10)
scores_k

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
naive_class=GaussianNB()
scores_nv = cross_val_score(naive_class, X_clean, target, cv=10)
scores_nv


#Gradient Boosting regression
import numpy as np

import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor


#Given values: n_estimators=500, max_depth = 4, learning_rate = 0.01, loss="ls"
clf = GradientBoostingRegressor(n_estimators=600, max_depth = 5, learning_rate = 0.01, loss='ls', verbose=1)

clf.fit(x_train, y_train )

import sklearn
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_train, clf.predict(x_train), pos_label=1)
auc_gbr = sklearn.metrics.auc(fpr, tpr)
auc_gbr


#AdaBossting
from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators=600, learning_rate = 0.01)

abc.fit(x_train, y_train )

fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_train, abc.predict(x_train), pos_label=1)
auc_abc = sklearn.metrics.auc(fpr, tpr)
auc_abc
