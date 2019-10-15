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
from scipy.stats import pearsonr

os.chdir('C:/Users/gutierj/Desktop/Programming/Kaggle/')


train = pd.read_csv('train.csv')



test = pd.read_csv('test.csv')
sample = pd.read_csv('sample_submission.csv')


#######################################
# clean train
#######################################

# Variables to exclude
exclusion = ['Neighborhood']
train = train[[c for c in train.columns
               if c not in exclusion]]

# Isolate in list variables of the same types
list_int = []
list_float = []
list_cat = []

for x in train.columns:
    if train[x].dtypes == "int64":
        list_int.append(x)
    elif train[x].dtypes == "float64":
        list_float.append(x)
    elif train[x].dtypes == "O":
        list_cat.append(x)

# Normalization of the floats
from sklearn.preprocessing import StandardScaler

train = train.fillna(0)

scaler = StandardScaler()
norm = scaler.fit_transform(train[train.columns.intersection(list_float)])
norm = pd.DataFrame(norm, columns=list_float)

### Train sample update
train.update(norm)

# Dealing with the categorical variables
"""
SET HTTPS_PROXY=https://ap-python-proxy:x2o7rCPYuN1JuV8H@p-gw.ecb.de:9090

pip install --trusted-host pypi.org category-encoders
"""


import category_encoders as ce
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import LabelEncoder

len(list_cat)
obs_lvl = {}
for var in list_cat:
    obs_lvl[var] = train[var].value_counts()
# Give an order to variables with Cond or Qual in the name
ordinal = [c for c in list_cat
           if "Condition" not in c and
           "Cond" in c or
           "Qual" in c]

cat_type = CategoricalDtype(categories=["NA", "Po", "Fa", "TA", "Gd", "Ex"],
                            ordered=True)

for var in ordinal:
    train[var] = train[var].astype(cat_type)

# Encode the ordinal variables
encoder = ce.ordinal.OrdinalEncoder(verbose=1, cols=ordinal)
train = encoder.fit_transform(train)

# LabelEncoder for the rest of the categorical variable
rest_var = set(list_cat).difference(ordinal)

encoder = LabelEncoder()
for var in rest_var:
    train[var] = encoder.fit_transform(train[var].astype(str))



#######################################
# clean train done
#######################################
    
    
    
#######################################
# clean test
#######################################

    
# Variables to exclude
exclusion = ['Neighborhood']
test = test[[c for c in test.columns
               if c not in exclusion]]

# Isolate in list variables of the same types
list_int = []
list_float = []
list_cat = []

for x in test.columns:
    if test[x].dtypes == "int64":
        list_int.append(x)
    elif test[x].dtypes == "float64":
        list_float.append(x)
    elif test[x].dtypes == "O":
        list_cat.append(x)

# Normalization of the floats

test = test.fillna(0)

scaler = StandardScaler()
norm = scaler.fit_transform(test[test.columns.intersection(list_float)])
norm = pd.DataFrame(norm, columns=list_float)

### Test sample update
test.update(norm)

# Dealing with the categorical variables

len(list_cat)
obs_lvl = {}
for var in list_cat:
    obs_lvl[var] = test[var].value_counts()
# Give an order to variables with Cond or Qual in the name
ordinal = [c for c in list_cat
           if "Condition" not in c and
           "Cond" in c or
           "Qual" in c]

cat_type = CategoricalDtype(categories=["NA", "Po", "Fa", "TA", "Gd", "Ex"],
                            ordered=True)

for var in ordinal:
    test[var] = test[var].astype(cat_type)

# Encode the ordinal variables
encoder = ce.ordinal.OrdinalEncoder(verbose=1, cols=ordinal)
test = encoder.fit_transform(test)

# LabelEncoder for the rest of the categorical variable
rest_var = set(list_cat).difference(ordinal)

encoder = LabelEncoder()
for var in rest_var:
    test[var] = encoder.fit_transform(test[var].astype(str))

    
    
#######################################
# clean test done
#######################################
    
    
 
"""   
#Select the X´s by removing Id and SalePrice
x_train = train.loc[:, train.columns != 'Id']
x_train = x_train.loc[:, x_train.columns != 'SalePrice']



#Select y train, SalePrice
y_train = train.loc[:, train.columns == 'SalePrice']

#pearsonr(train['LotArea'],train['SalePrice'])
#pearsonr(train['OverallQual'],train['SalePrice']).sort()

#sorted(train.corr(method='pearson')['SalePrice'])

#split and prepare test data

#these are the X's of the ID's we need to predict
test_x = test.loc[:, test.columns != 'Id']

#Id's for later's submission
test_id = test.loc[:, test.columns == 'Id']

# Initialize linear regression object
from sklearn import linear_model

regr  = linear_model.LinearRegression()


# Train the model



regr.fit(x_train, y_train)

#numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

#newdf = x_train.select_dtypes(include=numerics)



#regr.fit(newdf, y_train)

print ('Coefficients: \n', regr.coef_)

#plt.scatter(x_train, y_train, color='black')

#Ridge Regression
from sklearn.linear_model import Ridge
rreg = Ridge(alpha = 0.001)
rreg.fit(x_train, y_train)

print ('Coefficients: \n', rreg.coef_)

#Lasso
from sklearn.linear_model import Lasso

lreg = Lasso(alpha = 0.1)

lreg.fit(x_train, y_train)

print ('Coefficients: \n', lreg.coef_)  #many coefs set to 0


#convert test_x to dummies
test_xdummy = pd.get_dummies(test_x)

#fill NA's
test_xdummy.fillna(test_xdummy.mean(), inplace=True)


pred = lreg.predict(test_xdummy)


########### 
#Run a simple regression and prepare predictions
########### 


x_pred = train[['OverallQual', 'GrLivArea']]
test_pred = test_x[['OverallQual', 'GrLivArea']]

#Initializes OLS
ols  = linear_model.LinearRegression()

#Fits ols with selected variables from training and training target
ols.fit(x_pred, y_train)

#predicts, with OLS model, using the X's received for the test
pred = ols.predict(test_pred)


# paste Id and prediction for submission
test_id = test.loc[:, test.columns == 'Id']
test_id.loc[:,1] = pred

submission = test_id

submission.columns = ["Id", "SalePrice"]


submission.to_csv(r'Submission.csv', index = False)

"""

#################################
#Gradient Boosting regression
#################################
from sklearn.ensemble import GradientBoostingRegressor

#X's from test data used to predict
x_test = test.loc[:, test.columns != 'Id']

#Id for later's submission
test_id = test.loc[:, test.columns == 'Id']

#Target to predict in training
y_train = train.loc[:, train.columns == 'SalePrice']


#Given values: n_estimators=500, max_depth = 4, learning_rate = 0.01, loss="ls"
clf = GradientBoostingRegressor(n_estimators=600, max_depth = 5, learning_rate = 0.01, loss='ls', verbose=1)

#X's used to predict in training
x_train = train.loc[:, train.columns != 'Id']
x_train = x_train.loc[:, x_train.columns != 'SalePrice']



clf.fit(x_pred, y_train)

#predicts, with model, using the X's received for the test
pred = clf.predict(x_test)


# paste Id and prediction for submission

#add prediction to second column of test_id
test_id.loc[:,1] = pred

#rename
submission = test_id
submission.columns = ["Id", "SalePrice"]


submission.to_csv(r'Submission.csv', index = False)


"""

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

"""


"""
#AdaBossting
from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators=600, learning_rate = 0.01)

x_pred = train[['OverallQual', 'GrLivArea']]
abc.fit(x_pred, y_train )


#predicts, with model, using the X's received for the test
pred = abc.predict(test_pred)


# paste Id and prediction for submission

test_id.loc[:,1] = pred

submission = test_id

submission.columns = ["Id", "SalePrice"]


submission.to_csv(r'Submission.csv', index = False)

"""
