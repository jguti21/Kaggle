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
import matplotlib.pyplot as plt


def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes


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

train = train.fillna(train.mean())

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
    
    
####################################### 
    # final cleaning
#######################################    
    #x_train = train[['OverallQual', 'GrLivArea']]
#x_test = test[['OverallQual', 'GrLivArea']]

#X's from test data used to predict
x_test = test.loc[:, test.columns != 'Id']
#x_test = np.log(x_test+1)

#x_test = x_test.fillna(0)

#Id for later's submission
test_id = test.loc[:, test.columns == 'Id']

#Target to predict in training
y_train = train.loc[:, train.columns == 'SalePrice']
y_train = np.ravel(y_train)

#X's used to predict in training
x_train = train.loc[:, train.columns != 'Id']
x_train = x_train.loc[:, x_train.columns != 'SalePrice']
#x_train = np.log(x_train+1)

#x_train = x_train.fillna(0)



#################################
#Gradient Boosting regression
#################################



from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
#from math import sqrt

mean_squared_error = make_scorer(mean_squared_error)
#rms = sqrt(mean_squared_error(y_actual, y_predicted))


"""
#Given values: n_estimators=500, max_depth = 4, learning_rate = 0.01, loss="ls"
clf = GradientBoostingRegressor(n_estimators=600, max_depth = 5, learning_rate = 0.01, loss='ls', verbose=1)
clf = GradientBoostingRegressor(n_estimators=700, max_depth = 6, learning_rate = 0.1, loss='ls', verbose=1)

#X's used to predict in training
x_train = train.loc[:, train.columns != 'Id']
x_train = x_train.loc[:, x_train.columns != 'SalePrice']

clf.fit(x_train, y_train)
"""

def model_gradient_boosting_tree(Xtrain,Xtest,ytrain):
    
    X_train = Xtrain
    y_train = ytrain 
    gbr = GradientBoostingRegressor(random_state=0, loss='ls')
    param_grid = {
        'n_estimators': [420, 450, 470],
       # 'max_features': [20,15],
	    'max_depth': [0.3, 0.5, 0.6],
        'learning_rate': [0.009, 0.01, 0.015],
       'subsample': [0.95, 0.9, 0.85]
    }
    model = GridSearchCV(estimator=gbr, param_grid=param_grid, n_jobs=1, cv=10, scoring=mean_squared_error)
    model.fit(X_train, y_train)
    print('Gradient boosted tree regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(model.best_score_)

    y_pred = model.predict(Xtest)
    return y_pred, model.best_score_

#pred = model_gradient_boosting_tree(x_train,x_test,y_train)

clf = GradientBoostingRegressor(n_estimators=470, max_depth = 0.3, learning_rate = 0.015, loss='ls', subsample = 0.85)

clf = clf.fit(x_train, y_train)

pred = clf.predict(x_test)

"""
f, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim((0, max([max(pred[0]), max(y_train)]) + 10000))
ax.set_ylim((0, max([max(pred[0]), max(y_train)]) + 10000))
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.scatter(y_train, pred[0], c=".3")
add_identity(ax, color='r', ls='--')
plt.show()
"""

#Best CV Score:
#6315501866.2019005

#add prediction to second column of test_id
test_id = test.loc[:, test.columns == 'Id']
test_id.loc[:,1] = predx

#test_id.loc[:,1] = preds[0]
#rename
submission = test_id
submission.columns = ["Id", "SalePrice"]


submission.to_csv(r'Submission.csv', index = False)

#################################
# XGB
#################################
"""
SET HTTPS_PROXY=https://ap-python-proxy:x2o7rCPYuN1JuV8H@p-gw.ecb.de:9090

pip install --trusted-host pypi.org xgboost
"""


import xgboost as xgb
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=123)


xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg.fit(x_train,y_train)


preds = xg_reg.predict(x_test)

#########


params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}


data_dmatrix = xgb.DMatrix(data=x_train,label=y_train)

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)

print(cv_results, verbose = True)

predx = 0.5*pred + 0.5*preds


#########################
# optimize XGB
########################

from scipy import stats
from sklearn.model_selection import RandomizedSearchCV, KFold

xgb = xgb.XGBRegressor(objective ='reg:linear')
param_dist = {'n_estimators': stats.randint(150, 500),
              'learning_rate': stats.uniform(0.01, 0.07),
              'subsample': stats.uniform(0.3, 0.7),
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'colsample_bytree': stats.uniform(0.5, 0.45),
              'min_child_weight': [1, 2, 3]
             }
clf = RandomizedSearchCV(xgb, param_distributions = param_dist, n_iter = 25, scoring = 'f1', error_score = 0, verbose = 3, n_jobs = -1)

numFolds = 5
folds = KFold(n_splits = numFolds, shuffle = True)

estimators = []
results = np.zeros(len(x_train))
score = 0.0
for train_index, test_index in folds.split(x_train):
    X_train, X_test = x_train.iloc[train_index,:], x_train.iloc[test_index,:]
    Y_train, Y_test = y_train.iloc[train_index].values.ravel(), y_train.iloc[test_index].values.ravel()
    clf.fit(X_train, Y_train)

    estimators.append(clf.best_estimator_)
    results[test_index] = clf.predict(X_test)
    score += f1_score(y_test, results[test_index])
score /= numFolds

#################################
# Random forest
#################################

# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

random_forest = RandomForestClassifier(max_depth=100, n_estimators=20)
random_forest.fit(x_train, y_train)

scores_rf = cross_val_score(random_forest, x_train, y_train, cv=10)
scores_rf

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
