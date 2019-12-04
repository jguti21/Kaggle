# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 19:56:55 2019

@author: Jordi
"""


##############################################################################
##############################################################################

import os
import seaborn as sns
import numpy as np
import pandas as pd

os.chdir('C:/Users/Jordi/Desktop/Economics/Kaggle/CrimeSF')


train = pd.read_csv("train.csv")

y = train["Category"].unique()

test = pd.read_csv("test.csv")

data_dict = {}
count = 1
for data in y:
    data_dict[data] = count
    count+=1
train["Category"] = train["Category"].replace(data_dict)

#Replacing the day of weeks
data_week_dict = {
    "Monday": 1,
    "Tuesday":2,
    "Wednesday":3,
    "Thursday":4,
    "Friday":5,
    "Saturday":6,
    "Sunday":7
}
train["DayOfWeek"] = train["DayOfWeek"].replace(data_week_dict)
test["DayOfWeek"] = test["DayOfWeek"].replace(data_week_dict)

#District
district = train["PdDistrict"].unique()
data_dict_district = {}
count = 1
for data in district:
    data_dict_district[data] = count
    count+=1 
train["PdDistrict"] = train["PdDistrict"].replace(data_dict_district)
test["PdDistrict"] = test["PdDistrict"].replace(data_dict_district)


train = train.drop(['Resolution'], axis = 1)


#knn on numeric columns

features = ["DayOfWeek", "PdDistrict",  "X", "Y"]
X_train = train[features]
y_train = train["Category"]
X_test = test[features]

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

from collections import OrderedDict
data_dict_new = OrderedDict(sorted(data_dict.items()))


result_dataframe = pd.DataFrame({
    "Id": test["Id"]
})
    
    
for key,value in data_dict_new.items():
    result_dataframe[key] = 0

count = 0
for item in predictions:
    for key,value in data_dict.items():
        if(value == item):
            result_dataframe[key][count] = 1
    count+=1
result_dataframe.to_csv("submission_knn.csv", index=False)


######################### Logistic

from sklearn.linear_model import LogisticRegression
lgr = LogisticRegression()
lgr.fit(X_train, y_train)
predictions = lgr.predict(X_test)


data_dict_new = OrderedDict(sorted(data_dict.items()))


result_dataframe = pd.DataFrame({
    "Id": test["Id"]
})

for key,value in data_dict_new.items():
    result_dataframe[key] = 0
count = 0
for item in predictions:
    for key,value in data_dict.items():
        if(value == item):
            result_dataframe[key][count] = 1
    count+=1
result_dataframe.to_csv("submission_logistic.csv", index=False) 






############### try NB


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
model = gnb.fit(X_train,y_train)


#predict

pred = gnb.predict(X_test)

data_dict_new = OrderedDict(sorted(data_dict.items()))

result_dataframe = pd.DataFrame({
    "Id": test["Id"]
})
for key,value in data_dict_new.items():
    result_dataframe[key] = 0
count = 0
for item in pred:
    for key,value in data_dict.items():
        if(value == item):
            result_dataframe[key][count] = 1
    count+=1
result_dataframe.to_csv("submission_NB.csv", index=False) 


##############################################################################
#Voting
from sklearn.ensemble import VotingClassifier

eclf1 = VotingClassifier(estimators=[('knn', knn), ('lgr', lgr), ('gnb', gnb)], voting='hard')

eclf1 = eclf1.fit(X_train, y_train)


pred_vote = eclf1.predict(X_test)


data_dict_new = OrderedDict(sorted(data_dict.items()))


result_dataframe = pd.DataFrame({
    "Id": test["Id"]
})

for key,value in data_dict_new.items():
    result_dataframe[key] = 0
count = 0
for item in pred_vote:
    for key,value in data_dict.items():
        if(value == item):
            result_dataframe[key][count] = 1
    count+=1
result_dataframe.to_csv("submission_vote.csv", index=False) 



##############################################################################


##############################################################################

