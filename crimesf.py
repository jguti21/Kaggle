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

os.chdir('C:/Users/gutierj/Desktop/Programming/Kaggle/CrimeSF')


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


############ Process and get weather data
"""
train['Date']=pd.to_datetime(train['Dates'].dt.strftime('%Y-%m-%d'),format='%Y-%m-%d')
test['Date']=pd.to_datetime(test['Dates'].dt.strftime('%Y-%m-%d'),format='%Y-%m-%d')

# Preprocessing weather data
weather_data = pd.read_csv("weather.csv")
weather_data["Date"]= weather_data["Date"].str.replace(" ", "")
weather_data["Date"] = pd.to_datetime(weather_data["Date"],format='%Y-%m-%d')

weather_data.columns = ['t_max','t_avg','t_min','dew_max','dew_avg','dew_min','hum_max',
                        'hum_avg','hum_min','wind_max','wind_avg','wind_min','pres_max','pres_avg','pres_min','percip','Date']



train=pd.merge(train, weather_data, on ="Date", how="left")
test=pd.merge(test, weather_data, on ="Date", how="left")

"""

#####
features = ["DayOfWeek", "PdDistrict",  "X", "Y"]
X_train = train[features]
y_train = train["Category"]
X_test = test[features]

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

predictions_knn_proba = knn.predict(X_test)

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

predictions_log_proba = lgr.predict_proba(X_test)

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

predictions_nb_proba = gnb.predict_proba(X_test)

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


#Average of probability predictions of logarithmic and nb

#concatenates both dataframes (rbind), groups by row, and computes the mean 
p = pd.concat([pd.DataFrame(predictions_log_proba), pd.DataFrame(predictions_nb_proba)]).groupby(level=0).mean()

data_dict["Key"]
print(data_dict.keys())

p.columns = data_dict.keys()

p['Id'] = p.index

p_id = p['Id']

p.drop('Id', axis = 1, inplace = True)

p.insert(0, 'Id', p_id)

p.to_csv("submission_avg.csv", index = True)

#https://www.dataquest.io/blog/introduction-to-ensembles/

