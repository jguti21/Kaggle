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

#os.chdir('C:/Users/gutierj/Desktop/Programming/Kaggle/CrimeSF')

os.chdir('C:/Users/jordi/Desktop/Work/Kaggle/CrimeSF')

train = pd.read_csv("train.csv", parse_dates=['Dates'])

y = train["Category"].unique()

test = pd.read_csv("test.csv", parse_dates=['Dates'])

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

train['Date']=pd.to_datetime(train['Dates'].dt.strftime('%Y-%m-%d'),format='%Y-%m-%d')
test['Date']=pd.to_datetime(test['Dates'].dt.strftime('%Y-%m-%d'),format='%Y-%m-%d')

# Preprocessing weather data
weather_data = pd.read_csv("weather.csv", sep=';')

weather_data.columns = ['t_max','t_avg','t_min','dew_max','dew_avg','dew_min','hum_max',
                        'hum_avg','hum_min','wind_max','wind_avg','wind_min','pres_max','pres_avg','pres_min','percip','Date']


#weather_data["Date"]= weather_data["Date"].str.replace(" ", "")
weather_data["Date"] = pd.to_datetime(weather_data["Date"],format='%d/%m/%Y')



######################### PC on all weather except Date

from sklearn.preprocessing import StandardScaler

# Separating out the features
weather_pc = weather_data.loc[:, weather_data.columns != 'Date']

# Standardizing the features
weather_pc = StandardScaler().fit_transform(weather_pc)


from sklearn.decomposition import PCA
pca = PCA(.95)

principalComponents = pca.fit_transform(weather_pc)

pca.explained_variance_ratio_




principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'])


### re-merge to weather
weather_data = pd.concat([weather_data.reset_index(drop=True), principalDf], axis=1)


#merge back to train
train=pd.merge(train, weather_data, on ="Date", how="left")
test=pd.merge(test, weather_data, on ="Date", how="left")



######################### Feature selection



features = ["DayOfWeek", "PdDistrict",  "X", "Y", "PC1", "PC2", "PC3"]
X_train = train[features]
y_train = train["Category"]
X_test = test[features]


######################### KNN


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

predictions_knn_proba = knn.predict(X_test)

from collections import OrderedDict

"""
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
"""

######################### Logistic

from sklearn.linear_model import LogisticRegression
lgr = LogisticRegression()
lgr.fit(X_train, y_train)
predictions = lgr.predict(X_test)

predictions_log_proba = lgr.predict_proba(X_test)

data_dict_new = OrderedDict(sorted(data_dict.items()))


"""
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
"""





###############  NB


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train,y_train)


#predict

pred = gnb.predict(X_test)

predictions_nb_proba_train = gnb.predict_proba(X_train)
predictions_nb_proba = gnb.predict_proba(X_test)

data_dict_new = OrderedDict(sorted(data_dict.items()))


"""
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
"""



##############################################################################
#Voting
from sklearn.ensemble import VotingClassifier

eclf1 = VotingClassifier(estimators=[('knn', knn), ('lgr', lgr), ('gnb', gnb)], voting='hard')

eclf1 = eclf1.fit(X_train, y_train)


pred_vote = eclf1.predict(X_test)


data_dict_new = OrderedDict(sorted(data_dict.items()))


"""
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
"""




###############  

#AdaBoosting
from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier()
abc.fit(X_train, y_train)

predictions_abc_proba_train = abc.predict_proba(X_train)
predictions_abc_proba = abc.predict_proba(X_test)



###############  

# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)

predictions_rf_proba_train = random_forest.predict_proba(X_train)

predictions_rf_proba = random_forest.predict_proba(X_test)

#scores_rf = cross_val_score(random_forest, X_train, y_train, cv=10)
#scores_rf

##############################################################################


##############################################################################


#Average of probability predictions of logarithmic and nb

#concatenates both dataframes (rbind), groups by row, and computes the mean 
#p = pd.concat([pd.DataFrame(predictions_log_proba), pd.DataFrame(predictions_nb_proba)]).groupby(level=0).mean()


p = pd.concat([pd.DataFrame(predictions_log_proba), pd.DataFrame(predictions_nb_proba),
               pd.DataFrame(predictions_abc_proba), pd.DataFrame(predictions_rf_proba)]).groupby(level=0).mean()


#data_dict["Key"]
print(data_dict.keys())

p.columns = data_dict.keys()

p['Id'] = p.index

p_id = p['Id']

p.drop('Id', axis = 1, inplace = True)

p.insert(0, 'Id', p_id)

p.to_csv("submission_avg.csv", index = False)

#https://www.dataquest.io/blog/introduction-to-ensembles/
#https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/


##############################################################################
# New Stacking: https://towardsdatascience.com/automate-stacking-in-python-fc3e7834772e

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier

from vecstack import stacking

X_train = train[features]
y_train = train["Category"]
X_test = test[features]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


models = [
        AdaBoostClassifier(),
        RandomForestClassifier(),
        GaussianNB()
]


S_train, S_test = stacking(models, X_train, y_train, X_test, regression=False, mode='oof_pred_bag',
                           
                           needs_proba=True, save_dir=None, metric=accuracy_score,  n_folds=4,
         
                           stratified=True,  shuffle=True,  random_state=0,  verbose=0)


model = XGBClassifier()
    
model = LogisticRegression(max_iter=100)

model = model.fit(S_train, y_train)
y_pred = model.predict_proba(S_test)


p.columns = data_dict.keys()

p['Id'] = p.index

p_id = p['Id']

p.drop('Id', axis = 1, inplace = True)

p.insert(0, 'Id', p_id)

p.to_csv("submission_stack.csv", index = False)


#print('Final prediction score: [%.8f]' % accuracy_score(y_test, y_pred))

"""

###############  
#Stacking

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier


abc = AdaBoostClassifier()
random_forest = RandomForestClassifier()
gnb = GaussianNB()
lgr = LogisticRegression()


from sklearn.model_selection import train_test_split, StratifiedKFold



features = ["DayOfWeek", "PdDistrict",  "X", "Y", "PC1", "PC2", "PC3"]
X_train = train[features]
y_train = train["Category"]
X_test = test[features]


X_train, X_test, y_train, y_test = train_test_split(train[features], train["Category"], test_size=0.33, random_state=42)

"""
def Stacking(model,train,y,test,n_fold):
    folds=StratifiedKFold(n_splits=n_fold,random_state=1)
    test_pred=np.empty((test.shape[0],1),float)
    train_pred=np.empty((0,1),float)
  
    for train_indices,val_indices in folds.split(train,y.values):
      x_train,x_val=train.iloc[train_indices],train.iloc[val_indices]
      y_train,y_val=y.iloc[train_indices],y.iloc[val_indices]

      model.fit(X=x_train,y=y_train)
      train_pred=np.append(train_pred,model.predict(x_val))
      test_pred=np.append(test_pred,model.predict(test))
    return test_pred.reshape(-1,1),train_pred
"""

def Stacking(model,train,y,test,n_fold):
    folds=StratifiedKFold(n_splits=n_fold,random_state=1)
    test_pred=np.empty((test.shape[0],1),float)
    train_pred=np.empty((0,1),float)
  
    for train_indices,val_indices in folds.split(train,y.values):
      x_train,x_val=train.iloc[train_indices],train.iloc[val_indices]
      y_train,y_val=y.iloc[train_indices],y.iloc[val_indices]

      model.fit(X=x_train,y=y_train)
      train_pred=np.append(train_pred,model.predict_proba(x_val))
      test_pred=np.append(test_pred,model.predict_proba(test))
    return test_pred.reshape(-1,1),train_pred



model1 = abc

test_pred1 ,train_pred1=Stacking(model=model1,n_fold=10, train=X_train,test=X_test,y=y_train)

train_pred1=pd.DataFrame(train_pred1)
test_pred1=pd.DataFrame(test_pred1)

model2 = RandomForestClassifier()

test_pred2 ,train_pred2=Stacking(model=model2,n_fold=10,train=X_train,test=X_test,y=y_train)

train_pred2=pd.DataFrame(train_pred2)
test_pred2=pd.DataFrame(test_pred2)


model3 = GaussianNB()

test_pred3, train_pred3=Stacking(model=model3,n_fold=10,train=X_train,test=X_test,y=y_train)


train_pred3=pd.DataFrame(train_pred3)
test_pred3=pd.DataFrame(test_pred3)


df = pd.concat([train_pred1, train_pred2, train_pred3], axis=1)
df_test = pd.concat([test_pred1, test_pred2, test_pred3], axis=1)

model = LogisticRegression(random_state=1)
model.fit(df,y_train)
model.score(df_test, y_test)

#re-set main df's
X_train = train[features]
y_train = train["Category"]
X_test = test[features]


model.predict_proba(X_test)


df_proba = pd.concat([pd.DataFrame(predictions_nb_proba), 
                      pd.DataFrame(predictions_abc_proba), pd.DataFrame(predictions_rf_proba)], axis = 1)
    
df_proba_train = pd.concat([pd.DataFrame(predictions_nb_proba_train), 
                      pd.DataFrame(predictions_abc_proba_train), pd.DataFrame(predictions_rf_proba_train)], axis = 1)
    
model.fit(df_proba, y_train)
#model.fit(X_train, y_train)
model_proba = model.predict_proba(df_proba)

"""
#https://wiki.openstreetmap.org/wiki/Downloading_data

#https://towardsdatascience.com/a-guide-to-ensemble-learning-d3686c9bed9a