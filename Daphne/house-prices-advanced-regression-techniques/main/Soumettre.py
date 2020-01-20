########################################################################################################################
##################                                  IMPORT MODULES                                    ##################
########################################################################################################################
import numpy as np
import os
import pandas as pd
import category_encoders as ce
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import LocalOutlierFactor

########################################################################################################################
##################                                 LOADING THE DATA                                   ##################
########################################################################################################################
os.chdir(
        "C:/Users/daphn/Documents/Kaggle/house-prices-advanced-regression-techniques"
         )

# Read the training data
train = pd.read_csv("train.csv")

# Recover the data types of the variables
dict_schema ={}
for x in train.columns:
    if x != "SalePrice":
        dict_schema[x] = str(train[x].dtypes)

# Read the test data
test = pd.read_csv("test.csv")
test = test.fillna(0)
test = test.astype(dtype = dict_schema)

# Point estimate of the mean
mean_y = np.mean(train["SalePrice"])
median_y = np.median(train["SalePrice"])

# Apply the cleaning
from modules.preparationtrain import cleaning_function_train
train = cleaning_function_train(train)

from modules.preparationtest import cleaning_function_test
test = cleaning_function_test(test)

###########################################
#Target to predict in training
y_train = train["SalePrice"]
X_train = train.drop(["SalePrice", "Id"], axis = 1)

#y_test = test["SalePrice"]
X_test = test.loc[:, test.columns != 'Id']
test_id = test.loc[:, test.columns == 'Id']

#### TRAIN THE MODEL

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=0, ).fit(X_train)
X_train["Cluster"] = kmeans.labels_
total = pd.concat([X_train, y_train], axis = 1)

X_test["Cluster"] = kmeans.predict(X_test)
X_test = pd.concat([X_test, test_id], axis=1)
X_test["Id"].head(5)

# Prediction for each clusters
from sklearn.linear_model import LassoLarsCV

results = pd.DataFrame(columns = ["Id", "SalePrice"])
for cluster in range(0, kmeans.n_clusters):
    X_clus = total[total["Cluster"] == cluster].drop("SalePrice", axis = 1)
    y_clus = total[total["Cluster"] == cluster]
    mean_clus = np.mean(y_clus["SalePrice"])
    y_clus = y_clus["SalePrice"] - mean_clus
    model = LassoLarsCV(cv=3, max_iter=199999999).fit(X_clus, y_clus)

    X_test_clus = X_test[X_test["Cluster"] == cluster]
    X_test_id = X_test_clus.loc[:, X_test_clus .columns == 'Id']

    pred = model.predict(X_test_clus.drop("Id", axis = 1))
    X_test_id.loc[:, 1] = pred
    X_test_id.columns = ["Id", "SalePrice"]
    X_test_id["SalePrice"] += mean_clus
    results = pd.concat([results, X_test_id])

test_final = results
# Re-mean the prediction
#test_final["SalePrice"] += mean_y

test_final.head(5)
test_final.tail(5)
# Rename
submission = test_final

submission.to_csv(r'Submission_Daphne.csv', index = False)