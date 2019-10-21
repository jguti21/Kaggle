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

train = pd.read_csv("train.csv")
dict_schema ={}
for x in train.columns:
    if x != "SalePrice":
        dict_schema[x] = str(train[x].dtypes)

test = pd.read_csv("test.csv")
test = test.fillna(0)
test = test.astype(dtype = dict_schema)

mean_y = np.mean(train["SalePrice"])

########################################################################################################################
##################                                CLEANING FUNCTION                                   ##################
########################################################################################################################
def cleaning_function(train):

    # Isolate in list variables of the same types
    list_int = []
    list_float = []
    list_cat = []

    for x in train.columns:
        if train[x].dtypes == "int64" and x != "Id":
            list_int.append(x)
        elif train[x].dtypes == "float64":
            list_float.append(x)
        elif train[x].dtypes == "O":
            list_cat.append(x)

    # Normalization of the floats
    columns_to_user = list(list_float)

    try:
        columns_to_user.remove("SalePrice")
    except:
        columns_to_user = columns_to_user

    # SQUARE OF EVERYTHING
    for col in columns_to_user:
        train[col + "_sqrt"] = train[col] ** 2

    # Cube of everything
    for col in columns_to_user:
        train[col + "_cube"] = train[col] ** 3

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    norm = scaler.fit_transform(train[train.columns.intersection(list_float)])
    norm = pd.DataFrame(norm, columns=list_float)

    ### Train sample update
    train.update(norm)

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


    # Standardization of the categoricals
    scaler = StandardScaler()
    norm = scaler.fit_transform(train[train.columns.intersection(list_cat)])
    norm = pd.DataFrame(norm, columns = list_cat)
    ### Train sample update
    train.update(norm)

    # Standardization of the integer variables
    scaler = StandardScaler()
    try:
        list_int.remove("SalePrice")
    except:
        pass
    norm = scaler.fit_transform(train[train.columns.intersection(list_int)])
    norm = pd.DataFrame(norm, columns = list_int)
    ### Train sample update
    train.update(norm)

    Nas = train.isna().sum().where(lambda x: x > 0).dropna()
    Nas_var = list(Nas.index)
    # Float var. Set to
    train = train.fillna(0)


    # De-mean y
    try:
        train["SalePrice"] = train["SalePrice"] - np.mean(train["SalePrice"])
    except:
        pass

    neighbors = LocalOutlierFactor().fit_predict(train)
    train["outlier"] = neighbors
    #train = train[train["outlier"] == 1] # 1 are inlier

    # Exclude some variable
    exclusion = ['Neighborhood', "Alley"]
    train = train[[c for c in train.columns
                   if c not in exclusion]]

    return train


train = cleaning_function(train)
test = cleaning_function(test)

#Target to predict in training
y_train = train.loc[:, train.columns == 'SalePrice']

#X's used to predict in training
x_train = train.loc[:, train.columns != 'Id']
x_train = x_train.loc[:, x_train.columns != 'SalePrice']

#X's from test data used to predict
x_test = test.loc[:, test.columns != 'Id']
#Id for later's submission
test_id = test.loc[:, test.columns == 'Id']


########################################################################################################################
##################                             LEAST ANGLE REGRESSION                                 ##################
########################################################################################################################
from sklearn.linear_model import LassoLarsCV

model = LassoLarsCV(cv=10, max_iter=199999999999).fit(x_train, y_train)
CV_pred = model.predict(x_test)

# paste Id and prediction for submission
#add prediction to second column of test_id
test_id.loc[:,1] = CV_pred

# Rename
submission = test_id
submission.columns = ["Id", "SalePrice"]

# Re-mean the prediction
submission["SalePrice"] += mean_y

submission.to_csv(r'Submission_Daphne.csv', index = False)