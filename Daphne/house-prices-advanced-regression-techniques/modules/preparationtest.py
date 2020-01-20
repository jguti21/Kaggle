
import numpy as np
import os
import pandas as pd
import category_encoders as ce
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import LocalOutlierFactor

def cleaning_function_test(train):
    exclusion = ['Neighborhood', "Alley"]
    train = train[[c for c in train.columns
                   if c not in exclusion]]
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

############################# CATEGORICAL ##############################################################################
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

    # Combination of variables
    # import itertools
    # list_relevant_features = list(train.columns)
    # list_relevant_features.remove("Id")
    # cc = list(itertools.combinations(list_relevant_features, 2))
    # df = pd.concat([train[c[1]].sub(train[c[0]]) for c in cc], axis=1, keys=cc)
    # train = pd.concat([train, df], axis = 1)


############################# FLOATS ###################################################################################
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

    # log of eveyrhing
    for col in columns_to_user:
        train[col + "_log"] = np.log(train[col] + 1)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    norm = scaler.fit_transform(train[train.columns.intersection(list_float)])
    norm = pd.DataFrame(norm, columns=list_float)

    ### Train sample update
    train.update(norm)

    # Standardization of the categoricals
    scaler = StandardScaler()
    norm = scaler.fit_transform(train[train.columns.intersection(list_cat)])
    norm = pd.DataFrame(norm, columns = list_cat)
    ### Train sample update
    train.update(norm)

############################# INTEGERS #################################################################################
    # Standardization of the integer variables
    scaler = StandardScaler()

    norm = scaler.fit_transform(train[train.columns.intersection(list_int)])
    norm = pd.DataFrame(norm, columns = list_int)
    ### Train sample update
    train.update(norm)

##############################  ALL  ###################################################################################
    # Replace NA by 0 which is by construction our mean too
    Nas = train.isna().sum().where(lambda x: x > 0).dropna()
    Nas_var = list(Nas.index)
    # Float var. Set to
    train = train.fillna(0)

    return train
