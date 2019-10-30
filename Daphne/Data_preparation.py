########################################################################################################################
##################                                  IMPORT MODULES                                    ##################
########################################################################################################################
import numpy as np
import os
import pandas as pd

########################################################################################################################
##################                                 LOADING THE DATA                                   ##################
########################################################################################################################
os.chdir(
        "C:/Users/daphn/Documents/Kaggle/house-prices-advanced-regression-techniques"
         )

train = pd.read_csv("train.csv")

exclusion = ['Neighborhood', "Id", "Alley"]
train = train[[c for c in train.columns
               if c not in exclusion]]

########################################################################################################################
##################                           VARIABLE IDENTIFICATION                                   ##################
########################################################################################################################

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

########################################################################################################################
##################                           FLOATS                                   ##################
########################################################################################################################
# Normalization of the floats
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
norm = scaler.fit_transform(train[train.columns.intersection(list_float)])
norm = pd.DataFrame(norm, columns=list_float)

### Train sample update
train.update(norm)

########################################################################################################################
##################                           CATEGORICALS                                   ##################
########################################################################################################################
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


# Standardization of the categoricals
scaler = StandardScaler()
norm = scaler.fit_transform(train[train.columns.intersection(list_cat)])
norm = pd.DataFrame(norm, columns = list_cat)
### Train sample update
train.update(norm)

########################################################################################################################
##################                           INT                                   ##################
########################################################################################################################

# Standardization of the integer variables
scaler = StandardScaler()
list_int.remove("SalePrice")
norm = scaler.fit_transform(train[train.columns.intersection(list_int)])
norm = pd.DataFrame(norm, columns = list_int)
### Train sample update
train.update(norm)

########################################################################################################################
##################                           NULLS                                   ##################
########################################################################################################################
Nas = train.isna().sum().where(lambda x: x > 0).dropna()
Nas_var = list(Nas.index)
# Float var. Set to
train = train.fillna(0)

########################################################################################################################
##################                        CREATION OF MULTITUDE OF VAR                                ##################
########################################################################################################################
columns_to_user = list(train.columns)
columns_to_user.remove("SalePrice")
# SQUARE OF EVERYTHING
for col in columns_to_user:
    train[col + "_sqrt"] = train[col] ** 2

# Cube of everything
for col in columns_to_user:
    train[col + "_cube"] = train[col] ** 3

########################################################################################################################
##################                          DE-MEANING THE SALEPRICE                                  ##################
########################################################################################################################
train["SalePrice"] = train["SalePrice"] - np.mean(train["SalePrice"])

########################################################################################################################
##################                           OUTLIERS DETECTION                                    ##################
########################################################################################################################
from sklearn.neighbors import LocalOutlierFactor

neighbors = LocalOutlierFactor().fit_predict(train)
train["outlier"] = neighbors
train.head(5)
train = train[train["outlier"] == 1] # 1 are inlier
########################################################################################################################
##################                           SAVE                                   ##################
########################################################################################################################

train.to_csv("final.csv")