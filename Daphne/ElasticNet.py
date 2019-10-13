# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 21:44:49 2019

@author: daphne
"""
import numpy as np
import os
import pandas as pd

os.chdir(
    "C:/Users/daphn/Documents/Kaggle/house-prices-advanced-regression-techniques"
)

train = pd.read_csv("train.csv")

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

scaler = StandardScaler()
norm = scaler.fit_transform(train[train.columns.intersection(list_float)])
norm = pd.DataFrame(norm, columns=list_float)

### Train sample update
train.update(norm)

# Dealing with the categorical variables
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

# Checking the NAs
Nas = train.isna().sum().where(lambda x: x > 0).dropna()
Nas_var = list(Nas.index)
# Float var. Set to 0
final = train.fillna(0)

#### SAMPLE
X = final.drop(columns="SalePrice")
y = final["SalePrice"].copy()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.8,
                                                    random_state=42)

################### LASSO REGRESSION #####################################################
from sklearn.linear_model import Lasso

regr_lasso = Lasso(fit_intercept=False)
regr_lasso.fit(X_train, y_train)

print(regr_lasso.coef_)

pred_lasso = regr_lasso.predict(X_test)

RMSE = np.sqrt(np.mean(pred_lasso - y_test) ** 2)
R2_lasso = regr_lasso.score(X_test, y_test)

# cross-validation
from sklearn.linear_model import LassoCV
dictR2lasso_cv = {}
for x in range(1, 16):
    """
    Loop to try different k in the k-fold.
    k=1: no cross-validation. 
    """
    if x == 1:
        regr_lasso = Lasso(fit_intercept = False, # better without the intercept
                           max_iter=10000)
        regr_lasso.fit(X_train, y_train)
        pred_lasso = regr_lasso.predict(X_test)
        R2_lasso = regr_lasso.score(X_test, y_test)
        dictR2lasso_cv[x] = (R2_lasso,
                             regr_lasso.alpha)
    else:
        regr_lasso_cv = LassoCV(cv = x, fit_intercept = True,
                                normalize= False,
                                precompute= 'auto',
                                max_iter= 10000)
        regr_lasso_cv.fit(X_train, y_train)
        pred_lasso_cv = regr_lasso_cv.predict(X_test)
        R2_lasso_cv = regr_lasso_cv.score(X_test, y_test)
        dictR2lasso_cv[x] = (R2_lasso_cv,
                             regr_lasso_cv.alpha_)

dictR2lasso_cv

# TO BE CHECKED
# sklearn.linear_model.lasso_path and sklearn.linear_model.enet_path.

######################
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path, enet_path

# Compute paths
eps = 5e-3  # the smaller it is the longer is the path

print("Computing regularization path using the lasso...")
alphas_lasso, coefs_lasso, _ = lasso_path(X_train, y_train, eps, fit_intercept=False)

print("Computing regularization path using the positive lasso...")
alphas_positive_lasso, coefs_positive_lasso, _ = lasso_path(
    X_train, y_train, eps, positive=True, fit_intercept=False)

print("Computing regularization path using the elastic net...")
alphas_enet, coefs_enet, _ = enet_path(
    X_train, y_train, eps=eps, l1_ratio=0.8, fit_intercept=False)

print("Computing regularization path using the positive elastic net...")
alphas_positive_enet, coefs_positive_enet, _ = enet_path(
    X_train, y_train, eps=eps, l1_ratio=0.8, positive=True, fit_intercept=False)

# Display results

plt.figure(1)
colors = cycle(['b', 'r', 'g', 'c', 'k'])
neg_log_alphas_lasso = -np.log10(alphas_lasso)
neg_log_alphas_enet = -np.log10(alphas_enet)
for coef_l, coef_e, c in zip(coefs_lasso, coefs_enet, colors):
    l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
    l2 = plt.plot(neg_log_alphas_enet, coef_e, linestyle='--', c=c)

plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Lasso and Elastic-Net Paths')
plt.legend((l1[-1], l2[-1]), ('Lasso', 'Elastic-Net'), loc='lower left')
plt.axis('tight')


plt.figure(2)
neg_log_alphas_positive_lasso = -np.log10(alphas_positive_lasso)
for coef_l, coef_pl, c in zip(coefs_lasso, coefs_positive_lasso, colors):
    l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
    l2 = plt.plot(neg_log_alphas_positive_lasso, coef_pl, linestyle='--', c=c)

plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Lasso and positive Lasso')
plt.legend((l1[-1], l2[-1]), ('Lasso', 'positive Lasso'), loc='lower left')
plt.axis('tight')


plt.figure(3)
neg_log_alphas_positive_enet = -np.log10(alphas_positive_enet)
for (coef_e, coef_pe, c) in zip(coefs_enet, coefs_positive_enet, colors):
    l1 = plt.plot(neg_log_alphas_enet, coef_e, c=c)
    l2 = plt.plot(neg_log_alphas_positive_enet, coef_pe, linestyle='--', c=c)

plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Elastic-Net and positive Elastic-Net')
plt.legend((l1[-1], l2[-1]), ('Elastic-Net', 'positive Elastic-Net'),
           loc='lower left')
plt.axis('tight')
plt.show()




### ElasticNet
from sklearn.linear_model import ElasticNet

regr_En = ElasticNet(fit_intercept = False, max_iter=100000)
regr_En.fit(X_train, y_train)
print(regr_En.coef_)
pred_En = regr_En.predict(X_test)

regr_En.alpha

RMSE = np.sqrt(np.mean(pred_En - y_test) ** 2)
R2_En = regr_En.score(X_test, y_test)

# CV
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GridSearchCV
regr_En_cv = ElasticNetCV(cv = 3, fit_intercept = False,
                          normalize= False)

parametersGrid = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                  "l1_ratio": np.arange(0.001, 1.0, 0.1)}

grid = GridSearchCV(regr_En_cv, parametersGrid, scoring='r2', cv= 3
                    
grid.fit(X_train, y_train)
pred_En_cv = regr_En_cv.predict(X_test)
R2_En_cv = regr_En_cv.score(X_test, y_test)
regr_En_cv.alpha_

dictR2EN_cv



R2_En > R2_lasso # En is slightly better

import matplotlib.pyplot as plt
plt.plot(regr_En.coef_,alpha=0.7, linestyle='none',marker='*',markersize=5,
         color='red',label=r'ElasticNet; $\alpha = 1$',zorder=7) # zorder for ordering the markers

plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.show()