# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 16:32:42 2019

@author: daphn
"""

## IMPORT

import os

os.chdir(
        "C:/Users/daphn/Documents/Kaggle/house-prices-advanced-regression-techniques"
         )

os.listdir()

import pandas as pd

train = pd.read_csv("train.csv")

train.head(5)

train.columns
len(train.columns)

type(train.dtypes)

# Isolate in list variables of the same types
list_int =[]
list_float = []
list_cat = []

for x in train.columns:
    if train[x].dtypes == "int64":
        list_int.append(x)
    elif train[x].dtypes == "float64":
        list_float.append(x)
    elif train[x].dtypes == "O":
        list_cat.append(x)
        
"""
var = train.columns.to_series().groupby(train.dtypes).groups

var.values()
var.keys()
"""

# Create distribution and regression plots for all the variable on integer
# type agaisnt the Sale Price
import seaborn as sn
sn.set_style("darkgrid")
for x in list_int:
    save_plots = sn.jointplot(x = x, 
               y = "SalePrice",
               data = train, kind="reg").savefig("./plots/" + x + "vsSalePrives.png")


# NORMAILITY TESTS FOR PRICE
# QQ Plot
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
qqplot(train["SalePrice"], line='s')
pyplot.show()
	
# Shapiro-Wilk Test
from scipy.stats import shapiro
stat, p = shapiro(train["SalePrice"])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Sample looks Gaussian (fail to reject H0)')
else:
	print('Sample does not look Gaussian (reject H0)')


# Anderson-Darling Test
from scipy.stats import anderson
result = anderson(train["SalePrice"])
print('Statistic: %.3f' % result.statistic)
p = 0
for i in range(len(result.critical_values)):
	sl, cv = result.significance_level[i], result.critical_values[i]
	if result.statistic < result.critical_values[i]:
		print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
	else:
		print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))

# Price is not normal....



# DECISION TREE ON EVERYTHING!!!!

from sklearn.tree import DecisionTreeRegressor

X = train.drop("SalePrice", axis = 1)
y = train["SalePrice"]

# removing the categorical varialble
X_cat = X.copy()
map_cat = {}
for x in list_cat: 
    labels = X_cat[x].astype('category').cat.categories.tolist()
    replace_map_comp = {x : {k: v for k,v in zip(
        labels,list(range(1,len(labels)+1)))}}
    X_cat.replace(replace_map_comp, inplace=True)
    map_cat[x] = replace_map_comp
    
# Replacing the missing values
X_cat.fillna(X_cat.mean(), inplace=True)

regressor = DecisionTreeRegressor(min_samples_leaf=10,
                                  max_depth = 5)
regression = regressor.fit(X_cat, y)

y_1 = regressor.predict(X_cat)

regression.get_depth()
regression.get_n_leaves()
regression.get_params()
#regression.decision_path(X_cat).todense()

import numpy as np
RMSE = np.sqrt(np.mean(y_1 - y)**2)


regression.score(X_cat, y)


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz

export_graphviz(regression, 
                out_file='tree.dot', 
                feature_names = X_cat.columns,
                rounded = True, proportion = False, 
                precision = 2, filled = True)


# Get the png
#dot -Tpng tree.dot -o OutputFile.png  

# Display in python
import matplotlib.pyplot as plt
plt.figure(figsize = (14, 18))
plt.imshow(plt.imread('tree.png'))
plt.axis('off');
plt.show();



#import matplotlib.pyplot as plt
#plt.figure()
#
#plt.scatter(X_cat["OverallQual"], y, c = "k", 
#            label="training samples")
#
#plt.plot(X_cat["OverallQual"], y_1, c = "g", 
#         linewidth = 2 )
#
#plt.xlabel("data")
#plt.ylabel("target")
#plt.title("Decision Tree Regression")
#plt.legend()
#plt.show()
#



# Plot SaleCondition and OverallQual
import seaborn as sns
sns.scatterplot(x = "SalePrice", y = "OverallQual", 
                hue = "SaleCondition", 
                data = train[train.SaleCondition != "Normal"])

# Plot ExterCond and OverQual
from pandas.api.types import CategoricalDtype
cat_type = CategoricalDtype(categories = ["Po", "Fa", "TA", "Gd", "Ex"], 
                            ordered = True)
train["ExterCond"] = train["ExterCond"].astype(cat_type)
sns.catplot(x = "OverallQual", y = "ExterCond", data = train)
# suprisingly not linear ...

# Linear Reg on OverallQual
import statsmodels.api as sm

x_sm = train["OverallQual"]
x_sm = sm.add_constant(x_sm)

y = train["SalePrice"]
y_stand =  (y - y.mean()) / y.std()

model = sm.OLS(y_stand, x_sm)
results = model.fit()
print(results.summary())

print('Parameters: ', results.params)
print('R2: ', results.rsquared)


import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std

prstd, iv_l, iv_u = wls_prediction_std(results)
x = x_sm["OverallQual"]
fig, ax = plt.subplots(figsize=(8,6))

ax.plot(x, y_stand, 'o', label="data")
#ax.plot(x, y_stand, 'b-', label="True")
ax.plot(x, results.fittedvalues, 'r--.', label="OLS")
ax.plot(x, iv_u, 'r--')
ax.plot(x, iv_l, 'r--')
ax.legend(loc='best');



