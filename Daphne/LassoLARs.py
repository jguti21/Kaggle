########################################################################################################################
##################                                  IMPORT MODULES                                    ##################
########################################################################################################################
import time
import numpy as np
import os
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
########################################################################################################################
##################                              USEFUL FUNCTIONS                                   ##################
########################################################################################################################
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

########################################################################################################################
##################                                 LOADING THE DATA                                   ##################
########################################################################################################################

os.chdir(
    "C:/Users/daphn/Documents/Kaggle/house-prices-advanced-regression-techniques"
)

final = pd.read_csv("final.csv")

########################################################################################################################
##################                                SAMPLING THE DATA                                   ##################
########################################################################################################################

X = final.drop(columns="SalePrice")
y = final["SalePrice"].copy()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
len(X_train)
len(X_test)

########################################################################################################################
##################                                LASSO LARS                                          ##################
########################################################################################################################

from sklearn.linear_model import LassoLars


LLars = LassoLars(alpha=0.01)
LLars_fit = LLars.fit(X_train, y_train)

print(LLars.alpha)

# print(LLars.coef_)
# len(LLars.coef_path_)

print(LLars.intercept_)

LLars.active_ # indice of active variables at the end of the path
len(final.columns) - len(LLars.active_) # number of variable taken out

LLars_pred = LLars_fit.predict(X_test)
R2_LLars = r2_score(LLars_pred, y_test)
# 0.78589: base final (all var)
# 0.7883: categorical normalized
# 0.8480: same + var cubed and squared
# 0.8916: same + outliers removal

f, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim((0, max([max(LLars_pred), max(y_test)]) + 10000))
ax.set_ylim((0, max([max(LLars_pred), max(y_test)]) + 10000))
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.scatter(y_test, LLars_pred, c=".3")
add_identity(ax, color='r', ls='--')
plt.show()

from sklearn.metrics import explained_variance_score
EV_LLars = explained_variance_score(y_test, LLars_pred)


########################################################################################################################
##################                              CROSS-vALIDATION                                      ##################
########################################################################################################################


from sklearn.linear_model import LassoLarsCV