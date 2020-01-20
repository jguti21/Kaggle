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
final = train
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

LLars = LassoLars(alpha=0.01, max_iter=199999)
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
# 0.8916: same + outliers removal + demean of y

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


coef = pd.Series(LLars.coef_, index = X_train.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other "
      +  str(sum(coef == 0)) + " variables")

imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])

plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the LAR Model")


########################################################################################################################
##################                              PATH COMPUTATION                                      ##################
########################################################################################################################
from sklearn.linear_model import lars_path

alpha, active, coefs = lars_path(np.array(X_train), np.array(y_train), method='lar', verbose=True, Gram='auto',
                                 alpha_min = 0.01, max_iter= 1999999)

xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.axis('tight')
plt.show()


########################################################################################################################
##################                            BIC CRITERION                                     ##################
########################################################################################################################
from sklearn.linear_model import LassoLarsIC
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html#sphx-glr-auto-examples-linear-model-plot-lasso-model-selection-py
EPSILON = 1e-4
X = np.array(X_train)
y = np.array(y_train)

model_bic = LassoLarsIC(criterion='bic')

model_bic.fit(X, y)
alpha_bic_ = model_bic.alpha_
# alpha of 159
BIC_pred = model_bic.predict(np.array(X_test))
R2_BIC = r2_score(BIC_pred, np.array(y_test)) # 0.796

model_aic = LassoLarsIC(criterion='aic')
model_aic.fit(X, y)
alpha_aic_ = model_aic.alpha_
# alpha 54.23 (really different from the BIC)
AIC_pred = model_aic.predict(np.array(X_test))
R2_AIC = r2_score(AIC_pred, np.array(y_test)) # 0.879

def plot_ic_criterion(model, name, color):
    alpha_ = model.alpha_ + EPSILON
    alphas_ = model.alphas_ + EPSILON
    criterion_ = model.criterion_
    plt.plot(-np.log10(alphas_), criterion_, '--', color=color,
             linewidth=3, label='%s criterion' % name)
    plt.axvline(-np.log10(alpha_), color=color, linewidth=3,
                label='alpha: %s estimate' % name)
    plt.xlabel('-log(alpha)')
    plt.ylabel('criterion')

plt.figure()
plot_ic_criterion(model_aic, 'AIC', 'b')
plot_ic_criterion(model_bic, 'BIC', 'r')
plt.legend()
plt.title('Information-criterion for model selection')

########################################################################################################################
##################                              CROSS-VALIDATION                                      ##################
########################################################################################################################
from sklearn.linear_model import LassoLarsCV

# Compute paths
print("Computing regularization path using the Lars lasso...")
model = LassoLarsCV(cv=10, max_iter=199999999999).fit(X, y)
model.cv_alphas_
model.mse_path_

model.alpha_
# 96.84

letsee = model.coef_
letsee = [x for x in letsee if x != 0.00000000e+00]
len(letsee)

CV_pred = model.predict(np.array(X_test))
R2_CV = r2_score(CV_pred, np.array(y_test)) # 0.85

f, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim((0, max([max(LLars_pred), max(y_test)]) + 10000))
ax.set_ylim((0, max([max(LLars_pred), max(y_test)]) + 10000))
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.scatter(y_test, CV_pred, c=".3")
add_identity(ax, color='r', ls='--')
plt.show()


########################################################################################################################
##################                          WITH VARIABLES SELECTION                                  ##################
########################################################################################################################
