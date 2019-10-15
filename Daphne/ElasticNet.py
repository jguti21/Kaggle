# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 21:44:49 2019

@author: daphne
"""
########################################################################################################################
##################                                  IMPORT MODULES                                    ##################
########################################################################################################################
import time
import numpy as np
import os
import pandas as pd
from sklearn import linear_model, metrics
from sklearn.model_selection import GridSearchCV
from itertools import cycle
import matplotlib.pyplot as plt

########################################################################################################################
##################                                 LOADING THE DATA                                   ##################
########################################################################################################################

os.chdir(
    "C:/Users/daphn/Documents/Kaggle/house-prices-advanced-regression-techniques"
)

final = pd.read_csv("final.csv")

#### SAMPLE
X = final.drop(columns="SalePrice")
y = final["SalePrice"].copy()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.1,
                                                    random_state=42)

########################################################################################################################
##################                                 LASSO REGRESSION                                   ##################
########################################################################################################################
from sklearn.linear_model import Lasso

regr_lasso = Lasso(fit_intercept=False)
regr_lasso.fit(X_train, y_train)

print(regr_lasso.coef_)
print(regr_lasso.alpha) # why 1 ?!

pred_lasso = regr_lasso.predict(X_test)
RMSE = np.sqrt(np.mean(pred_lasso - y_test) ** 2)
R2_lasso = regr_lasso.score(X_test, y_test)
# 0.84

########################################### Hyper parameters tuning ####################################################
lasso_params = {'alpha': range(1, 10000, 10)}

Lasso_hyper = GridSearchCV(linear_model.Lasso(fit_intercept=False, precompute=True),
                           param_grid=lasso_params)
Lasso_fit = Lasso_hyper.fit(X_train, y_train)

Lasso_fit.best_estimator_
# alpha = 1161

Lasso_pred = Lasso_fit.predict(X_test)
R2_Lasso_hyper = metrics.r2_score(Lasso_pred, y_test)
# less good that the normal lasso...

################################################## CROSS VALIDATION ####################################################
# This is to avoid division by zero while doing np.log10
EPSILON = 1e-4

# LassoLarsIC: least angle regression with BIC/AIC criterion
from sklearn.linear_model import LassoLarsIC
model_bic = LassoLarsIC(criterion='bic')
t1 = time.time()
bic_fit = model_bic.fit(X, y)
t_bic = time.time() - t1
alpha_bic_ = model_bic.alpha_
bic_pred = bic_fit.predict(X_test)
R2_bic = metrics.r2_score(bic_pred, y_test)
# 0,556 du coup pas bon

from sklearn.linear_model import LassoLarsIC
model_aic = LassoLarsIC(criterion='aic')
aic_fit = model_aic.fit(X, y)
alpha_aic_ = model_aic.alpha_
aic_pred = aic_fit.predict(X_test)
R2_aic = metrics.r2_score(aic_pred, y_test)
# 0,756 du coup pas top

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
plt.title('Information-criterion for model selection (training time %.3fs)'
          % t_bic)

# #############################################################################
# LassoCV: coordinate descent
from sklearn.linear_model import LassoCV

# Compute paths
print("Computing regularization path using the coordinate descent lasso...")
LassoCV_fit = LassoCV(cv=20).fit(X, y)
LassoCV_pred = LassoCV_fit.predict(X_test)
R2_LassoCV = metrics.r2_score(LassoCV_pred, y_test)
# 0.60 pas top du tout

# Display results
m_log_alphas = -np.log10(model.alphas_ + EPSILON)

plt.figure()
ymin, ymax = 2300, 3800
plt.plot(m_log_alphas, model.mse_path_, ':')
plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_ + EPSILON), linestyle='--', color='k',
            label='alpha: CV estimate')

plt.legend()

plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.title('Mean square error on each fold: coordinate descent '
          '(train time: %.2fs)' % t_lasso_cv)
plt.axis('tight')
plt.ylim(ymin, ymax)

# LassoLarsCV: least angle regression
from sklearn.linear_model import LassoLarsCV
# Compute paths
print("Computing regularization path using the Lars lasso...")
LassoLarsCV_fit = LassoLarsCV(cv=20).fit(X, y)
LassoLarsCV_pred = LassoLarsCV_fit.predict(X_test)
R2_LassoLarsCV = metrics.r2_score(LassoLarsCV_pred, y_test)
# 0.776 du coup un peu meilleur

# Display results
m_log_alphas = -np.log10(model.cv_alphas_ + EPSILON)

plt.figure()
plt.plot(m_log_alphas, model.mse_path_, ':')
plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()

plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.title('Mean square error on each fold: Lars (train time: %.2fs)'
          % t_lasso_lars_cv)
plt.axis('tight')
plt.ylim(ymin, ymax)

plt.show()
################################################

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

# Compute paths
eps = 5e-3  # the smaller it is the longer is the path

from sklearn.linear_model import lasso_path
print("Computing regularization path using the lasso...")
alphas_lasso, coefs_lasso, _ = lasso_path(X_train, y_train, eps, fit_intercept=False)

print("Computing regularization path using the positive lasso...")
alphas_positive_lasso, coefs_positive_lasso, _ = lasso_path(
    X_train, y_train, eps, positive=True, fit_intercept=False)

print("Computing regularization path using the elastic net...")
from sklearn.linear_model import enet_path
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
# 0.837

# CV
from sklearn.linear_model import ElasticNetCV

regr_En_cv = ElasticNetCV(cv = 3, fit_intercept = False,
                          normalize= False)

parametersGrid = {"alpha": range(0, 10000, 10),
                  "l1_ratio": np.arange(0.001, 1.0, 0.1)}

grid = GridSearchCV(regr_En_cv, parametersGrid, scoring='r2', cv= 3)

grid.fit(X_train, y_train)
pred_En_cv = regr_En_cv.predict(X_test)
R2_En_cv = regr_En_cv.score(X_test, y_test)
regr_En_cv.alpha_





R2_En > R2_lasso # En is slightly better

import matplotlib.pyplot as plt
plt.plot(regr_En.coef_,alpha=0.7, linestyle='none',marker='*',markersize=5,
         color='red',label=r'ElasticNet; $\alpha = 1$',zorder=7) # zorder for ordering the markers

plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.show()



# Ali Furkan Kalay
def test(models, data, iterations = 100):
    results = {}
    for i in models:
        r2_train = []
        r2_test = []
        for j in range(iterations):
            X_train, X_test, y_train, y_test = train_test_split(data[X],
                                                                data[Y],
                                                                test_size= 0.2)
            r2_test.append(metrics.r2_score(y_test,
                                            models[i].fit(X_train,
                                                         y_train).predict(X_test)))
            r2_train.append(metrics.r2_score(y_train,
                                             models[i].fit(X_train,
                                                          y_train).predict(X_train)))
        results[i] = [np.mean(r2_train), np.mean(r2_test)]
    return pd.DataFrame(results)


from sklearn import linear_model
from sklearn import metrics

models = {'OLS': linear_model.LinearRegression(),
         'Lasso': linear_model.Lasso(),
         'Ridge': linear_model.Ridge()}

X = list(final.drop(columns="SalePrice").columns)
Y = ["SalePrice"]

test(models, final)


lasso_params = {'alpha': range(1, 10000, 10)}
ridge_params = {'alpha': range(1, 10000, 10)}

models2 = {'OLS': linear_model.LinearRegression(),
           'Lasso': GridSearchCV(linear_model.Lasso(),
                               param_grid=lasso_params).fit(final[X], final[Y]).best_estimator_,
           'Ridge': GridSearchCV(linear_model.Ridge(),
                               param_grid=ridge_params).fit(final[X], final[Y]).best_estimator_,}

test(models2, final)
# 0.78 pour Ridge ! good good

# Poly features added
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

pipe1 = Pipeline([('poly', PolynomialFeatures()),
                 ('fit', linear_model.LinearRegression())])
pipe2 = Pipeline([('poly', PolynomialFeatures()),
                 ('fit', linear_model.Lasso())])
pipe3 = Pipeline([('poly', PolynomialFeatures()),
                 ('fit', linear_model.Ridge())])

lasso_params = {'alpha': range(2, 10000, 10)}
ridge_params = {'alpha': range(2, 10000, 10)}

models3 = {'OLS': pipe1,
           'Lasso': GridSearchCV(pipe2,
                                 param_grid=lasso_params).fit(final[X], final[Y]).best_estimator_ ,
           'Ridge': GridSearchCV(pipe3,
                                 param_grid=ridge_params).fit(final[X], final[Y]).best_estimator_,}

test(models3, final)