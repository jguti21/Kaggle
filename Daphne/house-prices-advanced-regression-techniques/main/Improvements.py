########################################################################################################################
##################                                  IMPORT MODULES                                    ##################
########################################################################################################################
import os
import pandas as pd
import category_encoders as ce
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
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

########################################################################################################################
##################                                SAMPLING THE DATA                                   ##################
########################################################################################################################
final = pd.read_csv("train.csv")

final_copy = final.copy()
train = final_copy.sample(frac=0.75, random_state=79328)
test = final_copy.drop(train.index)

########################################################################################################################
##################                             EVALUATING THE MEAN                                    ##################
########################################################################################################################

mean_y = np.mean(train["SalePrice"])
median_y = np.median(train["SalePrice"])

########################################################################################################################
##################                                CLEANING FUNCTION                                   ##################
########################################################################################################################
from modules.preparationtrain import cleaning_function_train
train = cleaning_function_train(train)
test = cleaning_function_train(test)

y_train = train["SalePrice"]
X_train = train.drop(["SalePrice", "Id"], axis = 1)

y_test = test["SalePrice"]
X_test = test.drop(["SalePrice", "Id"], axis = 1)

#train["outlier"] = LocalOutlierFactor().fit_predict(train)
#test["outlier"] = LocalOutlierFactor().fit_predict(test)

########################################################################################################################
##################                             LEAST ANGLE REGRESSION                                 ##################
########################################################################################################################

from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
#import xgboost as xgb

y_LARS_train = train["SalePrice"] - np.mean(train["SalePrice"])
model = LassoLarsCV(cv=10, max_iter=199999999).fit(X_train, y_LARS_train)
model.alpha_
pred = model.predict(X_test)
pred += np.median(train["SalePrice"])
from modules.modelaccuracy import allyouneedtoknow
allyouneedtoknow(pred, y_test)

from sklearn import linear_model
y_other_train = np.log1p(train["SalePrice"])

from sklearn.svm import  SVR
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1)
rbf = svr_lin.fit(X_train, y_other_train)

pred = np.expm1(rbf.predict(X_test))
allyouneedtoknow(pred, y_test)
y_test[0:5]

#
#
#
#
#
#
#
#
#
# from sklearn.cluster import AgglomerativeClustering
# kmeans = AgglomerativeClustering(n_clusters=4,
#                                  affinity="manhattan", linkage= "average").fit(X_train)
# X_train["Cluster"] = kmeans.labels_
# total = pd.concat([X_train, y_train], axis = 1)
#
# X_test["Cluster"] = kmeans.fit_predict(X_test)
# total_test = pd.concat([X_test, y_test], axis = 1)
#
# # Prediction for each cluster
# for cluster in range(0, kmeans.n_clusters):
#     X_clus = total[total["Cluster"] == cluster].drop("SalePrice", axis = 1)
#     print(len(X_clus))
#     y_clus = total[total["Cluster"] == cluster]["SalePrice"]
#     model = LassoLarsCV(cv=3, max_iter=199999999).fit(X_clus, y_clus)
#
#     X_test_clus = total_test[total_test["Cluster"] == cluster].drop("SalePrice", axis = 1)
#     y_test_clus = total_test[total_test["Cluster"] == cluster]["SalePrice"]
#     pred = model.predict(X_test_clus)
#
#     from modules.modelaccuracy import allyouneedtoknow
#     allyouneedtoknow(pred, y_test_clus)
#
#
#
#
# # alpho: 0.001 best Rscore: 0.70
# #model = LassoCV(alphas = [0.001, 10, 30,100], cv=3, max_iter=1000).fit(X_train, y_train)
# model = LassoLarsCV(cv=3, max_iter=199999999).fit(X_train, y_train)
# model.alpha_
#
# pred = model.predict(X_test)
#
# coef = pd.Series(model.coef_, index = X_train.columns)
# print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other "
#       +  str(sum(coef == 0)) + " variables")
#
# from modules.modelaccuracy import allyouneedtoknow
# allyouneedtoknow(pred, y_test)
#
#
#
#
#
#
#
# # Non OUTLIER
# # Train
# train_inlier = train[train["outlier"] == 1]
# y_train_inlier = train_inlier["SalePrice"]
# X_train_inlier = train_inlier.drop(["SalePrice", "Id"], axis = 1)
# model_inlier = LassoLarsCV(cv=10, max_iter=1000).fit(X_train_inlier, y_train_inlier)
#
# # Score
# test_inlier = test[test["outlier"] == 1]
# y_test_inlier = test_inlier["SalePrice"]
# X_test_inlier = test_inlier.drop(["SalePrice", "Id"], axis = 1)
# Inlier_pred = model_inlier.predict(X_test_inlier)
#
# y_svr = svr.predict(X_test_inlier)
#
# y_svr[0:5]
# y_test_inlier[0:5]
# from modules.modelaccuracy import allyouneedtoknow
# allyouneedtoknow(Inlier_pred, y_test_inlier)
#
#
# from sklearn.metrics import r2_score
# R2_LLars = r2_score(Inlier_pred, y_test_inlier)
# y_test_inlier.head(5)
# np.expm1(Inlier_pred[0:5])
#
#
# # OUTLIER
# # train
# train_outlier = train[train["outlier"] == -1]
# y_train_outlier = train_outlier["SalePrice"]
# X_train_outlier = train_outlier.drop(["SalePrice", "Id"], axis = 1)
# model_outlier = LassoCV(cv=10, max_iter=1000).fit(X_train_outlier, y_train_outlier)
#
# # Score
# test_outlier = test[test["outlier"] == -1]
# X_test_outlier = test_outlier.drop("Id", axis = 1)
# outlier_pred = model_outlier.predict(X_test_outlier)
#
# print(model.coef_)
# len(model.coef_path_)
# print(model.intercept_)
#
# model.active_ # indice of active variables at the end of the path
# len(X_train.columns) - len(model.active_) # number of variable taken out
#
# from sklearn.metrics import r2_score
# R2_LLars = r2_score(CV_pred, y_test)
# # 0.66
#
# f, ax = plt.subplots(figsize=(6, 6))
# ax.set_xlim((0, max([max(CV_pred), max(y_test)]) + 10000))
# ax.set_ylim((0, max([max(CV_pred), max(y_test)]) + 10000))
# ax.set_xlabel("Actual")
# ax.set_ylabel("Predicted")
# ax.scatter(y_test, CV_pred, c=".3")
# add_identity(ax, color='r', ls='--')
# plt.show()
#
# from sklearn.metrics import explained_variance_score
# EV_LLars = explained_variance_score(y_test, CV_pred)
#
# from sklearn.metrics import mean_squared_error
# RMSE = np.sqrt(mean_squared_error(CV_pred, y_test))
#
#
# coef = pd.Series(model.coef_, index = X_train.columns)
# print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other "
#       +  str(sum(coef == 0)) + " variables")
#
# imp_coef = pd.concat([coef.sort_values().head(10),
#                      coef.sort_values().tail(10)])
#
# plt.rcParams['figure.figsize'] = (8.0, 10.0)
# imp_coef.plot(kind = "barh")
# plt.title("Coefficients in the LAR Model")
