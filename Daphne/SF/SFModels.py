import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from collections import OrderedDict

def allyouneedtoknow(pred, true):
    R2_LLars = r2_score(pred, true)
    print("Rscore :" + str(R2_LLars))

    EV_LLars = explained_variance_score(true, pred)
    print("Variance explained: " + str(EV_LLars))

    f, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim((0, max([max(pred), max(true)])))
    ax.set_ylim((0, max([max(pred), max(true)])))
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.scatter(true, pred, c=".3")
    add_identity(ax, color='r', ls='--')
    plt.show()

def save_pred(data_dict, test, prediction, name):
    data_dict_new = OrderedDict(sorted(data_dict.items()))

    result_df = pd.DataFrame({
        "Id": test["Id"]
    })

    for key, value in data_dict_new.items():
        result_df[key] = 0

    count = 0
    for item in prediction:
        for key, value in data_dict.items():
            if (value == item):
                result_df[key][count] = 1
        count += 1

    result_df.to_csv("./data/submission_" + name + ".csv", index=False)


# Read the data
train = pd.read_csv("./data/train_prepared.csv")

data_dict = {}
count = 1
cat = train["Category"].unique()
for data in cat:
    data_dict[data] = count
    count += 1
train["Category"] = train["Category"].replace(data_dict)

test = pd.read_csv("./data/test_prepared.csv")

#train, test = train_test_split(df, test_size=0.2, random_state=42)


# Split the data
Xtrain = train.drop(columns=['Category'])
Ytrain = train.Category

Xtest = test.drop(columns=['Id'])

# NN, Multi Layer Perception
MLP = MLPClassifier(solver='adam', alpha=1e-5,
                    random_state=1)
MLP.fit(Xtrain, Ytrain)
predictions_MLP = MLP.predict_proba(Xtest)

# Stochastic Gradient Descent
SGD = SGDClassifier(loss="modified_huber",
                    penalty="l2", learning_rate='optimal',
                    class_weight="balanced")
SGD.fit(Xtrain, Ytrain)
predictions_SGD = SGD.predict_proba(Xtest)

# Nearest Neighbors
NN = KNeighborsClassifier(len(cat))
NN.fit(Xtrain, Ytrain)
predictions_NN = NN.predict_proba(Xtest)
len(predictions_NN[0])

# Linear SVM
LSVM = SVC(kernel="linear", C=0.025, max_iter=10000)
LSVM.fit(Xtrain, Ytrain)
predictions_LSVM = LSVM.predict_proba(Xtest)

# RBF SVM
SVMRBF = SVC(gamma=2, C=1)
SVMRBF.fit(Xtrain, Ytrain)
predictions_RBF = SVMRBF.predict_proba(Xtest)

# Gaussian Process
GP = GaussianProcessClassifier(1.0 * RBF(1.0))
GP.fit(Xtrain, Ytrain)
predictions_GP = GP.predict_proba(Xtest)

# Decision Tree
DT = DecisionTreeClassifier(min_samples_leaf=10, class_weight="balanced")
DT.fit(Xtrain, Ytrain)
predictions_DT = DT.predict_proba(Xtest)

# Random Forest
RF = RandomForestClassifier(min_samples_leaf=10, class_weight="balanced")
RF.fit(Xtrain, Ytrain)
predictions_RF = RF.predict_proba(Xtest)

# AdaBoost
AD = AdaBoostClassifier()
AD.fit(Xtrain, Ytrain)
predictions_AD = AD.predict_proba(Xtest)

# Naive Bayes
NB = GaussianNB()
NB.fit(Xtrain, Ytrain)
predictions_NB = NB.predict_proba(Xtest)

# QDA
QDA = QuadraticDiscriminantAnalysis()
QDA.fit(Xtrain, Ytrain)
predictions_QDA = QDA.predict_proba(Xtest)


# Voting
from sklearn.ensemble import VotingClassifier
eclf1 = VotingClassifier(estimators=[('MLP', MLP),
                                     ('SGD', SGD),
                                     ('NN', NN),
#                                     ('GP', GP),
                                     ('DT', DT),
                                     ('RF', RF),
                                     ('AD', AD),
#                                     ('NB', NB),
                                     ('QDA', QDA)],
                         voting='soft',
                         n_jobs=-1)

eclf1 = eclf1.fit(Xtrain, Ytrain)
 pred_vote = eclf1.predict_proba(Xtest)
pred_vote.to_csv("./data/pred_voting.csv", index=False)
eclf1.classes_
eclf1.getparams()

data_dict_new = OrderedDict(sorted(data_dict.items()))
result_df = pd.DataFrame({
    "Id": test["Id"]
})

for key, value in data_dict_new.items():
    result_df[key] = 0

count = 0
for item in prediction:
    for key, value in data_dict.items():
        if value == item:
            result_df[key][count] = 1
    count += 1

p = pd.concat([pd.DataFrame(predictions_log_proba), pd.DataFrame(predictions_nb_proba),
               pd.DataFrame(predictions_abc_proba), pd.DataFrame(predictions_rf_proba)]).groupby(level=0).mean()

# data_dict["Key"]
print(data_dict.keys())
p.columns = data_dict.keys()
p['Id'] = p.index
p_id = p['Id']
p.drop('Id', axis=1, inplace=True)
p.insert(0, 'Id', p_id)

result_df.to_csv("./data/submission_voting.csv", index=False)
