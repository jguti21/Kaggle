import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

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


# Read the data
train = pd.read_csv("./data/train_prepared.csv")

data_dict = {}
count = 1
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
from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier(solver='adam', alpha=1e-5,
                    random_state=1)
MLP.fit(Xtrain, Ytrain)
predictions = MLP.predict_proba(Xtest)
prediction = MLP.predict(Xtest)

from collections import OrderedDict
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

result_df.to_csv("./data/submission_MLP.csv", index=False)


# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
SGD = SGDClassifier(loss="modified_huber",
                    penalty="l2", learning_rate='optimal',
                    class_weight="balanced")

SGD.fit(Xtrain, Ytrain)
predictions = SGD.predict_proba(Xtest)
prediction = SGD.predict(Xtest)

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

result_df.to_csv("./data/submission_SGD.csv", index=False)


# Support Vector Machines
from sklearn import svm
svc = svm.SVC(class_weight="balanced", decision_function_shape="ovo")
svc.fit(Xtrain, Ytrain)

predictions = svc.predict_proba(Xtest)
prediction = svc.predict(Xtest)

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

result_df.to_csv("./data/submission_SVC.csv", index=False)


# Voting
from sklearn.ensemble import VotingClassifier
eclf1 = VotingClassifier(estimators=[('MLP', MLP),
                                     ('SGD', SGD),
                                     ('SVC', svc)],
                         voting='hard')

eclf1 = eclf1.fit(Xtrain, Ytrain)

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

result_df.to_csv("./data/submission_voting.csv", index=False)
