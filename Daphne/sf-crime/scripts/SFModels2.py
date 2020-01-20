import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import explained_variance_score, r2_score

# Read the data
train = pd.read_csv("./data/train_prepared.csv")
test = pd.read_csv("./data/test_prepared.csv")

# Replace the categories by numbers and store the association in a dictionary
data_dict = {}
count = 1
cat = train["Category"].unique()
for data in cat:
    data_dict[data] = count
    count += 1
train["Category"] = train["Category"].replace(data_dict)

# Split the data
Xtrain = train.drop(columns=['Category'])
Ytrain = train.Category

Xtest = test.drop(columns=['Id'])

# First model on the first fold
# NN, Multi Layer Perception
# Can be useful too loop over the activation functions
# for the solver adam performs well on big dataset but for small ones they advice to choose the lbfgs one
# Tuning the alpha is the L2 penalty parameter
# learning_rate: not useful if we use lbfgs
data = train[0:int(len(train)/5)]
from sklearn.model_selection import train_test_split
sub_train, sub_test = train_test_split(data, test_size=0.2)
help(train_test_split)
xtrain = data.drop(columns=['Category'])
ytrain = data.Category

for alpha in [0.01, 0.05, 0.1, 0.5]:
    MLP = MLPClassifier(solver='lbfgs', alpha=alpha,
                        random_state=1, activation="logisitc")
    MLP.fit(xtrain, ytrain)
    predictions_MLP = MLP.predict_proba(Xtest)
    R2_LLars = r2_score(pred, true)
    print("Rscore :" + str(R2_LLars))

    EV_LLars = explained_variance_score(true, pred)
    print("Variance explained: " + str(EV_LLars))

predictions_MLP = MLP.predict_proba(Xtest)


help(MLPClassifier)
# Creation of the 5 folds ou sinon ça je le fais par indice sur un dataframe mélangé ce qui évite des manipulations
# redondantes
