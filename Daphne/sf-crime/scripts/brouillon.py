# Multinomial logistic regression
from sklearn.linear_model import LogisticRegressionCV
lf = LogisticRegressionCV(cv=5, random_state=0, multi_class='multinomial').fit(Xtrain, Ytrain)


from joblib import dump, load
dump(lf, './models/multinomLogisiticReg.joblib')
#lf = load('./models/multinomLogisiticReg.joblib')

lf.get_params()
lf.coef_
lf.intercept_

scc = lf.score(Xtest, Ytest)
pred_all_class = lf.predict_proba(Xtest)
mayor_class = [max(allpreds) for allpreds in pred_all_class]
mayor_class[0:2]

pred = lf.predict(Xtest)
allyouneedtoknow(pred, Ytest)

len(Ytest.unique())