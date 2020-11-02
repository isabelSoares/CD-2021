import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import ds_functions as ds
import os

graphsDir = './Results/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)


print('----------------------------')
print('-                          -')
print('-     HFCR Naive Bayes     -')
print('-                          -')
print('----------------------------')



data: pd.DataFrame = pd.read_csv('../../Dataset/heart_failure_clinical_records_dataset.csv')


print('HFCR Naive Bayes - Performance & Confusion matrix')
y: np.ndarray = data.pop('DEATH_EVENT').values
X: np.ndarray = data.values
labels = pd.unique(y)

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
splitIterator = iter(skf.split(X, y))

clf = GaussianNB()
best_iteration_accuracy = 0
model_sets = ([], [], [], [])
for model in splitIterator:
    trnX = X[model[0]]
    trnY = y[model[0]]
    tstX = X[model[1]]
    tstY = y[model[1]]

    clf.fit(trnX, trnY)
    prd_trn = clf.predict(trnX)
    prd_tst = clf.predict(tstX)

    iteration_accuracy = metrics.accuracy_score(tstY, prd_tst)

    if iteration_accuracy > best_iteration_accuracy:
    	best_iteration_accuracy = iteration_accuracy
    	model_sets = (trnY, prd_trn, tstY, prd_tst)

ds.plot_evaluation_results(pd.unique(y), model_sets[0], model_sets[1], model_sets[2], model_sets[3])
plt.suptitle('HFCR Naive Bayes - Performance & Confusion matrix')
plt.savefig(graphsDir + 'HFCR Naive Bayes - Performance & Confusion matrix')



print('HFCR Naive Bayes - Comparison of Naive Bayes Models')
estimators = {'GaussianNB': GaussianNB(),
              'MultinomialNB': MultinomialNB(),
              'BernoulyNB': BernoulliNB()}

xvalues = []
yvalues = []
for clf in estimators:
	xvalues.append(clf)
	splitIterator = iter(skf.split(X, y))

	best_iteration_accuracy = 0
	for model in splitIterator:
		trnX = X[model[0]]
		trnY = y[model[0]]
		tstX = X[model[1]]
		tstY = y[model[1]]

		estimators[clf].fit(trnX, trnY)
		prd_tst = estimators[clf].predict(tstX)

		iteration_accuracy = metrics.accuracy_score(tstY, prd_tst)

		if iteration_accuracy > best_iteration_accuracy:
			best_iteration_accuracy = iteration_accuracy

	yvalues.append(best_iteration_accuracy)

plt.figure(figsize=(7,7))
ds.bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
plt.suptitle('HFCR Naive Bayes - Comparison of Naive Bayes Models')
plt.savefig(graphsDir + 'HFCR Naive Bayes - Comparison of Naive Bayes Models')
