import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import ds_functions as ds
import os

graphsDir = './Results/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)


print('----------------------------')
print('-                          -')
print('-      QOT Naive Bayes     -')
print('-                          -')
print('----------------------------')



data: pd.DataFrame = pd.read_csv('../../Dataset/qsar_oral_toxicity.csv', sep=';', header=None)


print('QOT Naive Bayes - Performance & Confusion matrix')
y: np.ndarray = data.pop(1024).values
X: np.ndarray = data.values
labels = pd.unique(y)

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

clf = GaussianNB()
clf.fit(trnX, trnY)
prd_trn = clf.predict(trnX)
prd_tst = clf.predict(tstX)
ds.plot_evaluation_results(pd.unique(y), trnY, prd_trn, tstY, prd_tst)
plt.suptitle('QOT Naive Bayes - Performance & Confusion matrix')
plt.savefig(graphsDir + 'QOT Naive Bayes - Performance & Confusion matrix')



print('QOT Naive Bayes - Comparison of Naive Bayes Models')
estimators = {'GaussianNB': GaussianNB(),
              'MultinomialNB': MultinomialNB(),
              'BernoulyNB': BernoulliNB()}

xvalues = []
yvalues = []
for clf in estimators:
    xvalues.append(clf)
    estimators[clf].fit(trnX, trnY)
    prdY = estimators[clf].predict(tstX)
    yvalues.append(metrics.accuracy_score(tstY, prdY))

plt.figure(figsize=(7,7))
ds.bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
plt.suptitle('QOT Naive Bayes - Comparison of Naive Bayes Models')
plt.savefig(graphsDir + 'QOT Naive Bayes - Comparison of Naive Bayes Models')
