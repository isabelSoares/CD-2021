import os
import numpy as np
import pandas as pd
import ds_functions as ds
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import data_preparation_functions as prepfunctions
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split

graphsDir = './Results/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('--------------------------------------')
print('-                                    -')
print('-     QOT Naive Bayes - Treated     -')
print('-                                    -')
print('--------------------------------------')

data: pd.DataFrame = pd.read_csv('../../Dataset/heart_failure_clinical_records_dataset.csv')
datas = prepfunctions.prepare_dataset(data, 'DEATH_EVENT', True, True)

for key, value in datas.items():
    data = value
    subDir = graphsDir + key + '/'
    if not os.path.exists(subDir):
        os.makedirs(subDir) 

    print('QOT Naive Bayes - Performance & Confusion matrix')
    y: np.ndarray = data.pop('DEATH_EVENT').values
    X: np.ndarray = data.values
    labels = pd.unique(y)
    
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

    clf = GaussianNB()
    clf.fit(trnX, trnY)
    prd_trn = clf.predict(trnX)
    prd_tst = clf.predict(tstX)
    ds.plot_evaluation_results(pd.unique(y), trnY, prd_trn, tstY, prd_tst)
    plt.suptitle('QOT Naive Bayes - ' + key + ' - Performance & Confusion matrix')
    plt.savefig(subDir + 'QOT Naive Bayes - ' + key + ' - Performance & Confusion matrix')


    print('QOT Naive Bayes - Comparison of Naive Bayes Models')
    estimators = {'GaussianNB': GaussianNB(),
              #'MultinomialNB': MultinomialNB(),
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
    plt.suptitle(subDir + 'QOT Naive Bayes - ' + key + ' - Comparison of Naive Bayes Models')
    plt.savefig(subDir + 'QOT Naive Bayes - ' + key + ' - Comparison of Naive Bayes Models')


    plt.close("all")

