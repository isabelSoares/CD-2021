import os
import numpy as np
import pandas as pd
import ds_functions as ds
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import data_preparation_functions as prepfunctions
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split

graphsDir = './Results/Naive Bayes/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('--------------------------------------')
print('-                                    -')
print('-     QOT Naive Bayes - Treated      -')
print('-                                    -')
print('--------------------------------------')

data: pd.DataFrame = pd.read_csv('../Dataset/qsar_oral_toxicity.csv', sep=';', header=None)
train, test = train_test_split(data, train_size=0.7, stratify=data[1024].values)
testDatas = {}
datas = prepfunctions.prepare_dataset(train, 1024, False, False)
for key in datas:
    testDatas[key] = test.copy()

featured_datas = prepfunctions.mask_feature_selection(datas, 1024, True, './Results/FeatureSelection/QOT Feature Selection - Features')
featured_test_datas = prepfunctions.mask_feature_selection(testDatas, 1024, True, './Results/FeatureSelection/QOT Feature Selection - Features')

best_accuracies = {}

for key in datas:
    for do_feature_eng in [False, True]:
        if (do_feature_eng):
            data = featured_datas[key]
            testData = featured_test_datas[key].copy()
            subDir = graphsDir + 'FeatureEng/' +  key + '/'
            if not os.path.exists(subDir):
                os.makedirs(subDir)
        else:
            data = datas[key]
            testData = test.copy()
            subDir = graphsDir + key + '/'
            if not os.path.exists(subDir):
                os.makedirs(subDir)

        print('QOT Naive Bayes - Performance & Confusion matrix')
        trnY: np.ndarray = data.pop(1024).values 
        trnX: np.ndarray = data.values
        tstY: np.ndarray = testData.pop(1024).values 
        tstX: np.ndarray = testData.values

        clf = GaussianNB()
        clf.fit(trnX, trnY)
        prd_trn = clf.predict(trnX)
        prd_tst = clf.predict(tstX)
        train_accuracy = metrics.accuracy_score(trnY, prd_trn)
        test_accuracy = metrics.accuracy_score(tstY, prd_tst)
        ds.plot_evaluation_results(["negative", "positive"], trnY, prd_trn, tstY, prd_tst)
        plt.suptitle('QOT Naive Bayes - ' + key + ' - Performance & Confusion matrix')
        plt.savefig(subDir + 'QOT Naive Bayes - ' + key + ' - Performance & Confusion matrix')

        text = key
        if (do_feature_eng): text += ' with FS'
        best_accuracies[text] = [train_accuracy, test_accuracy]

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
        plt.suptitle(subDir + 'QOT Naive Bayes - ' + key + ' - Comparison of Naive Bayes Models')
        plt.savefig(subDir + 'QOT Naive Bayes - ' + key + ' - Comparison of Naive Bayes Models')


        plt.close("all")

plt.figure(figsize=(7,7))
ds.multiple_bar_chart(['Train', 'Test'], best_accuracies, ylabel='Accuracy')
plt.suptitle('QOT Sampling & Feature Selection')
plt.savefig(graphsDir + 'QOT Sampling & Feature Selection')