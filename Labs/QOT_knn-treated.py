import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier
import data_preparation_functions as prepfunctions
import ds_functions as ds
import os
from subprocess import call
from datetime import datetime

graphsDir = './Results/KNN/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

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

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(current_time, " : ", key)

        trnY: np.ndarray = data.pop(1024).values 
        trnX: np.ndarray = data.values
        tstY: np.ndarray = testData.pop(1024).values 
        tstX: np.ndarray = testData.values

        nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
        dist = ['manhattan', 'euclidean', 'chebyshev']
        values = {}
        best = (0, '')
        last_best = 0
        last_train_best = 0

        overfitting_values = {}
        for d in dist:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(current_time, " : ", d)
            yvalues = []
            overfitting_values[d] = {}
            overfitting_values[d]['test'] = []
            overfitting_values[d]['train'] = []
            for n in nvalues:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(current_time, " : ", n)
                knn = KNeighborsClassifier(n_neighbors=n, metric=d, n_jobs=-1)
                knn.fit(trnX, trnY)
                prdY = knn.predict(tstX)
                prdTrainY = knn.predict(trnX)

                test_accuracy = metrics.accuracy_score(tstY, prdY)
                train_accuracy = metrics.accuracy_score(trnY, prdTrainY)
                overfitting_values[d]['test'].append(test_accuracy)
                overfitting_values[d]['train'].append(train_accuracy)
                yvalues.append(test_accuracy)
                if yvalues[-1] > last_best:
                    best = (n, d)
                    last_best = yvalues[-1]
                    last_train_best = train_accuracy
            values[d] = yvalues

        text = key
        if (do_feature_eng): text += ' with FS'
        best_accuracies[text] = [last_train_best, last_best]

        plt.figure()
        ds.multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel='accuracy', percentage=True)
        plt.suptitle('QOT KNN - ' + key + ' - parameters')
        plt.savefig(subDir + 'QOT KNN - ' + key + ' - parameters')
        print('Best results with %d neighbors and %s'%(best[0], best[1]))

        plt.figure()
        fig, axs = plt.subplots(1, len(dist), figsize=(32, 8), squeeze=False)
        i = 0
        for k in range(len(dist)):
            d = dist[k]
            ds.multiple_line_chart(nvalues, overfitting_values[d], ax=axs[0, k], title='Overfitting for dist = %s'%(d), xlabel='K Neighbours', ylabel='accuracy', percentage=True)
        plt.suptitle('QOT Overfitting - KNN')
        plt.savefig(subDir + 'QOT Overfitting - KNN')

        clf = knn = KNeighborsClassifier(n_neighbors=best[0], metric=best[1])
        clf.fit(trnX, trnY)
        prd_trn = clf.predict(trnX)
        prd_tst = clf.predict(tstX)
        ds.plot_evaluation_results(["negative", "positive"], trnY, prd_trn, tstY, prd_tst)
        plt.suptitle('QOT KNN - ' + key + '- Performance & Confusion matrix - %d neighbors and %s'%(best[0], best[1]))
        plt.savefig(subDir + 'QOT KNN - ' + key + ' - Performance & Confusion matrix')

        plt.close("all")

plt.figure(figsize=(7,7))
ds.multiple_bar_chart(['Train', 'Test'], best_accuracies, ylabel='Accuracy')
plt.suptitle('QOT Sampling & Feature Selection')
plt.savefig(graphsDir + 'QOT Sampling & Feature Selection')