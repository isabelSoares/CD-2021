import os
import numpy as np
import pandas as pd
import ds_functions as ds
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import data_preparation_functions as prepfunctions
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from subprocess import call


graphsDir = './Results/KNN/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

data: pd.DataFrame = pd.read_csv('../Dataset/heart_failure_clinical_records_dataset.csv')
datas = prepfunctions.prepare_dataset(data, 'DEATH_EVENT', True, True)
featured_datas = prepfunctions.mask_feature_selection(datas, 'DEATH_EVENT', False, './Results/FeatureSelection/HFCR Feature Selection - Features')
best_accuracies = {}

for key in datas:
    for do_feature_eng in [False, True]:
        if (do_feature_eng):
            data = featured_datas[key]
            subDir = graphsDir + 'FeatureEng/' +  key + '/'
            if not os.path.exists(subDir):
                os.makedirs(subDir)
        else:
            data = datas[key]
            subDir = graphsDir + key + '/'
            if not os.path.exists(subDir):
                os.makedirs(subDir)

        y: np.ndarray = data.pop('DEATH_EVENT').values
        X: np.ndarray = data.values
        labels = pd.unique(y)
        
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        splitList = list(skf.split(X, y))

        nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
        dist = ['manhattan', 'euclidean', 'chebyshev']
        values = {}
        best = (0, '')
        last_best = 0
        last_train_best = 0

        overfitting_values = {}
        for d in dist:
            yvalues = []
            overfitting_values[d] = {}
            overfitting_values[d]['test'] = []
            overfitting_values[d]['train'] = []
            for n in nvalues:
                best_iteration_train_accuracy = 0
                best_iteration_accuracy = 0
                for model in splitList:
                    knn = KNeighborsClassifier(n_neighbors=n, metric=d)
                    trnX = X[model[0]] 
                    trnY = y[model[0]]
                    tstX = X[model[1]]
                    tstY = y[model[1]]
                    knn.fit(trnX, trnY)
                    prdY = knn.predict(tstX)
                    prd_trainY = knn.predict(trnX)

                    iteration_accuracy = metrics.accuracy_score(tstY, prdY)
                    if iteration_accuracy > best_iteration_accuracy:
                        best_iteration_accuracy = iteration_accuracy
                        best_iteration_train_accuracy = metrics.accuracy_score(trnY, prd_trainY)
                        model_sets = (trnX, trnY, tstX, tstY)
                
                overfitting_values[d]['train'].append(best_iteration_train_accuracy)
                overfitting_values[d]['test'].append(best_iteration_accuracy)
                yvalues.append(best_iteration_accuracy)
                if yvalues[-1] > last_best:
                    best = (n, d)
                    last_best = yvalues[-1]
                    last_train_best = best_iteration_train_accuracy
                    best_model = tuple(model_sets)             
            values[d] = yvalues

        text = key
        if (do_feature_eng): text += ' with FS'
        best_accuracies[text] = [last_train_best, last_best]

        plt.figure()
        ds.multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel='accuracy', percentage=True)
        plt.suptitle('HFCR KNN - ' + key + ' - parameters')
        plt.savefig(subDir + 'HFCR KNN - ' + key + ' - parameters')
        print('Best results with %d neighbors and %s'%(best[0], best[1]))

        plt.figure()
        fig, axs = plt.subplots(1, len(dist), figsize=(32, 8), squeeze=False)
        i = 0
        for k in range(len(dist)):
            d = dist[k]
            ds.multiple_line_chart(nvalues, overfitting_values[d], ax=axs[0, k], title='Overfitting for dist = %s'%(d), xlabel='K Neighbours', ylabel='accuracy', percentage=True)
        plt.suptitle('HFCR Overfitting - KNN')
        plt.savefig(subDir + 'HFCR Overfitting - KNN')

        trnX, trnY, tstX, tstY = best_model
        clf = knn = KNeighborsClassifier(n_neighbors=best[0], metric=best[1])
        clf.fit(trnX, trnY)
        prd_trn = clf.predict(trnX)
        prd_tst = clf.predict(tstX)
        ds.plot_evaluation_results(pd.unique(y), trnY, prd_trn, tstY, prd_tst)
        plt.suptitle('HFCR KNN - ' + key + ' - Performance & Confusion matrix - %d neighbors and %s'%(best[0], best[1]))
        plt.savefig(subDir + 'HFCR KNN - ' + key + '- Performance & Confusion matrix')

        plt.close("all")

plt.figure(figsize=(7,7))
ds.multiple_bar_chart(['Train', 'Test'], best_accuracies, ylabel='Accuracy')
plt.suptitle('HFCR Sampling & Feature Selection')
plt.savefig(graphsDir + 'HFCR Sampling & Feature Selection')