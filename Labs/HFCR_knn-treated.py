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
accuracies = {}

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

        trn_x_lst = []
        trn_y_lst = []
        tst_x_lst = []
        tst_y_lst = []
        for train_i, test_i in skf.split(X, y):
            # Train
            trn_X = X[train_i]
            trn_y = y[train_i]

            # Test
            tst_X = X[test_i]
            tst_y = y[test_i]

            trn_x_lst.append(trn_X)
            trn_y_lst.append(trn_y)
            tst_x_lst.append(tst_X)
            tst_y_lst.append(tst_y)

        nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
        dist = ['manhattan', 'euclidean', 'chebyshev']
        values = {}
        best = (0, '')
        last_best = 0
        last_train_best = 0
        best_model = None

        overfitting_values = {}
        for d in dist:
            yvalues = []
            overfitting_values[d] = {}
            overfitting_values[d]['test'] = []
            overfitting_values[d]['train'] = []
            for n in nvalues:
                prd_trn_lst = []
                prd_tst_lst = []
                test_accuracy = 0
                train_accuracy = 0
                for trn_X, trn_y, tst_X, tst_y in zip(trn_x_lst, trn_y_lst, tst_x_lst, tst_y_lst):
                    knn = KNeighborsClassifier(n_neighbors=n, metric=d)
                    knn.fit(trn_X, trn_y)
                    prd_trn = knn.predict(trn_X)
                    prd_tst = knn.predict(tst_X)

                    train_accuracy += metrics.accuracy_score(trn_y, prd_trn)
                    test_accuracy += metrics.accuracy_score(tst_y, prd_tst)

                    prd_trn_lst.append(prd_trn)
                    prd_tst_lst.append(prd_tst)

                test_accuracy /= n_splits
                train_accuracy /= n_splits
                
                overfitting_values[d]['train'].append(train_accuracy)
                overfitting_values[d]['test'].append(test_accuracy)
                yvalues.append(test_accuracy)
                if yvalues[-1] > last_best:
                    best = (n, d)
                    last_best = yvalues[-1]
                    last_train_best = train_accuracy
                    best_model = (prd_trn_lst, prd_tst_lst)
            values[d] = yvalues

        text = key
        if (do_feature_eng): text += ' with FS'
        accuracies[text] = [last_train_best, last_best]

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

        prd_trn_lst = best_model[0]
        prd_tst_lst = best_model[1]
        
        ds.plot_evaluation_results_kfold(pd.unique(y), trn_y_lst, prd_trn_lst, tst_y_lst, prd_tst_lst)
        plt.suptitle('HFCR KNN - ' + key + ' - Performance & Confusion matrix - %d neighbors and %s'%(best[0], best[1]))
        plt.savefig(subDir + 'HFCR KNN - ' + key + '- Performance & Confusion matrix')

        plt.close("all")

plt.figure(figsize=(7,7))
ds.multiple_bar_chart(['Train', 'Test'], accuracies, ylabel='Accuracy')
plt.suptitle('HFCR Sampling & Feature Selection')
plt.savefig(graphsDir + 'HFCR Sampling & Feature Selection')