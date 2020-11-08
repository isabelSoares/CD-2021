import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as metrics
import data_preparation_functions as prepfunctions
from sklearn.ensemble import GradientBoostingClassifier
import ds_functions as ds
import os

graphsDir = './Results/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

data: pd.DataFrame = pd.read_csv('../../Dataset/heart_failure_clinical_records_dataset.csv')
datas = prepfunctions.prepare_dataset(data, 'DEATH_EVENT', True, True)
featured_datas = prepfunctions.mask_feature_selection(datas, 'DEATH_EVENT', False, './Results/HFCR Feature Selection - Features')

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

        n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
        max_depths = [5, 10, 25]
        learning_rate = [.1, .3, .5, .7, .9]
        best = ('', 0, 0)
        last_best = 0
        best_tree = None

        cols = len(max_depths)
        plt.figure()
        fig, axs = plt.subplots(1, cols, figsize=(cols*ds.HEIGHT, ds.HEIGHT), squeeze=False)
        overfitting_values = {}
        for k in range(len(max_depths)):
            d = max_depths[k]
            values = {}
            for lr in learning_rate:
                yvalues = []
                overfitting_values[d] = {}
                overfitting_values[d]['test'] = []
                overfitting_values[d]['train'] = []
                train_acc_values = []
                test_acc_values = []
                for n in n_estimators:
                    best_iteration_train_accuracy = 0
                    best_iteration_accuracy = 0
                    for model in splitList:
                        gb = GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=lr)
                        trnX = X[model[0]] 
                        trnY = y[model[0]]
                        tstX = X[model[1]]
                        tstY = y[model[1]]
                        gb.fit(trnX, trnY)
                        prdY = gb.predict(tstX)
                        prd_trainY = gb.predict(trnX)
                        prdY = gb.predict(tstX)

                        iteration_accuracy = metrics.accuracy_score(tstY, prdY)
                        if iteration_accuracy > best_iteration_accuracy:
                            best_iteration_accuracy = iteration_accuracy
                            best_iteration_train_accuracy = metrics.accuracy_score(trnY, prd_trainY)
                            model_sets = (trnX, trnY, tstX, tstY)
                
                    yvalues.append(best_iteration_accuracy)
                    train_acc_values.append(best_iteration_train_accuracy)
                    test_acc_values.append(best_iteration_accuracy) 
                    if yvalues[-1] > last_best:
                        best = (d, lr, n)
                        best_model = tuple(model_sets)
                        last_best = yvalues[-1]
                        best_tree = gb
                    values[lr] = yvalues
            ds.multiple_line_chart(n_estimators, values, ax=axs[0, k], title='Gradient Boorsting with max_depth=%d'%d,
                                xlabel='nr estimators', ylabel='accuracy', percentage=True)
        plt.figure()
        plt.suptitle('HFCR Gradient Boosting - ' + key + ' - parameters')
        plt.savefig(subDir + 'HFCR Gradient Boosting - ' + key + ' - parameters')
        print('Best results with depth=%d, learning rate=%1.2f and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[2], last_best))

        plt.figure()
        fig, axs = plt.subplots(1, len(learning_rate), figsize=(32, 8), squeeze=False)
        i = 0
        for k in range(len(learning_rate)):
            d = learning_rate[k]
            ds.multiple_line_chart(n_estimators, overfitting_values[d], ax=axs[0, k], title='Overfitting for dist = %s'%(d), xlabel='Gradient Boosting', ylabel='accuracy', percentage=True)
        plt.suptitle('HFCR Overfitting - Gradient Boosting')
        plt.savefig(subDir + 'HFCR Overfitting - Gradient Boosting')
        
        trnX, trnY, tstX, tstY = best_model
        prd_trn = best_tree.predict(trnX)
        prd_tst = best_tree.predict(tstX)
        ds.plot_evaluation_results(pd.unique(y), trnY, prd_trn, tstY, prd_tst)
        plt.suptitle('HFCR Gradient Boosting - ' + key + ' - Performance & Confusion matrix')
        plt.savefig(graphsDir + 'HFCR Gradient Boosting - ' + key + ' - Performance & Confusion matrix')

        plt.close("all")

