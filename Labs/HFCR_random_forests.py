import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
import data_preparation_functions as prepfunctions
import ds_functions as ds
import os

graphsDir = './Results/Random Forests/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)


print('-------------------------------')
print('-                             -')
print('-     HFCR Random Forests     -')
print('-                             -')
print('-------------------------------')



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

        print('HFCR Random Forests - Parameters Combinations')
        y: np.ndarray = data.pop('DEATH_EVENT').values
        X: np.ndarray = data.values
        labels = pd.unique(y)

        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

        n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
        max_depths = [5, 10, 25]
        max_features = [.1, .3, .5, .7, .9, 1]
        best = ('', 0, 0)
        last_best = 0
        last_best_train = 0
        best_tree = None
        best_model = None
        overfit_values = {}

        cols = len(max_depths)
        plt.figure()
        fig, axs = plt.subplots(1, cols, figsize=(cols*ds.HEIGHT*2.5, ds.HEIGHT), squeeze=False)
        for k in range(len(max_depths)):
            d = max_depths[k]
            values = {}
            overfit_values[d] = {}
            for f in max_features:
                yvalues = []
                train_acc_values = []
                test_acc_values = []
                for n in n_estimators:
                    trn_y_lst = []
                    prd_trn_lst = []
                    tst_y_lst = []
                    prd_tst_lst = []
                    test_accuracy = 0
                    train_accuracy = 0
                    for train_i, test_i in skf.split(X, y):
                        # Train
                        trn_X = X[train_i]
                        trn_y = y[train_i]

                        # Test
                        tst_X = X[test_i]
                        tst_y = y[test_i]

                        rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
                        rf.fit(trn_X, trn_y)
                        prd_trn = rf.predict(trn_X)
                        prd_tst = rf.predict(tst_X)

                        train_accuracy += metrics.accuracy_score(trn_y, prd_trn)
                        test_accuracy += metrics.accuracy_score(tst_y, prd_tst)

                        trn_y_lst.append(trn_y)
                        prd_trn_lst.append(prd_trn)
                        tst_y_lst.append(tst_y)
                        prd_tst_lst.append(prd_tst)

                    test_accuracy /= n_splits
                    train_accuracy /= n_splits

                    yvalues.append(test_accuracy)
                    train_acc_values.append(train_accuracy)
                    test_acc_values.append(test_accuracy)
                    if yvalues[-1] > last_best:
                        best = (d, f, n)
                        last_best = yvalues[-1]
                        last_best_train = train_acc_values[-1]
                        best_tree = rf
                        best_model = (trn_y_lst, prd_trn_lst, tst_y_lst, prd_tst_lst)

                values[f] = yvalues
                overfit_values[d][f]= {}
                overfit_values[d][f]['train'] = train_acc_values
                overfit_values[d][f]['test'] = test_acc_values
            ds.multiple_line_chart(n_estimators, values, ax=axs[0, k], title='Random Forests with max_depth=%d'%d,
                                xlabel='nr estimators', ylabel='accuracy', percentage=True)

        text = key
        if (do_feature_eng): text += ' with FS'
        best_accuracies[text] = [last_best_train, last_best]

        print('Best results with depth=%d, %1.2f features and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[2], last_best))
        fig.text(0.5, 0.03, 'Best results with depth=%d, %1.2f features and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[2], last_best), fontsize=7, ha='center', va='center')
        plt.suptitle('HFCR Random Forests - ' + key + ' - Parameters Combinations')
        plt.savefig(subDir + 'HFCR Random Forests - ' + key + ' - Parameters Combinations')
        print()

        plt.figure()
        fig, axs = plt.subplots(3, 6, figsize=(50, 16), squeeze=False)
        for k in range(len(max_depths)):
            d = max_depths[k]
            for i in range(len(max_features)):
                f = max_features[i]
                ds.multiple_line_chart(n_estimators, overfit_values[d][f], ax=axs[k, i], title='Overfitting for max_depth = %d with max_features = %f'%(d,f), xlabel='n_estimators', ylabel='accuracy', percentage=True)
        plt.suptitle('HFCR Overfitting - Random Forests')
        plt.savefig(subDir + 'HFCR Overfitting - Random Forests')

        print('HFCR Random Forests - Performance & Confusion Matrix')
        trn_y_lst = best_model[0]
        prd_trn_lst = best_model[1]
        tst_y_lst = best_model[2]
        prd_tst_lst = best_model[3]

        ds.plot_evaluation_results_kfold(pd.unique(y), trn_y_lst, prd_trn_lst, tst_y_lst, prd_tst_lst)
        plt.suptitle('HFCR Random Forests - ' + key + '- Performance & Confusion Matrix')
        plt.savefig(subDir + 'HFCR Random Forests - ' + key + '- Performance & Confusion Matrix')
        print()

        plt.close("all")

plt.figure(figsize=(7,7))
ds.multiple_bar_chart(['Train', 'Test'], best_accuracies, ylabel='Accuracy')
plt.suptitle('HFCR Sampling & Feature Selection')
plt.savefig(graphsDir + 'HFCR Sampling & Feature Selection')