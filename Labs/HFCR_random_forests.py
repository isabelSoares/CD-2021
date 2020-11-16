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
                    best_iteration_train_accuracy = 0
                    best_iteration_accuracy = 0
                    model_sets = ([], [], [], [])
                    splitIterator = iter(skf.split(X, y))
                    for model in splitIterator:
                        trnX = X[model[0]]
                        trnY = y[model[0]]
                        tstX = X[model[1]]
                        tstY = y[model[1]]

                        rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
                        rf.fit(trnX, trnY)
                        prdY = rf.predict(tstX)
                        prd_trainY = rf.predict(trnX)
                    
                        iteration_accuracy = metrics.accuracy_score(tstY, prdY)

                        if iteration_accuracy > best_iteration_accuracy:
                            best_iteration_accuracy = iteration_accuracy
                            best_iteration_train_accuracy = metrics.accuracy_score(trnY, prd_trainY)
                            model_sets = (trnX, trnY, tstX, tstY, prd_trainY, prdY)

                    yvalues.append(best_iteration_accuracy)
                    train_acc_values.append(best_iteration_train_accuracy)
                    test_acc_values.append(best_iteration_accuracy)
                    if yvalues[-1] > last_best:
                        best = (d, f, n)
                        last_best = yvalues[-1]
                        last_best_train = train_acc_values[-1]
                        best_tree = rf
                        best_model = model_sets

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
        trnX = best_model[0]
        trnY = best_model[1]
        tstX = best_model[2]
        tstY = best_model[3]
        prd_trn = best_model[4]
        prd_tst = best_model[5]
        ds.plot_evaluation_results(pd.unique(y), trnY, prd_trn, tstY, prd_tst)
        plt.suptitle('HFCR Random Forests - ' + key + '- Performance & Confusion Matrix')
        plt.savefig(subDir + 'HFCR Random Forests - ' + key + '- Performance & Confusion Matrix')
        print()

        plt.close("all")

plt.figure(figsize=(7,7))
ds.multiple_bar_chart(['Train', 'Test'], best_accuracies, ylabel='Accuracy')
plt.suptitle('HFCR Sampling & Feature Selection')
plt.savefig(graphsDir + 'HFCR Sampling & Feature Selection')