import data_preparation_functions as prepfunctions
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import ds_functions as ds
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from subprocess import call

graphsDir = './Results/DecisionTrees/'
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

        min_impurity_decrease = [0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001, 0.00005, 0.000025]
        max_depths = [2, 5, 10, 15, 20, 25, 30]
        criteria = ['entropy', 'gini']
        best = ('',  0, 0.0)
        best_model = None
        last_best = 0
        last_best_train = 0
        best_tree = None
        overfit_values = {}

        plt.figure()
        fig, axs = plt.subplots(1, 2, figsize=(16, 5), squeeze=False)
        for k in range(len(criteria)):
            f = criteria[k]
            values = {} 
            overfit_values[f] = {}
            for d in max_depths:
                yvalues = []
                train_acc_values = []
                test_acc_values = []
                for imp in min_impurity_decrease:
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

                        tree = DecisionTreeClassifier(min_samples_leaf=1, max_depth=d, criterion=f, min_impurity_decrease=imp)
                        tree.fit(trn_X, trn_y)
                        prd_tst = tree.predict(tst_X)
                        prd_trn = tree.predict(trn_X)

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
                        best = (f, d, imp)
                        best_model = (trn_y_lst, prd_trn_lst, tst_y_lst, prd_tst_lst)
                        last_best = yvalues[-1]
                        last_best_train = train_acc_values[-1]
                        best_tree = tree

                values[d] = yvalues
                overfit_values[f][d] = {}
                overfit_values[f][d]['train'] = train_acc_values
                overfit_values[f][d]['test'] = test_acc_values
            ds.multiple_line_chart(min_impurity_decrease, values, ax=axs[0, k], title='Decision Trees with %s criteria'%f,
                                xlabel='min_impurity_decrease', ylabel='accuracy', percentage=True)

        text = key
        if (do_feature_eng): text += ' with FS'
        best_accuracies[text] = [last_best_train, last_best]
        
        print('Best results achieved with %s criteria, depth=%d and min_impurity_decrease=%1.5f ==> accuracy=%1.5f'%(best[0], best[1], best[2], last_best))
        fig.text(0.5, 0.03, 'Best results with %s criteria, depth=%d and min_impurity_decrease=%1.5f ==> accuracy=%1.5f'%(best[0], best[1], best[2], last_best), fontsize=7, ha='center', va='center')
        plt.suptitle('HFCR Decision Trees - ' + key + ' - parameters')
        plt.savefig(subDir + 'HFCR Decision Trees - ' + key + ' - parameters')

        plt.figure()
        fig, axs = plt.subplots(4, 4, figsize=(32, 16), squeeze=False)
        i = 0
        for k in range(len(criteria)):
            f = criteria[k]
            for d in max_depths:
                ds.multiple_line_chart(min_impurity_decrease, overfit_values[f][d], ax=axs[i // 4, i % 4], title='Overfitting for max_depth = %d with %s criteria'%(d, f), xlabel='min_impurity_decrease', ylabel='accuracy', percentage=True)
                i += 1
            i += 1
        plt.suptitle('HFCR Overfitting - Decision Trees')
        plt.savefig(subDir + 'HFCR Overfitting - Decision Trees')
        
        labels_features = list(data.columns.values)
        dot_data = export_graphviz(best_tree, out_file=(subDir + 'HFCR - ' + key + ' - dtree.dot'), filled=True, rounded=True, special_characters=True, feature_names=labels_features, class_names=['0', '1'])
        # Convert to png
        call(['dot', '-Tpng', (subDir + 'HFCR - ' + key + ' - dtree.dot'), '-o', (subDir + 'HFCR Decision Trees - ' + key + ' - tree representation.png'), '-Gdpi=600'])

        trn_y_lst = best_model[0]
        prd_trn_lst = best_model[1]
        tst_y_lst = best_model[2]
        prd_tst_lst = best_model[3]

        ds.plot_evaluation_results_kfold(pd.unique(y), trn_y_lst, prd_trn_lst, tst_y_lst, prd_tst_lst)
        plt.suptitle('HFCR Decision Trees - ' + key + ' - Performance & Confusion matrix')
        plt.savefig(subDir + 'HFCR Decision Trees - ' + key + ' - Performance & Confusion matrix')

        plt.close("all")

plt.figure(figsize=(7,7))
ds.multiple_bar_chart(['Train', 'Test'], best_accuracies, ylabel='Accuracy')
plt.suptitle('HFCR Sampling & Feature Selection')
plt.savefig(graphsDir + 'HFCR Sampling & Feature Selection')