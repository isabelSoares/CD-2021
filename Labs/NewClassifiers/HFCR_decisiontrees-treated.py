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

data: pd.DataFrame = pd.read_csv('../../Dataset/heart_failure_clinical_records_dataset.csv')
datas = prepfunctions.prepare_dataset(data.copy(), 'DEATH_EVENT', False, False)

datas_outliers = prepfunctions.prepare_dataset(data.copy(), 'DEATH_EVENT', False, True)
datas_outliers_scaling = prepfunctions.prepare_dataset(data.copy(), 'DEATH_EVENT', True, True)
datas_outliers_featureselection = prepfunctions.mask_feature_selection(datas_outliers.copy(), 'DEATH_EVENT', False, './Results/FeatureSelection/HFCR Feature Selection - Features')
datas_outliers_scaling_featureselection = prepfunctions.mask_feature_selection(datas_outliers_scaling.copy(), 'DEATH_EVENT', False, './Results/FeatureSelection/HFCR Feature Selection - Features')

datas_scaling = prepfunctions.prepare_dataset(data.copy(), 'DEATH_EVENT', True, False)
datas_scaling_featureselection = prepfunctions.mask_feature_selection(datas_scaling.copy(), 'DEATH_EVENT', False, './Results/FeatureSelection/HFCR Feature Selection - Features')

datas_featureselection = prepfunctions.mask_feature_selection(datas.copy(), 'DEATH_EVENT', False, './Results/FeatureSelection/HFCR Feature Selection - Features')

all_datas = [datas, datas_outliers, datas_scaling, datas_featureselection, datas_outliers_scaling, datas_outliers_featureselection, datas_outliers_scaling_featureselection]
all_datas_names = ['', ' - No Outliers', ' - Scaling', ' - Feature Selection', ' - No Outliers & Scaling', ' - No Outliers & Feature Selection', ' - No Outliers, Scaling & Feature Selection']
provisorio_data_scaling = ' - Scaling & Feature Selection'

accuracies = {}
for key in datas:
    last_name = 'None'
    best_accuracy = -1
    last_accuracy = -1
    offset = 3
    count = 0
    for dt in range(len(all_datas)):
        if(dt != count): continue
        data = all_datas[dt][key]
        if(last_name == ' - Scaling' and offset == 1):
            data = datas_scaling_featureselection.copy()[key]
            subDir = graphsDir + key + '/' + provisorio_data_scaling + '/'
            last_name = provisorio_data_scaling
        elif(all_datas_names[count] == ''):
            subDir = graphsDir + key + '/' + 'First' + '/'
            last_name = all_datas_names[count]
        else:
            subDir = graphsDir + key + '/' + all_datas_names[count] + '/'
            last_name = all_datas_names[count]
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
                    prd_trn_lst = []
                    prd_tst_lst = []
                    test_accuracy = 0
                    train_accuracy = 0
                    for trn_X, trn_y, tst_X, tst_y in zip(trn_x_lst, trn_y_lst, tst_x_lst, tst_y_lst):
                        tree = DecisionTreeClassifier(min_samples_leaf=1, max_depth=d, criterion=f, min_impurity_decrease=imp)
                        tree.fit(trn_X, trn_y)
                        prd_tst = tree.predict(tst_X)
                        prd_trn = tree.predict(trn_X)

                        train_accuracy += metrics.accuracy_score(trn_y, prd_trn)
                        test_accuracy += metrics.accuracy_score(tst_y, prd_tst)

                        prd_trn_lst.append(prd_trn)
                        prd_tst_lst.append(prd_tst)

                    test_accuracy /= n_splits
                    train_accuracy /= n_splits

                    yvalues.append(test_accuracy)
                    train_acc_values.append(train_accuracy)
                    test_acc_values.append(test_accuracy)  
                    if yvalues[-1] > last_best:
                        best = (f, d, imp)
                        best_model = (prd_trn_lst, prd_tst_lst)
                        last_best = yvalues[-1]
                        last_best_train = train_acc_values[-1]
                        best_tree = tree

                values[d] = yvalues
                overfit_values[f][d] = {}
                overfit_values[f][d]['train'] = train_acc_values
                overfit_values[f][d]['test'] = test_acc_values
            ds.multiple_line_chart(min_impurity_decrease, values, ax=axs[0, k], title='Decision Trees with %s criteria'%f,
                                xlabel='min_impurity_decrease', ylabel='accuracy', percentage=True)

        if(count == 0): text = key
        else: text = last_name + ' - ' + key
        accuracies[text] = [last_best_train, last_best]

        last_accuracy = last_best
        
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

        prd_trn_lst = best_model[0]
        prd_tst_lst = best_model[1]

        ds.plot_evaluation_results_kfold(pd.unique(y), trn_y_lst, prd_trn_lst, tst_y_lst, prd_tst_lst)
        plt.suptitle('HFCR Decision Trees - ' + key + ' - Performance & Confusion matrix')
        plt.savefig(subDir + 'HFCR Decision Trees - ' + key + ' - Performance & Confusion matrix')

        if(offset == 1):
            break
        if(last_accuracy > best_accuracy and best_accuracy != -1):
            best_accuracy = last_accuracy
            last_accuracy = -1
            count += offset
            offset -= 1
        elif(best_accuracy == -1):
            best_accuracy = last_accuracy
            count += 1
        else:
            count += 1
            offset -= 1


        plt.close("all")
        plt.clf()

for a in accuracies:
    print(str(accuracies[a]) + ' <-- ' + a)

plt.figure(figsize=(7,4))
ds.multiple_bar_chart(['Train', 'Test'], accuracies, ylabel='Accuracy')
plt.suptitle('HFCR Accuracy Comparison')
plt.savefig(graphsDir + 'HFCR Accuracy Comparison')