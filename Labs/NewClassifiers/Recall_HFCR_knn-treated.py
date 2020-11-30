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


graphsDir = './Results_Accuracy_and_Recall/KNN/'
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
rrecalls = {}
specificities = {}
precisions = {}
f1s = {}
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
            yvalues_recall = []
            overfitting_values[d] = {}
            overfitting_values[d]['test'] = []
            overfitting_values[d]['train'] = []
            for n in nvalues:
                prd_trn_lst = []
                prd_tst_lst = []
                test_accuracy = 0
                train_accuracy = 0
                test_recall = 0
                train_recall = 0
                test_specificity = 0
                train_specificity = 0
                test_precision = 0
                train_precision = 0
                test_f1 = 0
                train_f1 = 0
                for trn_X, trn_y, tst_X, tst_y in zip(trn_x_lst, trn_y_lst, tst_x_lst, tst_y_lst):
                    knn = KNeighborsClassifier(n_neighbors=n, metric=d)
                    knn.fit(trn_X, trn_y)
                    prd_trn = knn.predict(trn_X)
                    prd_tst = knn.predict(tst_X)

                    train_accuracy += metrics.accuracy_score(trn_y, prd_trn)
                    test_accuracy += metrics.accuracy_score(tst_y, prd_tst)
                    train_recall += metrics.recall_score(trn_y, prd_trn)
                    test_recall += metrics.recall_score(tst_y, prd_tst)
                    a_tn, a_fp, a_fn, a_tp = metrics.confusion_matrix(trn_y, prd_trn).ravel()
                    train_specificity = a_tn / (a_tn+a_fp)

                    b_tn, b_fp, b_fn, b_tp = metrics.confusion_matrix(tst_y, prd_tst).ravel()
                    test_specificity = b_tn / (b_tn+b_fp)

                    train_precision += metrics.precision_score(trn_y, prd_trn)
                    test_precision += metrics.precision_score(tst_y, prd_tst)

                    train_f1 += metrics.f1_score(trn_y, prd_trn)
                    test_f1 += metrics.f1_score(tst_y, prd_tst)

                    prd_trn_lst.append(prd_trn)
                    prd_tst_lst.append(prd_tst)

                test_accuracy /= n_splits
                train_accuracy /= n_splits
                test_recall /= n_splits
                train_recall /= n_splits
                
                overfitting_values[d]['train'].append(train_accuracy)
                overfitting_values[d]['test'].append(test_accuracy)
                yvalues.append(test_accuracy)
                yvalues_recall.append(test_recall)
                if yvalues[-1] > last_best:
                    best = (n, d)
                    last_best = yvalues[-1]
                    last_train_best = train_accuracy
                    last_best_recall = yvalues_recall[-1]
                    last_train_best_recall = train_recall
                    best_model = (prd_trn_lst, prd_tst_lst)
            values[d] = yvalues

        if(count == 0): text = key
        else: text = last_name + ' - ' + key
        accuracies[text] = [last_train_best, last_best]
        recalls[text] = [last_train_best_recall, last_best_recall]


        last_accuracy = last_best

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

plt.figure(figsize=(7,7))
ds.multiple_bar_chart(['Train', 'Test'], accuracies, ylabel='Accuracy')
plt.suptitle('HFCR Accuracy Comparison')
plt.savefig(graphsDir + 'HFCR Accuracy Comparison')

plt.figure(figsize=(7,7))
ds.multiple_bar_chart(['Train', 'Test'], recalls, ylabel='Recall')
plt.suptitle('HFCR Recall Comparison')
plt.savefig(graphsDir + 'HFCR Recall Comparison')