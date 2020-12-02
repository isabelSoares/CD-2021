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
#data = prepfunctions.new_prepare_dataset(data.copy(), 'DEATH_EVENT', False, False)

data_outliers = prepfunctions.new_prepare_dataset(data.copy(), 'DEATH_EVENT', False, True)
data_outliers_scaling = prepfunctions.new_prepare_dataset(data.copy(), 'DEATH_EVENT', True, True)
data_scaling = prepfunctions.new_prepare_dataset(data.copy(), 'DEATH_EVENT', True, False)

all_datas = [data, data_outliers, data_scaling, data_outliers_scaling]
all_datas_index = [(0, 3), (1, 5), (2,-1), (4, 6)]
all_datas_splits = [{}, {}, {}, {}, {}, {}, {}]
datas_splits_scaling_featureselection = []
c = 0
for dt in all_datas:
    y: np.ndarray = dt.copy().pop('DEATH_EVENT').values
    X: np.ndarray = dt.copy().values
    labels = [0,1]

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    trn_x_b_lst = {'Original':[], 'UnderSample':[], 'OverSample':[], 'SMOTE':[]}
    trn_y_b_lst = {'Original':[], 'UnderSample':[], 'OverSample':[], 'SMOTE':[]}
    tst_x_b_lst = {'Original':[], 'UnderSample':[], 'OverSample':[], 'SMOTE':[]}
    tst_y_b_lst = {'Original':[], 'UnderSample':[], 'OverSample':[], 'SMOTE':[]}
    trn_x_fs_lst = {'Original':[], 'UnderSample':[], 'OverSample':[], 'SMOTE':[]}
    trn_y_fs_lst = {'Original':[], 'UnderSample':[], 'OverSample':[], 'SMOTE':[]}
    tst_x_fs_lst = {'Original':[], 'UnderSample':[], 'OverSample':[], 'SMOTE':[]}
    tst_y_fs_lst = {'Original':[], 'UnderSample':[], 'OverSample':[], 'SMOTE':[]}
    for train_i, test_i in skf.split(X, y):
        trn_data = prepfunctions.data_balancing(dt.iloc[train_i].copy(), 'DEATH_EVENT')
        tst_data = {'Original':dt.iloc[test_i].copy(), 'UnderSample':dt.iloc[test_i].copy(), 'OverSample':dt.iloc[test_i].copy(), 'SMOTE':dt.iloc[test_i].copy()}

        trn_data_fs = prepfunctions.mask_feature_selection(trn_data.copy(), 'DEATH_EVENT', False, './Results/FeatureSelection/HFCR Feature Selection - Features')
        tst_data_fs = prepfunctions.mask_feature_selection(tst_data.copy(), 'DEATH_EVENT', False, './Results/FeatureSelection/HFCR Feature Selection - Features')

        for category in ['Original', 'UnderSample', 'OverSample', 'SMOTE']:
            # Train
            trn_y_b: np.ndarray = trn_data[category].pop('DEATH_EVENT').values
            trn_X_b: np.ndarray = trn_data[category].values
            # Test
            tst_y_b: np.ndarray = tst_data[category].pop('DEATH_EVENT').values
            tst_X_b: np.ndarray = tst_data[category].values

            trn_x_b_lst[category].append(trn_X_b.copy())
            trn_y_b_lst[category].append(trn_y_b.copy())
            tst_x_b_lst[category].append(tst_X_b.copy())
            tst_y_b_lst[category].append(tst_y_b.copy())

        for category in ['Original', 'UnderSample', 'OverSample', 'SMOTE']:
            # Train
            trn_y_fs: np.ndarray = trn_data_fs[category].pop('DEATH_EVENT').values
            trn_X_fs: np.ndarray = trn_data_fs[category].values
            # Test
            tst_y_fs: np.ndarray = tst_data_fs[category].pop('DEATH_EVENT').values
            tst_X_fs: np.ndarray = tst_data_fs[category].values

            trn_x_fs_lst[category].append(trn_X_fs.copy())
            trn_y_fs_lst[category].append(trn_y_fs.copy())
            tst_x_fs_lst[category].append(tst_X_fs.copy())
            tst_y_fs_lst[category].append(tst_y_fs.copy())

    final_lst_b = {}
    final_lst_fs = {}
    for category in ['Original', 'UnderSample', 'OverSample', 'SMOTE']:
        final_lst_b[category] = [trn_x_b_lst[category], trn_y_b_lst[category], tst_x_b_lst[category], tst_y_b_lst[category]]
        final_lst_fs[category] = [trn_x_fs_lst[category], trn_y_fs_lst[category], tst_x_fs_lst[category], tst_y_fs_lst[category]]

    all_datas_splits[all_datas_index[c][0]] = final_lst_b.copy()
    
    if(c == 2):
        datas_splits_scaling_featureselection = final_lst_fs.copy()
        c += 1
        continue
    all_datas_splits[all_datas_index[c][1]] = final_lst_fs.copy()

    c += 1

#datas_outliers_featureselection = prepfunctions.mask_feature_selection(datas_outliers.copy(), 'DEATH_EVENT', False, './Results/FeatureSelection/HFCR Feature Selection - Features')

#datas_outliers_scaling_featureselection = prepfunctions.mask_feature_selection(datas_outliers_scaling.copy(), 'DEATH_EVENT', False, './Results/FeatureSelection/HFCR Feature Selection - Features')

#datas_scaling_featureselection = prepfunctions.mask_feature_selection(datas_scaling.copy(), 'DEATH_EVENT', False, './Results/FeatureSelection/HFCR Feature Selection - Features')

#datas_featureselection = prepfunctions.mask_feature_selection(datas.copy(), 'DEATH_EVENT', False, './Results/FeatureSelection/HFCR Feature Selection - Features')

#all_datas = [datas, datas_outliers, datas_scaling, datas_featureselection, datas_outliers_scaling, datas_outliers_featureselection, datas_outliers_scaling_featureselection]
all_datas_names = ['', ' - No Outliers', ' - Scaling', ' - Feature Selection', ' - No Outliers & Scaling', ' - No Outliers & Feature Selection', ' - No Outliers, Scaling & Feature Selection']
provisorio_data_scaling = ' - Scaling & Feature Selection'

accuracies = {}
recalls = {}
specificities = {}
precisions = {}
f1s = {}
for key in ['Original', 'UnderSample', 'OverSample', 'SMOTE']:
    last_name = 'None'
    best_accuracy = -1
    last_accuracy = -1
    offset = 3
    count = 0
    for dt in range(len(all_datas_splits)):
        if(dt != count): continue
        #data = all_datas[dt][key]
        trn_x_lst = all_datas_splits[dt][key][0]
        trn_y_lst = all_datas_splits[dt][key][1]
        tst_x_lst = all_datas_splits[dt][key][2]
        tst_y_lst = all_datas_splits[dt][key][3]
        if(last_name == ' - Scaling' and offset == 1):
            #data = datas_scaling_featureselection.copy()[key]
            trn_x_lst = datas_splits_scaling_featureselection[key][0]
            trn_y_lst = datas_splits_scaling_featureselection[key][1]
            tst_x_lst = datas_splits_scaling_featureselection[key][2]
            tst_y_lst = datas_splits_scaling_featureselection[key][3]
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
            yvalues_specificity = []
            yvalues_precision = []
            yvalues_f1 = []
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

                    cnf_mtx_trn = metrics.confusion_matrix(trn_y, prd_trn, [0,1])
                    tn_trn, fp_trn, fn_trn, tp_trn = cnf_mtx_trn.ravel()
                    cnf_mtx_tst = metrics.confusion_matrix(tst_y, prd_tst, [0,1])
                    tn_tst, fp_tst, fn_tst, tp_tst = cnf_mtx_tst.ravel()

                    train_recall += tp_trn / (tp_trn + fn_trn)
                    test_recall += tp_tst / (tp_tst + fn_tst)

                    a_tn, a_fp, a_fn, a_tp = metrics.confusion_matrix(trn_y, prd_trn).ravel()
                    train_specificity += a_tn / (a_tn+a_fp)

                    b_tn, b_fp, b_fn, b_tp = metrics.confusion_matrix(tst_y, prd_tst).ravel()
                    test_specificity += b_tn / (b_tn+b_fp)

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
                print(test_recall)
                test_specificity /= n_splits
                train_specificity /= n_splits
                test_precision /= n_splits
                train_precision /= n_splits
                test_f1 /= n_splits
                train_f1 /= n_splits

                overfitting_values[d]['train'].append(train_accuracy)
                overfitting_values[d]['test'].append(test_accuracy)
                yvalues.append(test_accuracy)
                yvalues_recall.append(test_recall)
                yvalues_specificity.append(test_specificity)
                yvalues_precision.append(test_precision)
                yvalues_f1.append(test_f1)
                if yvalues[-1] > last_best:
                    best = (n, d)
                    last_best = yvalues[-1]
                    last_best_recall = yvalues_recall[-1]
                    last_best_specificity = yvalues_specificity[-1]
                    last_best_precision = yvalues_precision[-1]
                    last_best_f1 = yvalues_f1[-1]

                    last_train_best = train_accuracy
                    last_train_best_recall = train_recall
                    last_train_best_specificity = train_specificity
                    last_train_best_precision = train_precision
                    last_train_best_f1 = train_f1
                    best_model = (prd_trn_lst, prd_tst_lst)
            values[d] = yvalues

        if(count == 0): text = key
        else: text = last_name + ' - ' + key
        accuracies[text] = [last_train_best, last_best]
        recalls[text] = [last_train_best_recall, last_best_recall]
        specificities[text] = [last_train_best_specificity, last_best_specificity]
        precisions[text] = [last_train_best_precision, last_best_precision]
        f1s[text] = [last_train_best_f1, last_best_f1]


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
        
        ds.plot_evaluation_results_kfold([0,1], trn_y_lst, prd_trn_lst, tst_y_lst, prd_tst_lst)
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

for r in recalls:
    print(str(recalls[r]) + ' <-- ' + r)

for s in specificities:
    print(str(specificities[s]) + ' <-- ' + s)

for p in precisions:
    print(str(precisions[p]) + ' <-- ' + p)

for f in f1s:
    print(str(f1s[f]) + ' <-- ' + f)

plt.figure(figsize=(7,7))
ds.multiple_bar_chart(['Train', 'Test'], accuracies, ylabel='Accuracy')
plt.suptitle('HFCR Accuracy Comparison')
plt.savefig(graphsDir + 'HFCR Accuracy Comparison')

plt.figure(figsize=(7,7))
ds.multiple_bar_chart(['Train', 'Test'], recalls, ylabel='Recall')
plt.suptitle('HFCR Recall Comparison')
plt.savefig(graphsDir + 'HFCR Recall Comparison')

plt.figure(figsize=(7,7))
ds.multiple_bar_chart(['Train', 'Test'], specificities, ylabel='Specificity')
plt.suptitle('HFCR Specificity Comparison')
plt.savefig(graphsDir + 'HFCR Specificity Comparison')

plt.figure(figsize=(7,7))
ds.multiple_bar_chart(['Train', 'Test'], precisions, ylabel='Precision')
plt.suptitle('HFCR Precision Comparison')
plt.savefig(graphsDir + 'HFCR Precision Comparison')

plt.figure(figsize=(7,7))
ds.multiple_bar_chart(['Train', 'Test'], f1s, ylabel='F1')
plt.suptitle('HFCR F1 Comparison')
plt.savefig(graphsDir + 'HFCR F1 Comparison')