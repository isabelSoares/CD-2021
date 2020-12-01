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
            trn_X_b: np.ndarray = trn_data.copy()[category].values
            trn_y_b: np.ndarray = trn_data.copy()[category].pop('DEATH_EVENT').values
            # Test
            tst_X_b: np.ndarray = tst_data.copy()[category].values
            tst_y_b: np.ndarray = tst_data.copy()[category].pop('DEATH_EVENT').values

            trn_x_b_lst[category].append(trn_X_b)
            trn_y_b_lst[category].append(trn_y_b)
            tst_x_b_lst[category].append(tst_X_b)
            tst_y_b_lst[category].append(tst_y_b)

        for category in ['Original', 'UnderSample', 'OverSample', 'SMOTE']:
            # Train
            trn_X_fs: np.ndarray = trn_data_fs.copy()[category].values
            trn_y_fs: np.ndarray = trn_data_fs.copy()[category].pop('DEATH_EVENT').values
            # Test
            tst_X_fs: np.ndarray = tst_data_fs.copy()[category].values
            tst_y_fs: np.ndarray = tst_data_fs.copy()[category].pop('DEATH_EVENT').values

            trn_x_fs_lst[category].append(trn_X_fs)
            trn_y_fs_lst[category].append(trn_y_fs)
            tst_x_fs_lst[category].append(tst_X_fs)
            tst_y_fs_lst[category].append(tst_y_fs)

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
            trn_X = datas_splits_scaling_featureselection[key][0]
            trn_y = datas_splits_scaling_featureselection[key][1]
            tst_X = datas_splits_scaling_featureselection[key][2]
            tst_y = datas_splits_scaling_featureselection[key][3]
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
                    prd_trn_lst = []
                    prd_tst_lst = []
                    test_accuracy = 0
                    train_accuracy = 0
                    for trn_X, trn_y, tst_X, tst_y in zip(trn_x_lst, trn_y_lst, tst_x_lst, tst_y_lst):
                        rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
                        rf.fit(trn_X, trn_y)
                        prd_trn = rf.predict(trn_X)
                        prd_tst = rf.predict(tst_X)

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
                        best = (d, f, n)
                        last_best = yvalues[-1]
                        last_best_train = train_acc_values[-1]
                        best_tree = rf
                        best_model = (prd_trn_lst, prd_tst_lst)

                values[f] = yvalues
                overfit_values[d][f]= {}
                overfit_values[d][f]['train'] = train_acc_values
                overfit_values[d][f]['test'] = test_acc_values
            ds.multiple_line_chart(n_estimators, values, ax=axs[0, k], title='Random Forests with max_depth=%d'%d,
                                xlabel='nr estimators', ylabel='accuracy', percentage=True)

        if(count == 0): text = key
        else: text = last_name + ' - ' + key
        accuracies[text] = [last_best_train, last_best]

        last_accuracy = last_best

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
        prd_trn_lst = best_model[0]
        prd_tst_lst = best_model[1]

        ds.plot_evaluation_results_kfold(pd.unique(y), trn_y_lst, prd_trn_lst, tst_y_lst, prd_tst_lst)
        plt.suptitle('HFCR Random Forests - ' + key + '- Performance & Confusion Matrix')
        plt.savefig(subDir + 'HFCR Random Forests - ' + key + '- Performance & Confusion Matrix')
        print()

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