import data_preparation_functions as prepfunctions
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import ds_functions as ds
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

graphsDir = './Results/GradientBoosting/'
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
recalls = {}
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

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(current_time, ": Key: ", key, ", feature eng: ", do_feature_eng)
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

        criterions = ['friedman_mse', 'mse', 'mae']
        n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
        max_depths = [5, 10, 25]
        learning_rate = [.1, .3, .5, .7, .9]
        max_features = [None, 'auto', 'sqrt', 'log2']

        values_by_criteria = {}
        for criterion in criterions:
            criterionDir = subDir + criterion + '/'
            if not os.path.exists(criterionDir):
                os.makedirs(criterionDir)

            best = ('', '', 0, 0)
            last_best = 0
            best_tree = None

            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(current_time, ": Criterion: ", criterion)

            cols = len(max_depths)
            rows = len(max_features)
            plt.figure()
            fig, axs = plt.subplots(rows, cols, figsize=(cols*(ds.HEIGHT * 2), rows * (ds.HEIGHT + 3)), squeeze=False)
            overfitting_values = {}
            for w in range(len(max_features)):
                max_feat = max_features[w]
                max_feat_string = max_feat
                if (max_feat == None): max_feat_string = 'None'
                overfitting_values[max_feat_string] = {}

                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(current_time, ": Max Features: ", max_feat_string)

                for k in range(len(max_depths)):
                    d = max_depths[k]
                    values = {}
                    overfitting_values[max_feat_string][d] = {}

                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    print(current_time, ": D: ", d)
                    for lr in learning_rate:
                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")
                        print(current_time, ": Lr: ", lr)
                        yvalues = []
                        yvalues_recall = []
                        yvalues_specificity = []
                        yvalues_precision = []
                        yvalues_f1 = []
                        train_acc_values = []
                        train_acc_values_recall = []
                        train_acc_values_specificity = []
                        train_acc_values_precision = []
                        train_acc_values_f1 = []
                        test_acc_values = []
                        for n in n_estimators:
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
                            	gb = GradientBoostingClassifier(criterion=criterion, max_features=max_feat, n_estimators=n, max_depth=d, learning_rate=lr)
                            	gb.fit(trn_X, trn_y)
                            	prd_tst = gb.predict(tst_X)
                            	prd_trn = gb.predict(trn_X)

                            	train_accuracy += metrics.accuracy_score(trn_y, prd_trn)
                            	test_accuracy += metrics.accuracy_score(tst_y, prd_tst)

                                train_recall += metrics.recall_score(trn_y, prd_trn)
                                test_recall += metrics.recall_score(tst_y, prd_tst)

                                a_tn, a_fp, a_fn, a_tp = confusion_matrix(trn_y, prd_trn).ravel()
                                train_specificity = a_tn / (a_tn+a_fp)

                                b_tn, b_fp, b_fn, b_tp = confusion_matrix(tst_y, prd_tst).ravel()
                                test_specificity = b_tn / (b_tn+b_fp)

                                train_precision += metrics.precision_score(trn_y, prd_trn)
                                test_precision += metrics.precision_score(tst_y, prd_tst)

                                train_f1 += metrics.f1_score(trn_y, prd_trn)
                                test_f1 += metrics.f1_score(tst_y, prd_tst)

                            	prd_trn_lst.append(prd_trn)
                            	prd_tst_lst.append(prd_tst)

                            test_accuracy /= n_splits
                            train_accuracy /= n_splits

                            yvalues.append(test_accuracy)
                            yvalues_recall.append(test_accuracy)
                            yvalues_specificity.append(test_accuracy)
                            yvalues_precision.append(test_accuracy)
                            yvalues_f1.append(test_accuracy)
                            train_acc_values.append(train_accuracy)
                            test_acc_values.append(test_accuracy)
                            if yvalues[-1] > last_best:
                                best = (max_feat_string, d, lr, n)
                                best_model = (prd_trn_lst, prd_tst_lst)
                                last_best = yvalues[-1]
                                last_best_recall = yvalues_recall[-1]
                                last_best_specificity = yvalues_specificity[-1]
                                last_best_precision = yvalues_precision[-1]
                                last_best_f1 = yvalues_f1[-1]

                                last_best_train = train_acc_values[-1]
                                last_best_train_recall = train_acc_values_recall[-1]
                                last_best_train_specificity = train_acc_values_specificity[-1]
                                last_best_train_precision = train_acc_values_precision[-1]
                                last_best_train_f1 = train_acc_values_f1[-1]
                                values_by_criteria[criterion] = [last_best_train, last_best]
                                best_tree = gb
                        
                        values[lr] = yvalues
                        overfitting_values[max_feat_string][d][lr] = {}
                        overfitting_values[max_feat_string][d][lr]['train'] = train_acc_values
                        overfitting_values[max_feat_string][d][lr]['test'] = test_acc_values

                    ds.multiple_line_chart(n_estimators, values, ax=axs[w, k], title='Gradient Boorsting with max_features=%s max_depth=%d'%(max_feat_string, d),
                                        xlabel='nr estimators', ylabel='accuracy', percentage=True)
            
            print('Best results with max_features=%s, depth=%d, learning rate=%1.2f and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[2], best[3], last_best))
            fig.text(0.5, 0.03, 'Best results with max_features=%s, depth=%d, learning rate=%1.2f and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[2], best[3], last_best), fontsize=7, ha='center', va='center')
            plt.suptitle('HFCR Gradient Boosting - ' + key + ' - parameters')
            plt.savefig(criterionDir + 'HFCR Gradient Boosting - ' + key + ' - parameters')

            if(count == 0): text = key
            else: text = last_name + ' - ' + key
            if ((text not in best_accuracies.keys()) or (best_accuracies[text][1] < last_best)):
                accuracies[text] = [last_best_train, last_best]
                recalls[text] = [last_best_train_recall, last_best_recall]
                specificities[text] = [last_best_train_specificity, last_best_specificity]
                precisions[text] = [last_best_train_precision, last_best_precision]
                f1s[text] = [last_best_train_f1, last_best_f1]
                last_accuracy = last_best

            plt.figure()
            fig, axs = plt.subplots(len(max_depths), len(learning_rate), figsize=(32, 8), squeeze=False)
            for i in range(len(max_depths)):
                d = max_depths[i]
                for j in range(len(learning_rate)):
                    lr = learning_rate[j]
                    ds.multiple_line_chart(n_estimators, overfitting_values[best[0]][d][lr], ax=axs[i, j], title='Overfitting for max_depth = %d, with learning rate = %1.2f'%(d, lr), xlabel='n_estimators', ylabel='accuracy', percentage=True)
            
            plt.subplots_adjust(hspace=0.5)
            plt.suptitle('HFCR Overfitting - Gradient Boosting with max_features=%s'%best[0])
            plt.savefig(criterionDir + 'HFCR Overfitting - Gradient Boosting')
            
            prd_trn_lst = best_model[0]
            prd_tst_lst = best_model[1]

            ds.plot_evaluation_results_kfold(pd.unique(y), trn_y_lst, prd_trn_lst, tst_y_lst, prd_tst_lst)
            plt.suptitle('HFCR Gradient Boosting - ' + key + ' - Performance & Confusion matrix')
            plt.savefig(criterionDir + 'HFCR Gradient Boosting - ' + key + ' - Performance & Confusion matrix')

            plt.close("all")

        plt.figure(figsize=(7,7))
        ds.multiple_bar_chart(['Train', 'Test'], values_by_criteria, ylabel='Accuracy')
        plt.suptitle('HFCR Gradient Boosting Criteria')
        plt.savefig(subDir + 'HFCR Gradient Boosting Criteria')
    
plt.figure(figsize=(7,7))
ds.multiple_bar_chart(['Train', 'Test'], accuracies, ylabel='Accuracy')
plt.suptitle('HFCR Accuracy Comparison')
plt.savefig(graphsDir + 'HFCR Accuracy Comparison')

plt.figure(figsize=(7,7))
ds.multiple_bar_chart(['Train', 'Test'], recall, ylabel='Recall')
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