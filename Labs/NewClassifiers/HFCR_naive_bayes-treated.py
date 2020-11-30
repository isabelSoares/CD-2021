import os
import numpy as np
import pandas as pd
import ds_functions as ds
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import data_preparation_functions as prepfunctions
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import StratifiedKFold

graphsDir = './Results/Naive Bayes/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('--------------------------------------')
print('-                                    -')
print('-     HFCR Naive Bayes - Treated     -')
print('-                                    -')
print('--------------------------------------')

data: pd.DataFrame = pd.read_csv('../../Dataset/heart_failure_clinical_records_dataset.csv')
datas = prepfunctions.prepare_dataset(data.copy(), 'DEATH_EVENT', False, False)
datas_outliers = prepfunctions.prepare_dataset(data.copy(), 'DEATH_EVENT', False, True)
#datas_outliers_scaling = prepfunctions.prepare_dataset(data.copy(), 'DEATH_EVENT', True, True)
#datas_outliers_scaling_featureselection = prepfunctions.mask_feature_selection(datas_outliers_scaling.copy(), 'DEATH_EVENT', False, './Results/FeatureSelection/HFCR Feature Selection - Features')
datas_featureselection = prepfunctions.mask_feature_selection(datas.copy(), 'DEATH_EVENT', False, './Results/FeatureSelection/HFCR Feature Selection - Features')
datas_outliers_featureselection = prepfunctions.mask_feature_selection(datas_outliers.copy(), 'DEATH_EVENT', False, './Results/FeatureSelection/HFCR Feature Selection - Features')

all_datas = [datas, datas_outliers, datas_featureselection, datas_outliers_featureselection]
all_datas_names = ['', ' - No Outliers', ' - Feature Selection', ' - No Outliers & Feature Selection']

#offset = 3
#all_datas = [datas, datas_outliers, datas_scaling, datas_featureselection, datas_outliers_scaling, datas_outliers_featureselection, datas_outliers_scaling_featureselection]
#if(offset == 1): break
#if(>): count += offset; offset -= 1
#else: count += 1

accuracies = {}
for key in datas:
    best_accuracy = -1
    last_accuracy = -1
    offset = 2
    count = 0
    for d in range(len(all_datas)):
        if(d != count): continue
        data = all_datas[d][key]
        if(all_datas_names[count] == ''):
            subDir = graphsDir + key + '/' + 'First' + '/'
        else:
            subDir = graphsDir + key + '/' + all_datas_names[count] + '/'
        if not os.path.exists(subDir):
            os.makedirs(subDir)

        print('HFCR Naive Bayes - Performance & Confusion matrix')
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

        clf = GaussianNB()
        prd_trn_lst = []
        prd_tst_lst = []
        test_accuracy = 0
        train_accuracy = 0
        for trn_X, trn_y, tst_X, tst_y in zip(trn_x_lst, trn_y_lst, tst_x_lst, tst_y_lst):
            clf.fit(trn_X, trn_y)
            prd_trn = clf.predict(trn_X)
            prd_tst = clf.predict(tst_X)

            train_accuracy += metrics.accuracy_score(trn_y, prd_trn)
            test_accuracy += metrics.accuracy_score(tst_y, prd_tst)

            prd_trn_lst.append(prd_trn)
            prd_tst_lst.append(prd_tst)

        if(count == 0): text = key
        else: text = all_datas_names[count] + ' - ' + key
        accuracies[text] = [train_accuracy/n_splits, test_accuracy/n_splits]

        last_accuracy = test_accuracy/n_splits

        ds.plot_evaluation_results_kfold(pd.unique(y), trn_y_lst, prd_trn_lst, tst_y_lst, prd_tst_lst)
        plt.suptitle('HFCR Naive Bayes - ' + key + ' - Performance & Confusion matrix')
        plt.savefig(subDir + 'HFCR Naive Bayes - ' + key + ' - Performance & Confusion matrix')


        print('HFCR Naive Bayes - Comparison of Naive Bayes Models')
        estimators = {#'GaussianNB': GaussianNB(),
              'MultinomialNB': MultinomialNB(),
              'BernoulyNB': BernoulliNB()}

        xvalues = []
        yvalues = []
        xvalues.append('GaussianNB')
        yvalues.append(test_accuracy/n_splits)
        accuracy = 0
        for clf in estimators:
            xvalues.append(clf)
            for trn_X, trn_y, tst_X, tst_y in zip(trn_x_lst, trn_y_lst, tst_x_lst, tst_y_lst):
                estimators[clf].fit(trn_X, trn_y)
                prd_tst = estimators[clf].predict(tst_X)

                accuracy += metrics.accuracy_score(tst_y, prd_tst)

            yvalues.append(accuracy/n_splits)

        plt.figure(figsize=(7,7))
        ds.bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
        plt.suptitle(subDir + 'HFCR Naive Bayes - ' + key + ' - Comparison of Naive Bayes Models')
        plt.savefig(subDir + 'HFCR Naive Bayes - ' + key + ' - Comparison of Naive Bayes Models')

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