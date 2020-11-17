import os
import numpy as np
import pandas as pd
import ds_functions as ds
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import data_preparation_functions as prepfunctions

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

graphsDir = './Results/Log Regression/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('--------------------------------------')
print('-                                    -')
print('-   HFCR Log Regression - Treated    -')
print('-                                    -')
print('--------------------------------------')

RANDOM_STATE = 42
data: pd.DataFrame = pd.read_csv('../Dataset/heart_failure_clinical_records_dataset.csv')
datas = prepfunctions.prepare_dataset(data, 'DEATH_EVENT', False, True)
featured_datas = prepfunctions.mask_feature_selection(datas, 'DEATH_EVENT', False, './Results/FeatureSelection/HFCR Feature Selection - Features')
accuracies = {}

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

        print('HFCR Log Regression - Performance & Confusion matrix')
        y: np.ndarray = data.pop('DEATH_EVENT').values
        X: np.ndarray = data.values
        labels = pd.unique(y)
        
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

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

            clf = LogisticRegression(random_state=RANDOM_STATE)
            clf.fit(trn_X, trn_y)
            prd_trn = clf.predict(trn_X)
            prd_tst = clf.predict(tst_X)

            train_accuracy += metrics.accuracy_score(trn_y, prd_trn)
            test_accuracy += metrics.accuracy_score(tst_y, prd_tst)

            trn_y_lst.append(trn_y)
            prd_trn_lst.append(prd_trn)
            tst_y_lst.append(tst_y)
            prd_tst_lst.append(prd_tst)

        text = key
        if (do_feature_eng): text += ' with FS'
        accuracies[text] = [train_accuracy/n_splits, test_accuracy/n_splits]

        ds.plot_evaluation_results_kfold(pd.unique(y), trn_y_lst, prd_trn_lst, tst_y_lst, prd_tst_lst)
        plt.suptitle('HFCR Log Regression - ' + key + ' - Performance & Confusion matrix')
        plt.savefig(subDir + 'HFCR Log Regression - ' + key + ' - Performance & Confusion matrix')

        plt.close("all")

plt.figure(figsize=(7,7))
ds.multiple_bar_chart(['Train', 'Test'], accuracies, ylabel='Accuracy')
plt.suptitle('HFCR Sampling & Feature Selection')
plt.savefig(graphsDir + 'HFCR Sampling & Feature Selection')