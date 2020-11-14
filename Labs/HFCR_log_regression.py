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
        splitIterator = iter(skf.split(X, y))

        best_iteration_accuracy = 0
        model_sets = ([], [], [], [])
        for model in splitIterator:
            trnX = X[model[0]]
            trnY = y[model[0]]
            tstX = X[model[1]]
            tstY = y[model[1]]

            clf = LogisticRegression(random_state=RANDOM_STATE)
            clf.fit(trnX, trnY)
            prd_trn = clf.predict(trnX)
            prd_tst = clf.predict(tstX)

            iteration_accuracy = metrics.accuracy_score(tstY, prd_tst)

            if iteration_accuracy > best_iteration_accuracy:
                best_iteration_accuracy = iteration_accuracy
                model_sets = (trnY, prd_trn, tstY, prd_tst)

        ds.plot_evaluation_results(pd.unique(y), model_sets[0], model_sets[1], model_sets[2], model_sets[3])
        plt.suptitle('HFCR Log Regression - ' + key + ' - Performance & Confusion matrix')
        plt.savefig(subDir + 'HFCR Log Regression - ' + key + ' - Performance & Confusion matrix')

        plt.close("all")