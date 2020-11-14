import os
import numpy as np
import pandas as pd
import ds_functions as ds
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import data_preparation_functions as prepfunctions

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

graphsDir = './Results/Log Regression/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('--------------------------------------')
print('-                                    -')
print('-    QOT Log Regression - Treated    -')
print('-                                    -')
print('--------------------------------------')

RANDOM_STATE = 42
data: pd.DataFrame = pd.read_csv('../Dataset/qsar_oral_toxicity.csv', sep=';', header=None)
datas = prepfunctions.prepare_dataset(data, 1024, False, False)
featured_datas = prepfunctions.mask_feature_selection(datas, 1024, True, './Results/FeatureSelection/QOT Feature Selection - Features')

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

        print('QOT Log Regression - Performance & Confusion matrix')
        y: np.ndarray = data.pop(1024).values
        X: np.ndarray = data.values
        labels = pd.unique(y)
        
        trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

        clf = LogisticRegression(random_state=RANDOM_STATE)
        clf.fit(trnX, trnY)
        prd_trn = clf.predict(trnX)
        prd_tst = clf.predict(tstX)
        ds.plot_evaluation_results(pd.unique(y), trnY, prd_trn, tstY, prd_tst)
        plt.suptitle('QOT Log Regression - ' + key + ' - Performance & Confusion matrix')
        plt.savefig(subDir + 'QOT Log Regression - ' + key + ' - Performance & Confusion matrix')

        plt.close("all")

