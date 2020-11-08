import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.ensemble import GradientBoostingClassifier
import data_preparation_functions as prepfunctions
import ds_functions as ds
import os
from datetime import datetime

graphsDir = './Results/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

data: pd.DataFrame = pd.read_csv('../../Dataset/qsar_oral_toxicity.csv', sep=';', header=None)
datas = prepfunctions.prepare_dataset(data, 1024, False, False)
featured_datas = prepfunctions.mask_feature_selection(datas, 1024, True, './Results/QOT Feature Selection - Features')

for key in datas:
    for do_feature_eng in [True]:
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

    y: np.ndarray = data.pop(1024).values
    X: np.ndarray = data.values
    labels = pd.unique(y)

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

    n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
    max_depths = [5, 10, 25]
    learning_rate = [.1, .3, .5, .7, .9]
    best = ('', 0, 0)
    last_best = 0
    best_tree = None 

    cols = len(max_depths)
    plt.figure()
    fig, axs = plt.subplots(1, cols, figsize=(cols*ds.HEIGHT, ds.HEIGHT), squeeze=False)
    for k in range(len(max_depths)):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print('1st for' + current_time)
        d = max_depths[k]
        values = {}
        overfitting_values = {}
        for lr in learning_rate:
            now = datetime.now()
            print('2nd for' + current_time)
            yvalues = []
            overfitting_values[d] = {}
            overfitting_values[d]['test'] = []
            overfitting_values[d]['train'] = []
            for n in n_estimators:
                now = datetime.now()
                print('3rd for' + current_time)
                gb = GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=lr)
                gb.fit(trnX, trnY)
                prdY = gb.predict(tstX)
                yvalues.append(metrics.accuracy_score(tstY, prdY))
                if yvalues[-1] > last_best:
                    now = datetime.now()
                    print('1st if' + current_time)
                    best = (d, lr, n)
                    last_best = yvalues[-1]
                    best_tree = gb
            values[lr] = yvalues
        ds.multiple_line_chart(n_estimators, values, ax=axs[0, k], title='Gradient Boorsting with max_depth=%d'%d,
                            xlabel='nr estimators', ylabel='accuracy', percentage=True)
    plt.figure()
    plt.suptitle('QOT Gradient Boosting - ' + key + ' - parameters')
    plt.savefig(subDir + 'QOT Gradient Boosting - ' + key + ' - parameters')
    print('Best results with depth=%d, learning rate=%1.2f and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[2], last_best))

    plt.figure()
    fig, axs = plt.subplots(1, len(learning_rate), figsize=(32, 8), squeeze=False)
    i = 0
    for k in range(len(learning_rate)):
        d = learning_rate[k]
        ds.multiple_line_chart(n_estimators, overfitting_values[d], ax=axs[0, k], title='Overfitting for dist = %s'%(d), xlabel='Gradient Boosting', ylabel='accuracy', percentage=True)
    plt.suptitle('QOT Overfitting - KNN')
    plt.savefig(subDir + 'QOT Overfitting - KNN')
    
    prd_tst = best_tree.predict(tstX)
    ds.plot_evaluation_results(pd.unique(y), trnY, prd_trn, tstY, prd_tst)
    plt.suptitle('QOT Gradient Boosting - ' + key + ' - Performance & Confusion matrix')
    plt.savefig(graphsDir + 'QOT Gradient Boosting - ' + key + ' - Performance & Confusion matrix')

    plt.close("all")
