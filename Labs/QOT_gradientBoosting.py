import data_preparation_functions as prepfunctions
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import ds_functions as ds
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime

graphsDir = './Results/GradientBoosting/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

data: pd.DataFrame = pd.read_csv('../Dataset/qsar_oral_toxicity.csv', sep=';', header=None)
datas = prepfunctions.prepare_dataset(data, 1024, False, False)
featured_datas = prepfunctions.mask_feature_selection(datas, 1024, True, './Results/FeatureSelection/QOT Feature Selection - Features')

for key in datas:
    if (key != "SMOTE"): continue
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

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(current_time, ": Key: ", key, ", feature eng: ", do_feature_eng)
        
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
        overfitting_values = {}
        for k in range(len(max_depths)):
            d = max_depths[k]
            values = {}
            overfitting_values[d] = {}

            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(current_time, ": D: ", d)
            for lr in learning_rate:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(current_time, ": Lr: ", lr)
                yvalues = []
                train_acc_values = []
                test_acc_values = []
                for n in n_estimators:
                    gb = GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=lr)
                    gb.fit(trnX, trnY)
                    prdY = gb.predict(tstX)
                    prd_trainY = gb.predict(trnX)

                    yvalues.append(metrics.accuracy_score(tstY, prdY))
                    train_acc_values.append(metrics.accuracy_score(trnY, prd_trainY))
                    test_acc_values.append(metrics.accuracy_score(tstY, prdY))
                    if yvalues[-1] > last_best:
                        best = (d, lr, n)
                        last_best = yvalues[-1]
                        best_tree = gb
                
                values[lr] = yvalues
                overfitting_values[d][lr] = {}
                overfitting_values[d][lr]['train'] = train_acc_values
                overfitting_values[d][lr]['test'] = test_acc_values
            ds.multiple_line_chart(n_estimators, values, ax=axs[0, k], title='Gradient Boorsting with max_depth=%d'%d,
                                xlabel='nr estimators', ylabel='accuracy', percentage=True)
        
        print('Best results with depth=%d, learning rate=%1.2f and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[2], last_best))
        fig.text(0.5, 0.03, 'Best results with depth=%d, learning rate=%1.2f and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[2], last_best), fontsize=7, ha='center', va='center')
        plt.suptitle('QOT Gradient Boosting - ' + key + ' - parameters')
        plt.savefig(subDir + 'QOT Gradient Boosting - ' + key + ' - parameters')

        plt.figure()
        fig, axs = plt.subplots(len(max_depths), len(learning_rate), figsize=(32, 8), squeeze=False)
        for i in range(len(max_depths)):
            d = max_depths[i]
            for j in range(len(learning_rate)):
                lr = learning_rate[j]
                ds.multiple_line_chart(n_estimators, overfitting_values[d][lr], ax=axs[i, j], title='Overfitting for max_depth = %d, with learning rate = %1.2f'%(d, lr), xlabel='n_estimators', ylabel='accuracy', percentage=True)
        
        plt.subplots_adjust(hspace=0.5)
        plt.suptitle('QOT Overfitting - Gradient Boosting')
        plt.savefig(subDir + 'QOT Overfitting - Gradient Boosting')
        
        prd_trn = best_tree.predict(trnX)
        prd_tst = best_tree.predict(tstX)
        ds.plot_evaluation_results(pd.unique(y), trnY, prd_trn, tstY, prd_tst)
        plt.suptitle('QOT Gradient Boosting - ' + key + ' - Performance & Confusion matrix')
        plt.savefig(subDir + 'QOT Gradient Boosting - ' + key + ' - Performance & Confusion matrix')

        plt.close("all")
