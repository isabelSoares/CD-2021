import data_preparation_functions as prepfunctions
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import ds_functions as ds
import pandas as pd
import numpy as np
import os
import json

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime

graphsDir = './Results/GradientBoosting/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

data: pd.DataFrame = pd.read_csv('../Dataset/qsar_oral_toxicity.csv', sep=';', header=None)
datas = prepfunctions.prepare_dataset(data, 1024, False, False)
featured_datas = prepfunctions.mask_feature_selection(datas, 1024, True, './Results/FeatureSelection/QOT Feature Selection - Features')
best_accuracies = {}

for key in datas:
    if (key != 'SMOTE'): continue
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
        max_features = [None, 'auto', 'sqrt', 'log2']

        best = ('', '', 0, 0)
        last_best = 0
        best_tree = None

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
                    train_acc_values = []
                    test_acc_values = []
                    for n in n_estimators:
                        gb = GradientBoostingClassifier(max_features=max_feat, n_estimators=n, max_depth=d, learning_rate=lr)
                        gb.fit(trnX, trnY)
                        prdY = gb.predict(tstX)
                        prd_trainY = gb.predict(trnX)

                        yvalues.append(metrics.accuracy_score(tstY, prdY))
                        train_acc_values.append(metrics.accuracy_score(trnY, prd_trainY))
                        test_acc_values.append(metrics.accuracy_score(tstY, prdY))
                        if yvalues[-1] > last_best:
                            best = (max_feat_string, d, lr, n)
                            last_best = yvalues[-1]
                            last_train_best = train_acc_values[-1]
                            best_tree = gb
                    
                    values[lr] = yvalues
                    overfitting_values[max_feat_string][d][lr] = {}
                    overfitting_values[max_feat_string][d][lr]['train'] = train_acc_values
                    overfitting_values[max_feat_string][d][lr]['test'] = test_acc_values
                ds.multiple_line_chart(n_estimators, values, ax=axs[w, k], title='Gradient Boorsting with max_features=%s max_depth=%d'%(max_feat_string, d),
                                    xlabel='nr estimators', ylabel='accuracy', percentage=True)
            
        print('Best results with max_features=%s, depth=%d, learning rate=%1.2f and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[2], best[3], last_best))
        fig.text(0.5, 0.03, 'Best results with max_features=%s, depth=%d, learning rate=%1.2f and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[2], best[3], last_best), fontsize=7, ha='center', va='center')
        plt.suptitle('QOT Gradient Boosting - ' + key + ' - parameters')
        plt.savefig(subDir + 'QOT Gradient Boosting - ' + key + ' - parameters')
        
        text = key
        if (do_feature_eng): text += ' with FS'
        if ((text not in best_accuracies.keys()) or (best_accuracies[text][1] < last_best)):
            best_accuracies[text] = [last_train_best, last_best]
            f = open(graphsDir + 'best_accuracies - ' + text + '.json', 'w')
            f.write(json.dumps(best_accuracies))
            f.close()

        plt.figure()
        fig, axs = plt.subplots(len(max_depths), len(learning_rate), figsize=(32, 8), squeeze=False)
        for i in range(len(max_depths)):
            d = max_depths[i]
            for j in range(len(learning_rate)):
                lr = learning_rate[j]
                ds.multiple_line_chart(n_estimators, overfitting_values[best[0]][d][lr], ax=axs[i, j], title='Overfitting for max_depth = %d, with learning rate = %1.2f'%(d, lr), xlabel='n_estimators', ylabel='accuracy', percentage=True)
        
        plt.subplots_adjust(hspace=0.5)
        plt.suptitle('QOT Overfitting - Gradient Boosting with max_features=%s'%best[0])
        plt.savefig(subDir + 'QOT Overfitting - Gradient Boosting')
        
        prd_trn = best_tree.predict(trnX)
        prd_tst = best_tree.predict(tstX)
        ds.plot_evaluation_results(pd.unique(y), trnY, prd_trn, tstY, prd_tst)
        plt.suptitle('QOT Gradient Boosting - ' + key + ' - Performance & Confusion matrix')
        plt.savefig(subDir + 'QOT Gradient Boosting - ' + key + ' - Performance & Confusion matrix')

        plt.close("all")

plt.figure(figsize=(7,7))
ds.multiple_bar_chart(['Train', 'Test'], best_accuracies, ylabel='Accuracy')
plt.suptitle('QOT Sampling & Feature Selection')
plt.savefig(graphsDir + 'QOT Sampling & Feature Selection')