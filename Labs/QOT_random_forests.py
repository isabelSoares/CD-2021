import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
import data_preparation_functions as prepfunctions
import ds_functions as ds
import os
from datetime import datetime

graphsDir = './Results/Random Forests/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)


print('------------------------------')
print('-                            -')
print('-     QOT Random Forests     -')
print('-                            -')
print('------------------------------')

data: pd.DataFrame = pd.read_csv('../Dataset/qsar_oral_toxicity.csv', sep=';', header=None)
train, test = train_test_split(data, train_size=0.7, stratify=data[1024].values)
testDatas = {}
datas = prepfunctions.prepare_dataset(train, 1024, False, False)
for key in datas:
    testDatas[key] = test.copy()

featured_datas = prepfunctions.mask_feature_selection(datas, 1024, True, './Results/FeatureSelection/QOT Feature Selection - Features')
featured_test_datas = prepfunctions.mask_feature_selection(testDatas, 1024, True, './Results/FeatureSelection/QOT Feature Selection - Features')

best_accuracies = {}

for key in datas:
    for do_feature_eng in [False, True]:
        if (do_feature_eng):
            data = featured_datas[key]
            testData = featured_test_datas[key].copy()
            subDir = graphsDir + 'FeatureEng/' +  key + '/'
            if not os.path.exists(subDir):
                os.makedirs(subDir)
        else:
            data = datas[key]
            testData = test.copy()
            subDir = graphsDir + key + '/'
            if not os.path.exists(subDir):
                os.makedirs(subDir)

        print('QOT Random Forests - Parameters Combinations')
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(current_time, ": Key: ", key, ", feature eng: ", do_feature_eng)

        trnY: np.ndarray = data.pop(1024).values 
        trnX: np.ndarray = data.values
        tstY: np.ndarray = testData.pop(1024).values 
        tstX: np.ndarray = testData.values

        n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
        max_depths = [5, 10, 25]
        max_features = [.1, .3, .5, .7, .9, 1]
        best = ('', 0, 0)
        last_best = 0
        best_tree = None
        overfit_values = {}

        cols = len(max_depths)
        plt.figure()
        fig, axs = plt.subplots(1, cols, figsize=(cols*ds.HEIGHT*2.5, ds.HEIGHT), squeeze=False)
        for k in range(len(max_depths)):
            d = max_depths[k]
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(current_time, ": d: ", d)
            values = {}
            overfit_values[d] = {}
            for f in max_features:
                yvalues = []
                train_acc_values = []
                test_acc_values = []
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(current_time, ": f: ", f)
                for n in n_estimators:
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    print(current_time, ": n: ", n)
                    rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
                    rf.fit(trnX, trnY)
                    prdY = rf.predict(tstX)
                    prd_trainY = rf.predict(trnX)
                    yvalues.append(metrics.accuracy_score(tstY, prdY))
                    train_acc_values.append(metrics.accuracy_score(trnY, prd_trainY))
                    test_acc_values.append(metrics.accuracy_score(tstY, prdY))
                    if yvalues[-1] > last_best:
                        best = (d, f, n)
                        last_best = yvalues[-1]
                        last_best_train = train_acc_values[-1]
                        best_tree = rf

                values[f] = yvalues
                overfit_values[d][f] = {}
                overfit_values[d][f]['train'] = train_acc_values
                overfit_values[d][f]['test'] = test_acc_values
            ds.multiple_line_chart(n_estimators, values, ax=axs[0, k], title='Random Forests with max_depth=%d'%d,
                                xlabel='nr estimators', ylabel='accuracy', percentage=True)

        text = key
        if (do_feature_eng): text += ' with FS'
        best_accuracies[text] = [last_best_train, last_best]

        print('Best results with depth=%d, %1.2f features and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[2], last_best))
        fig.text(0.5, 0.03, 'Best results with depth=%d, %1.2f features and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[2], last_best), fontsize=7, ha='center', va='center')
        plt.suptitle('QOT Random Forests - ' + key + '- Parameters Combinations')
        plt.savefig(subDir + 'QOT Random Forests - ' + key + '- Parameters Combinations')
        print()

        plt.figure()
        fig, axs = plt.subplots(3, 6, figsize=(32, 16), squeeze=False)
        for k in range(len(max_depths)):
            d = max_depths[k]
            for i in range(len(max_features)):
                f = max_features[i]
                ds.multiple_line_chart(n_estimators, overfit_values[d][f], ax=axs[k,i], title='Overfitting for max_depth = %d with max_features = %f'%(d,f), xlabel='n_estimators', ylabel='accuracy', percentage=True)
        plt.suptitle('QOT Overfitting - Random Forests')
        plt.savefig(subDir + 'QOT Overfitting - Random Forests')

        print('QOT Random Forests - Performance & Confusion Matrix')
        prd_trn = best_tree.predict(trnX)
        prd_tst = best_tree.predict(tstX)
        ds.plot_evaluation_results(["negative", "positive"], trnY, prd_trn, tstY, prd_tst)
        plt.suptitle('QOT Random Forests - ' + key + ' - Performance & Confusion Matrix')
        plt.savefig(subDir + 'QOT Random Forests - ' + key + ' - Performance & Confusion Matrix')
        print()

plt.figure(figsize=(7,7))
ds.multiple_bar_chart(['Train', 'Test'], best_accuracies, ylabel='Accuracy')
plt.suptitle('QOT Sampling & Feature Selection')
plt.savefig(graphsDir + 'QOT Sampling & Feature Selection')