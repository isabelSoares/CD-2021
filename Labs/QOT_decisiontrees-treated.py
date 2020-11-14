import data_preparation_functions as prepfunctions
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import ds_functions as ds
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from subprocess import call

graphsDir = './Results/DecisionTrees/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

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

        y: np.ndarray = data.pop(1024).values
        X: np.ndarray = data.values
        labels = pd.unique(y)

        trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

        min_impurity_decrease = [0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001, 0.00005, 0.000025]
        max_depths = [2, 5, 10, 15, 20, 25, 30]
        criteria = ['entropy', 'gini']
        best = ('',  0, 0.0)
        last_best = 0
        best_tree = None
        overfit_values = {}

        plt.figure()
        fig, axs = plt.subplots(1, 2, figsize=(16, 5), squeeze=False)
        for k in range(len(criteria)):
            f = criteria[k]
            values = {}
            overfit_values[f] = {}
            for d in max_depths:
                yvalues = []
                train_acc_values = []
                test_acc_values = []
                for imp in min_impurity_decrease:
                    tree = DecisionTreeClassifier(min_samples_leaf=1, max_depth=d, criterion=f, min_impurity_decrease=imp)
                    tree.fit(trnX, trnY)
                    prdY = tree.predict(tstX)
                    prd_trainY = tree.predict(trnX)
                    yvalues.append(metrics.accuracy_score(tstY, prdY))
                    train_acc_values.append(metrics.accuracy_score(trnY, prd_trainY))
                    test_acc_values.append(metrics.accuracy_score(tstY, prdY))
                    if yvalues[-1] > last_best:
                        best = (f, d, imp)
                        last_best = yvalues[-1]
                        best_tree = tree

                values[d] = yvalues
                overfit_values[f][d] = {}
                overfit_values[f][d]['train'] = train_acc_values
                overfit_values[f][d]['test'] = test_acc_values
            ds.multiple_line_chart(min_impurity_decrease, values, ax=axs[0, k], title='Decision Trees with %s criteria'%f,
                                xlabel='min_impurity_decrease', ylabel='accuracy', percentage=True)

        print('Best results achieved with %s criteria, depth=%d and min_impurity_decrease=%1.5f ==> accuracy=%1.5f'%(best[0], best[1], best[2], last_best))
        fig.text(0.5, 0.03, 'Best results with %s criteria, depth=%d and min_impurity_decrease=%1.5f ==> accuracy=%1.5f'%(best[0], best[1], best[2], last_best), fontsize=7, ha='center', va='center')
        plt.suptitle('QOT Decision Trees - ' + key + ' - parameters')
        plt.savefig(subDir + 'QOT Decision Trees - ' + key + ' - parameters')

        plt.figure()
        fig, axs = plt.subplots(4, 4, figsize=(32, 16), squeeze=False)
        i = 0
        for k in range(len(criteria)):
            f = criteria[k]
            for d in max_depths:
                ds.multiple_line_chart(min_impurity_decrease, overfit_values[f][d], ax=axs[i // 4, i % 4], title='Overfitting for max_depth = %d with %s criteria'%(d, f), xlabel='min_impurity_decrease', ylabel='accuracy', percentage=True)
                i += 1
            i += 1
        plt.suptitle('QOT Overfitting - Decision Trees')
        plt.savefig(subDir + 'QOT Overfitting - Decision Trees')

        dot_data = export_graphviz(best_tree, out_file=(subDir + 'QOT - ' + key + ' - dtree.dot'), filled=True, rounded=True, special_characters=True, class_names=['negative', 'positive'])
        # Convert to png
        call(['dot', '-Tpng', (subDir + 'QOT - ' + key + ' - dtree.dot'), '-o', (subDir + 'QOT Decision Trees - ' + key + ' - tree representation.png'), '-Gdpi=600'])

        prd_trn = best_tree.predict(trnX)
        prd_tst = best_tree.predict(tstX)
        ds.plot_evaluation_results(pd.unique(y), trnY, prd_trn, tstY, prd_tst)
        plt.suptitle('QOT Decision Trees - ' + key + ' - Performance & Confusion matrix')
        plt.savefig(subDir + 'QOT Decision Trees - ' + key + ' - Performance & Confusion matrix')

        plt.close("all")