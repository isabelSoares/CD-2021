import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier
import ds_functions as ds
import os

graphsDir = './Results/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

data: pd.DataFrame = pd.read_csv('../../Dataset/heart_failure_clinical_records_dataset.csv')
y: np.ndarray = data.pop('DEATH_EVENT').values
X: np.ndarray = data.values
labels = pd.unique(y)

skf = StratifiedKFold(n_splits=5, shuffle=True)
splitList = list(skf.split(X, y))
splitCounter = 1

min_impurity_decrease = [0.025, 0.01, 0.005, 0.0025, 0.001]
max_depths = [2, 5, 10, 15, 20, 25, 30]
criteria = ['entropy', 'gini']
best = ('',  0, 0.0)
best_model = ()
last_best = 0
best_tree = None

plt.figure()
fig, axs = plt.subplots(1, 2, figsize=(16, 5), squeeze=False)
for k in range(len(criteria)):
    f = criteria[k]
    values = {}
    for d in max_depths:
        yvalues = []
        for imp in min_impurity_decrease:
            best_iteration_accuracy = 0
            for model in splitList:
                trnX, trnY = X[model[0]], y[model[0]]
                tstX, tstY = X[model[1]], y[model[1]]

                tree = DecisionTreeClassifier(min_samples_leaf=1, max_depth=d, criterion=f, min_impurity_decrease=imp)
                tree.fit(trnX, trnY)
                prdY = tree.predict(tstX)

                iteration_accuracy = metrics.accuracy_score(tstY, prdY)
                if iteration_accuracy > best_iteration_accuracy:
                    best_iteration_accuracy = iteration_accuracy
                    model_sets = (trnX, trnY, tstX, tstY)
                
            yvalues.append(best_iteration_accuracy)
            if yvalues[-1] > last_best:
                best = (f, d, imp)
                best_model = tuple(model_sets)
                last_best = yvalues[-1]
                best_tree = tree

        values[d] = yvalues
    ds.multiple_line_chart(min_impurity_decrease, values, ax=axs[0, k], title='Decision Trees with %s criteria'%f,
                           xlabel='min_impurity_decrease', ylabel='accuracy', percentage=True)

print('Best results achieved with %s criteria, depth=%d and min_impurity_decrease=%1.5f ==> accuracy=%1.5f'%(best[0], best[1], best[2], last_best))
fig.text(0.5, 0.03, 'Best results with %s criteria, depth=%d and min_impurity_decrease=%1.5f ==> accuracy=%1.5f'%(best[0], best[1], best[2], last_best), fontsize=7, ha='center', va='center')
plt.suptitle('HFCR Decision Trees - parameters')
plt.savefig(graphsDir + 'HFCR Decision Trees - parameters')

from sklearn.tree import export_graphviz
labels_features = list(data.columns.values)
dot_data = export_graphviz(best_tree, out_file=(graphsDir + 'HFCR - dtree.dot'), filled=True, rounded=True, special_characters=True, feature_names=labels_features, class_names=['0', '1'])
# Convert to png
from subprocess import call
call(['dot', '-Tpng', (graphsDir + 'HFCR - dtree.dot'), '-o', (graphsDir + 'HFCR Decision Trees - tree representation.png'), '-Gdpi=600'])

plt.figure(figsize = (14, 18))
plt.imshow(plt.imread(graphsDir + 'HFCR Decision Trees - tree representation.png'))
plt.axis('off')
plt.savefig(graphsDir + 'HFCR Decision Trees - tree representation')

trnX, trnY, tstX, tstY = best_model
prd_trn = best_tree.predict(trnX)
prd_tst = best_tree.predict(tstX)
ds.plot_evaluation_results(pd.unique(y), trnY, prd_trn, tstY, prd_tst)
plt.suptitle('HFCR Decision Trees - Performance & Confusion matrix')
plt.savefig(graphsDir + 'HFCR Decision Trees - Performance & Confusion matrix')