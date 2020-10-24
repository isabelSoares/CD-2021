import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier
import ds_functions as ds
import os

graphsDir = './Results/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

data: pd.DataFrame = pd.read_csv('../../Dataset/qsar_oral_toxicity.csv', sep=';', header=None)
y: np.ndarray = data.pop(1024).values
X: np.ndarray = data.values
labels = pd.unique(y)

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

min_impurity_decrease = [0.025, 0.01, 0.005, 0.0025, 0.001]
max_depths = [50, 100, 150, 250, 550, 1100, 1600, 2200, 2600]
criteria = ['entropy', 'gini']
best = ('',  0, 0.0)
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
            tree = DecisionTreeClassifier(min_samples_leaf=1, max_depth=d, criterion=f, min_impurity_decrease=imp)
            tree.fit(trnX, trnY)
            prdY = tree.predict(tstX)
            yvalues.append(metrics.accuracy_score(tstY, prdY))
            if yvalues[-1] > last_best:
                best = (f, d, imp)
                last_best = yvalues[-1]
                best_tree = tree

        values[d] = yvalues
    ds.multiple_line_chart(min_impurity_decrease, values, ax=axs[0, k], title='Decision Trees with %s criteria'%f,
                           xlabel='min_impurity_decrease', ylabel='accuracy', percentage=True)

print('Best results achieved with %s criteria, depth=%d and min_impurity_decrease=%1.5f ==> accuracy=%1.5f'%(best[0], best[1], best[2], last_best))
fig.text(0.5, 0.03, '%s criteria, depth=%d and min_impurity_decrease=%1.5f ==> accuracy=%1.5f'%(best[0], best[1], best[2], last_best), fontsize=7, ha='center', va='center')
plt.suptitle('QOT Decision Trees - parameters')
plt.savefig(graphsDir + 'QOT Decision Trees - parameters')

from sklearn.tree import export_graphviz

dot_data = export_graphviz(tree, out_file=(graphsDir + 'dtree.dot'), filled=True, rounded=True, special_characters=True, class_names=['negative', 'positive'])
# Convert to png
from subprocess import call
call(['dot', '-Tpng', (graphsDir + 'dtree.dot'), '-o', (graphsDir + 'QOT Decision Trees - tree representation.png'), '-Gdpi=600'])

plt.figure(figsize = (14, 18))
plt.imshow(plt.imread(graphsDir + 'QOT Decision Trees - tree representation.png'))
plt.axis('off')
plt.savefig(graphsDir + 'QOT Decision Trees - tree representation')

prd_trn = best_tree.predict(trnX)
prd_tst = best_tree.predict(tstX)
ds.plot_evaluation_results(pd.unique(y), trnY, prd_trn, tstY, prd_tst)
plt.suptitle('QOT Decision Trees - Performance & Confusion matrix')
plt.savefig(graphsDir + 'QOT Decision Trees - Performance & Confusion matrix')