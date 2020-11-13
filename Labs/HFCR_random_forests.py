import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
import ds_functions as ds
import os

graphsDir = './Results/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)


print('-------------------------------')
print('-                             -')
print('-     HFCR Random Forests     -')
print('-                             -')
print('-------------------------------')



data: pd.DataFrame = pd.read_csv('../../Dataset/heart_failure_clinical_records_dataset.csv')

print('HFCR Random Forests - Parameters Combinations')
y: np.ndarray = data.pop('DEATH_EVENT').values
X: np.ndarray = data.values
labels = pd.unique(y)

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
max_depths = [5, 10, 25]
max_features = [.1, .3, .5, .7, .9, 1]
best = ('', 0, 0)
last_best = 0
best_tree = None
best_model = None

cols = len(max_depths)
plt.figure()
fig, axs = plt.subplots(1, cols, figsize=(cols*ds.HEIGHT*2.5, ds.HEIGHT), squeeze=False)
for k in range(len(max_depths)):
    d = max_depths[k]
    values = {}
    for f in max_features:
        yvalues = []
        for n in n_estimators:
            best_iteration_accuracy = 0
            model_sets = ([], [], [], [])
            splitIterator = iter(skf.split(X, y))
            for model in splitIterator:
                trnX = X[model[0]]
                trnY = y[model[0]]
                tstX = X[model[1]]
                tstY = y[model[1]]

                rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
                rf.fit(trnX, trnY)
                prdY = rf.predict(tstX)

                iteration_accuracy = metrics.accuracy_score(tstY, prdY)

                if iteration_accuracy > best_iteration_accuracy:
                    best_iteration_accuracy = iteration_accuracy
                    model_sets = (trnX, trnY, tstX, tstY)

            yvalues.append(best_iteration_accuracy)
            if yvalues[-1] > last_best:
                best = (d, f, n)
                last_best = yvalues[-1]
                best_tree = rf
                best_model = model_sets

        values[f] = yvalues
    ds.multiple_line_chart(n_estimators, values, ax=axs[0, k], title='Random Forests with max_depth=%d'%d,
                           xlabel='nr estimators', ylabel='accuracy', percentage=True)

plt.suptitle('HFCR Random Forests - Parameters Combinations')
plt.savefig(graphsDir + 'HFCR Random Forests - Parameters Combinations')
print('Best results with depth=%d, %1.2f features and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[2], last_best))
print()

print('HFCR Random Forests - Performance & Confusion Matrix')
trnX = best_model[0]
trnY = best_model[1]
tstX = best_model[2]
tstY = best_model[3]
prd_trn = best_tree.predict(trnX)
prd_tst = best_tree.predict(tstX)
ds.plot_evaluation_results(pd.unique(y), trnY, prd_trn, tstY, prd_tst)
plt.suptitle('HFCR Random Forests - Performance & Confusion Matrix')
plt.savefig(graphsDir + 'HFCR Random Forests - Performance & Confusion Matrix')
print()