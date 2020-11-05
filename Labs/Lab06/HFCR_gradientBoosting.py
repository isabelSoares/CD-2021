import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as metrics
from sklearn.ensemble import GradientBoostingClassifier
import ds_functions as ds
import os

graphsDir = './Results/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

data: pd.DataFrame = pd.read_csv('../../Dataset/heart_failure_clinical_records_dataset.csv')
y: np.ndarray = data.pop('DEATH_EVENT').values
X: np.ndarray = data.values
labels = pd.unique(y)

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
splitList = list(skf.split(X, y))

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
    d = max_depths[k]
    values = {}
    for lr in learning_rate:
        yvalues = []
        train_acc_values = []
        test_acc_values = []
        for n in n_estimators:
            best_iteration_train_accuracy = 0
            best_iteration_accuracy = 0
            for model in splitList:
                gb = GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=lr)
                trnX = X[model[0]] 
                trnY = y[model[0]]
                tstX = X[model[1]]
                tstY = y[model[1]]
                gb.fit(trnX, trnY)
                prdY = gb.predict(tstX)
                prd_trainY = gb.predict(trnX)
                prdY = gb.predict(tstX)

                iteration_accuracy = metrics.accuracy_score(tstY, prdY)
                if iteration_accuracy > best_iteration_accuracy:
                    best_iteration_accuracy = iteration_accuracy
                    best_iteration_train_accuracy = metrics.accuracy_score(trnY, prd_trainY)
                    model_sets = (trnX, trnY, tstX, tstY)
        
            yvalues.append(best_iteration_accuracy)
            train_acc_values.append(best_iteration_train_accuracy)
            test_acc_values.append(best_iteration_accuracy) 
            if yvalues[-1] > last_best:
                best = (d, lr, n)
                best_model = tuple(model_sets)
                last_best = yvalues[-1]
                best_tree = gb
            values[lr] = yvalues
    ds.multiple_line_chart(n_estimators, values, ax=axs[0, k], title='Gradient Boorsting with max_depth=%d'%d,
                           xlabel='nr estimators', ylabel='accuracy', percentage=True)
plt.figure()
print('Best results with depth=%d, learning rate=%1.2f and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[2], last_best))

trnX, trnY, tstX, tstY = best_model
prd_trn = best_tree.predict(trnX)
prd_tst = best_tree.predict(tstX)
ds.plot_evaluation_results(pd.unique(y), trnY, prd_trn, tstY, prd_tst)
plt.suptitle('HFCR Gradient Boosting - Performance & Confusion matrix')
plt.savefig(graphsDir + 'HFCR Gradient Boosting - Performance & Confusion matrix')


