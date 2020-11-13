import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ds_functions as ds
import os
from sklearn.model_selection import StratifiedKFold

graphsDir = './Results/Training Strategies/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)
    
data: pd.DataFrame = pd.read_csv('../Dataset/heart_failure_clinical_records_dataset.csv')
target = 'DEATH_EVENT'
positive = 1
negative = 0
values = {'Original': [len(data[data[target] == positive]), len(data[data[target] == negative])]}

n_splits = 5

y: np.ndarray = data.pop('DEATH_EVENT').values
X: np.ndarray = data.values
labels: np.ndarray = pd.unique(y)

skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
splitIterator = iter(skf.split(X, y))
splitCounter = 1

for model in splitIterator:
    trnX = X[model[0]] 
    trnY = y[model[0]]
    tstX = X[model[1]]
    tstY = y[model[1]]

    values['Train Split ' + str(splitCounter)] = [len(np.delete(trnY, np.argwhere(trnY==negative))), len(np.delete(trnY, np.argwhere(trnY==positive)))]
    values['Test Split ' + str(splitCounter)] = [len(np.delete(tstY, np.argwhere(tstY==negative))), len(np.delete(tstY, np.argwhere(tstY==positive)))]

    splitCounter += 1

plt.figure(figsize=(7,7))
ds.multiple_bar_chart([positive, negative], values, title='Data distribution per dataset')
plt.suptitle('HFCR Training Strategies')
plt.savefig(graphsDir + 'HFCR Training Strategies')