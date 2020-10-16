import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ds_functions as ds
from sklearn.model_selection import train_test_split

graphsDir = './Results/'

data: pd.DataFrame = pd.read_csv('../../Dataset/qsar_oral_toxicity.csv', sep=';', header=None)
target = 1024
positive = 1
negative = 0
values = {'Original': [len(data[data[target] == positive]), len(data[data[target] == negative])]}


y: np.ndarray = data.pop(1024).values
X: np.ndarray = data.values
labels: np.ndarray = pd.unique(y)

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

values['Train'] = [len(np.delete(trnY, np.argwhere(trnY==negative))), len(np.delete(trnY, np.argwhere(trnY==positive)))]
values['Test'] = [len(np.delete(tstY, np.argwhere(tstY==negative))), len(np.delete(tstY, np.argwhere(tstY==positive)))]

plt.figure(figsize=(7,7))
ds.multiple_bar_chart([positive, negative], values, title='Data distribution per dataset')
plt.suptitle('QOT Training Strategies')
plt.savefig(graphsDir + 'QOT Training Strategies')