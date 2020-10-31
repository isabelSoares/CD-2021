import matplotlib.pyplot as plt
import ds_functions as ds
import pandas as pd
import os

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2

graphsDir = './Results/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

def getData():
    data: pd.DataFrame = pd.read_csv('../../Dataset/qsar_oral_toxicity.csv', sep=';', header=None)
    y = data.pop(1024)
    return data, y

data, y = getData()
labels = ['Original']
values = [data.shape[1]]

data, y = getData()
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(data)
labels += ['VarianceThreshold']
values += [data.shape[1]]

percentile = 80
data, y = getData()
data_new = SelectPercentile(chi2, percentile).fit_transform(data, y)
labels += ['%d%% with Chi2'%percentile]
values += [data_new.shape[1]]

plt.figure()
plt.title('Nr of Features')
bars = plt.bar(labels, values)

i = 0
for rect in bars:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % values[i], ha='center', va='bottom', fontsize=7)
    i += 1

plt.savefig(graphsDir + 'QOT Feature Selection')