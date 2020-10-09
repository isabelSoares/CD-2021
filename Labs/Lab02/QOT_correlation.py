import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import config as cfg
import os
import seaborn as sns

register_matplotlib_converters()
graphsDir = './Results/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)
print('---------------------------')
print('-                         -')
print('-     QOT Correlation     -')
print('-                         -')
print('---------------------------')

data = pd.read_csv('../../Dataset/qsar_oral_toxicity.csv', sep=';', header=None)
sample = data.sample(10)

print('QOT Correlation analysis')
fig = plt.figure(figsize=[70, 70])
corr_mtx = sample.corr()
sns.heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
plt.title('QOT Correlation analysis')
plt.savefig(graphsDir + 'QOT Correlation analysis')
