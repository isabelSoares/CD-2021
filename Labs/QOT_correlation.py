import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import config as cfg
import os
import seaborn as sns
import numpy as np

register_matplotlib_converters()

graphsDir = './Results/Correlation/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('---------------------------')
print('-                         -')
print('-     QOT Correlation     -')
print('-                         -')
print('---------------------------')

data = pd.read_csv('../Dataset/qsar_oral_toxicity.csv', sep=';', header=None)

print('QOT Correlation analysis')
corr_mtx = data.corr()
print(corr_mtx)
corr_mtx.to_csv(graphsDir + 'QOT Correlation analysis.csv')

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr_mtx, dtype=bool))

print('Starting ploting without values')
fig = plt.figure(figsize=[210, 210])
sns.heatmap(corr_mtx, mask=mask, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=False, cmap='Blues')
plt.title('QOT Correlation analysis')
plt.tight_layout()
print("Saving figure file")
plt.savefig(graphsDir + 'QOT Correlation analysis without values', dpi=150)
plt.close()

print('Starting ploting with values')
fig = plt.figure(figsize=[210, 210])
sns.heatmap(corr_mtx, mask=mask, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, annot_kws={"size": 3}, cmap='Blues')
plt.title('QOT Correlation analysis')
plt.tight_layout()
print("Saving figure file")
plt.savefig(graphsDir + 'QOT Correlation analysis with values', dpi=150)
plt.close()
