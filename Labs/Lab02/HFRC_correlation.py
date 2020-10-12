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
print('----------------------------')
print('-                          -')
print('-     HFRC Correlation     -')
print('-                          -')
print('----------------------------')

data = pd.read_csv('../../Dataset/heart_failure_clinical_records_dataset.csv')

print('HFRC Correlation analysis')
corr_mtx = data.corr()
print(corr_mtx)
corr_mtx.to_csv(graphsDir + 'HFRC Correlation analysis.csv')

print('Starting ploting without values')
fig = plt.figure(figsize=[12, 12])
sns.heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=False, cmap='Blues')
plt.title('HFRC Correlation analysis')
plt.tight_layout()
print("Saving figure file")
plt.savefig(graphsDir + 'HFRC Correlation analysis without values', dpi=150)
plt.close()

print('Starting ploting with values')
fig = plt.figure(figsize=[12, 12])
sns.heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
plt.title('HFRC Correlation analysis')
plt.tight_layout()
print("Saving figure file")
plt.savefig(graphsDir + 'HFRC Correlation analysis with values', dpi=150)
plt.close()
