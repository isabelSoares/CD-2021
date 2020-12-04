import pandas as pd
import matplotlib.pyplot as plt
import ds_functions as ds
import os

graphsDir = './Results/Granularity/Small/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

data = pd.read_csv('../Dataset/heart_failure_clinical_records_dataset.csv')
values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
print(values)

print("Computing HFCR Granularity - 10.100.1000bins ...")
columns = data.select_dtypes(include='number').columns
bins = (10, 100, 1000)
cols = len(bins)
fig, axs = plt.subplots(1, cols, figsize=(cols*ds.HEIGHT, 1*ds.HEIGHT))
for j in range(cols):
    axs[j].set_title('Histogram for %s %d bins'%(columns[0], bins[j]))
    axs[j].set_xlabel(columns[0])
    axs[j].set_ylabel('Nr records')
    axs[j].hist(data[columns[0]].values, bins=bins[j])
fig.suptitle("HFCR Granularity - 10, 100, 1000 Bins per Variable")
plt.savefig(graphsDir + 'HFCR Granularity - 10.100.1000bins.png')

print("Computing HFCR Granularity - 2bins ...")
variables = data.select_dtypes(include='number').columns
plt.figure(figsize=(ds.HEIGHT, ds.HEIGHT))
plt.title('Histogram for %s'%variables[1])
plt.xlabel(variables[1])
plt.ylabel('nr records')
plt.hist(data[variables[1]].values, bins=2, rwidth=0.95)
plt.suptitle("HFCR Granularity - 2 Bins per Variable")
plt.savefig(graphsDir + 'HFCR Granularity - 2bins.png')