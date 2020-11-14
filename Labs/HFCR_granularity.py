import pandas as pd
import matplotlib.pyplot as plt
import ds_functions as ds
import os

graphsDir = './Results/Granularity/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

data = pd.read_csv('../Dataset/heart_failure_clinical_records_dataset.csv')
values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
print(values)

print("Computing HFCR Granularity - 100bins ...")
variables = data.select_dtypes(include='number').columns
rows, cols = ds.choose_grid(len(variables))
fig, axs = plt.subplots(rows, cols, figsize=(cols*ds.HEIGHT, rows*ds.HEIGHT))
i, j = 0, 0
for n in range(len(variables)):
    axs[i, j].set_title('Histogram for %s'%variables[n], color='chocolate')
    axs[i, j].set_xlabel(variables[n])
    axs[i, j].set_ylabel('nr records')
    axs[i, j].hist(data[variables[n]].values, bins=100, color='peachpuff')
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
fig.suptitle("HFCR Granularity - 100 Bins per Variable", color='chocolate')
plt.savefig(graphsDir + 'HFCR Granularity - 100bins.png')

print("Computing HFCR Granularity - 10.100.1000bins ...")
columns = data.select_dtypes(include='number').columns
rows = len(columns)
bins = (10, 100, 1000)
cols = len(bins)
fig, axs = plt.subplots(rows, cols, figsize=(cols*ds.HEIGHT, rows*ds.HEIGHT))
for i in range(rows):
    for j in range(cols):
        axs[i, j].set_title('Histogram for %s %d bins'%(columns[i], bins[j]), color='chocolate')
        axs[i, j].set_xlabel(columns[i])
        axs[i, j].set_ylabel('Nr records')
        axs[i, j].hist(data[columns[i]].values, bins=bins[j], color='peachpuff')
fig.suptitle("HFCR Granularity - 10, 100, 1000 Bins per Variable", color='chocolate')
plt.savefig(graphsDir + 'HFCR Granularity - 10.100.1000bins.png')

print("Computing HFCR Granularity - BestFit bins ...")
variables = data.select_dtypes(include='number').columns
bins = [100, 10, 1000, 10, 100, 10, 100, 100, 100, 10, 10, 100, 10]
rows, cols = ds.choose_grid(len(variables))
fig, axs = plt.subplots(rows, cols, figsize=(cols*ds.HEIGHT, rows*ds.HEIGHT))
i, j = 0, 0
for n in range(len(variables)):
    axs[i, j].set_title('Histogram for %s - %d bins'%(variables[n], bins[n]), color='chocolate')
    axs[i, j].set_xlabel(variables[n])
    axs[i, j].set_ylabel('nr records')
    axs[i, j].hist(data[variables[n]].values, bins=bins[n], color='peachpuff')
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
fig.suptitle("HFCR Granularity - Best Fit Bins per Variable", color='chocolate')
plt.savefig(graphsDir + 'HFCR Granularity - BestFit bins.png')

print("Finished Calculations for HFCR Granularity.")