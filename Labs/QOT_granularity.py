import pandas as pd
import matplotlib.pyplot as plt
import ds_functions as ds
import os

graphsDir = './Results/Granularity/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

data = pd.read_csv('../Dataset/qsar_oral_toxicity.csv', sep=';', header=None)
values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
print(values)

print("Computing QOT Granularity - 10bins ...")
variables = data.select_dtypes(include='number').columns
rows, cols = ds.choose_grid(len(variables), 25)
fig, axs = plt.subplots(rows, cols, figsize=(cols*ds.HEIGHT, rows*ds.HEIGHT))
i, j = 0, 0
for n in range(len(variables)):
    axs[i, j].set_title('Histogram for %s'%variables[n], color='chocolate')
    axs[i, j].set_xlabel(variables[n])
    axs[i, j].set_ylabel('nr records')
    axs[i, j].hist(data[variables[n]].values, bins=10, color='peachpuff')
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
fig.suptitle("QOT Granularity - 10 Bins per Variable", color='chocolate')
plt.savefig(graphsDir + 'QOT Granularity - 10bins.png')

print("Finished Calculations for QOT Granularity.")