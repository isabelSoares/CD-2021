import pandas as pd
import matplotlib.pyplot as plt
import ds_functions as ds
import os

graphsDir = './Results/Granularity/Small/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

data = pd.read_csv('../Dataset/qsar_oral_toxicity.csv', sep=';', header=None)
values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
print(values)

print("Computing QOT Granularity - 2bins ...")
variables = data.select_dtypes(include='number').columns
plt.figure(figsize=(ds.HEIGHT, ds.HEIGHT))
plt.title('Histogram for %s'%variables[0])
plt.xlabel(variables[0])
plt.ylabel('nr records')
plt.hist(data[variables[0]].values, bins=2, rwidth=0.95)
plt.suptitle("QOT Granularity - 2 Bins per Variable")
plt.savefig(graphsDir + 'QOT Granularity - 2bins.png')