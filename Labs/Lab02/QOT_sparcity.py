import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import config as cfg
import os

register_matplotlib_converters()
graphsDir = './Results/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)
print('-------------------------')
print('-                       -')
print('-      QOT Sparcity     -')
print('-                       -')
print('-------------------------')

data = pd.read_csv('../../Dataset/qsar_oral_toxicity.csv', sep=';', header=None)
sample = data.sample(1)


print('QOT Sparcity')
columns = sample.select_dtypes(include='number').columns
rows, cols = len(columns)-1, len(columns)-1
plt.figure()
fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4), squeeze=False)
for i in range(len(columns)):
	var1 = columns[i]
	for j in range(i+1, len(columns)):
		var2 = columns[j]
		axs[i, j-1].set_title("%s x %s"%(var1,var2))
		axs[i, j-1].set_xlabel(var1)
		axs[i, j-1].set_ylabel(var2)
		axs[i, j-1].scatter(sample[var1], sample[var2])
plt.title('QOT Sparcity')
plt.savefig(graphsDir + 'QOT Sparcity')
