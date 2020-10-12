import pandas as pd
import matplotlib
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
sampleSize = 1000;
sample = data.sample(sampleSize)


print('QOT Sparcity')
columns = sample.select_dtypes(include='number').columns
rows, cols = 35, 30
for i in range(len(columns)):
	var1 = columns[i]
	print("Dealing now the var: %s"%(var1))
	plt.figure()
	fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4), squeeze=False)

	print("Setting plt attributes")
	for j in range(i+1, len(columns)):
		var2 = columns[j]
		axs[j // 30, j % 30].set_title("%s x %s"%(var1,var2))
		axs[j // 30, j % 30].set_xlabel(var1)
		axs[j // 30, j % 30].set_ylabel(var2)
		axs[j // 30, j % 30].scatter(sample[var1], sample[var2], c='darkblue')

	plt.suptitle('QOT Sparcity %s\n%d samples'%(var1, sampleSize))
	print("Saving Figure")
	plt.savefig(graphsDir + 'QOT Sparcity/Var' + "%s"%(var1))
	print("Closing figure")
	plt.close(fig)
