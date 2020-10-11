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
sampleSize = 10;
sample = data.sample(10)


print('QOT Sparcity')
columns = sample.select_dtypes(include='number').columns
rows, cols = 32, 32
for i in range(len(columns)):
	plt.figure()
	fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4), squeeze=False)

	var1 = columns[i]
	print("Dealing now the var: %s"%(var1))
	for j in range(i+1, len(columns)):
		var2 = columns[j]
		axs[j // 32, j % 32].set_title("%s x %s"%(var1,var2))
		axs[j // 32, j % 32].set_xlabel(var1)
		axs[j // 32, j % 32].set_ylabel(var2)
		axs[j // 32, j % 32].scatter(sample[var1], sample[var2], c='darkblue')

	plt.suptitle('QOT Sparcity %s\n%d samples'%(var1, sampleSize))
	plt.savefig(graphsDir + 'QOT Sparcity/Var' + "%s"%(var1))
	plt.close()
