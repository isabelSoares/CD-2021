import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ds_functions as ds
import os

graphsDir = './Results/Outliers/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)


print('-----------------------------------')
print('-                                 -')
print('-     QOT Outliers Imputation     -')
print('-                                 -')
print('-----------------------------------')



data: pd.DataFrame = pd.read_csv('../Dataset/qsar_oral_toxicity.csv', sep=';', header=None)


print('QOT Outliers Imputation - Description')
print()
print('Initial description table')
print(data.describe())
print()
print('Outilers Imputation...')
print()
print('New description table')

for var in data:
	if var == 1024:
		break

	q1 = data[var].quantile(0.25)
	q3 = data[var].quantile(0.75)
	iqr = q3 - q1
	lower_limit = q1 -  1.5*iqr
	higher_limit = q3 + 1.5*iqr

	acceptable_values = data.loc[(data[var] >= lower_limit) & (data[var] <= higher_limit)]

	var_mean = acceptable_values[var].mean()
	max_value = acceptable_values[var].max()
	min_value = acceptable_values[var].min()


	data.loc[(data[var] < min_value), var] = min_value
	data.loc[(data[var] > max_value), var] = max_value

print(data.describe())
data.describe().to_csv(graphsDir + 'QOT Outliers Imputation - Description.csv')
print()

print('QOT Outliers Imputation - Boxplot')
data.boxplot(rot=45, figsize=(150,3), whis=1.5)
plt.suptitle('QOT Outliers Imputation - Boxplot')
plt.savefig(graphsDir + 'QOT Outliers Imputation - Boxplot')
print()

print('QOT Outliers Imputation - Boxplot for each variable')
numeric_vars = data.select_dtypes(include='number').columns
rows, cols = ds.choose_grid(len(numeric_vars), 18)
height_fix = ds.HEIGHT/1.7
fig, axs = plt.subplots(rows, cols, figsize=(cols*height_fix, rows*height_fix))
i, j = 0, 0
for n in range(len(numeric_vars)):
    axs[i, j].set_title('Boxplot for %s'%numeric_vars[n])
    axs[i, j].boxplot(data[numeric_vars[n]].dropna().values, whis=1.5)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
plt.suptitle('QOT Outliers Imputation - Boxplot for each variable')
plt.savefig(graphsDir + 'QOT Outliers Imputation - Boxplot for each variable')
print()