import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ds_functions as ds
import os

graphsDir = './Results/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)


print('-----------------------------------')
print('-                                 -')
print('-     HFCR Outliers Imputation     -')
print('-                                 -')
print('-----------------------------------')



data: pd.DataFrame = pd.read_csv('../../Dataset/heart_failure_clinical_records_dataset.csv')

print('HFCR Outliers Imputation - Description')
print()
print('Initial description table')
print(data.describe(include='all'))
print()
print('Outilers Imputation...')
print()
print('New description table')

# Mean

# Winsorization
for var in data:
	if var == 'DEATH_EVENT':
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

print(data.describe(include='all'))
data.describe().to_csv(graphsDir + 'HFCR Outliers Imputation - Description.csv')
print()

print('HFCR Outliers Imputation - Boxplot')
data.boxplot(rot=45, figsize=(9,12), whis=1.5)
plt.suptitle('HFCR Outliers Imputation - Boxplot')
plt.savefig(graphsDir + 'HFCR Outliers Imputation - Boxplot')
print()

print('HFCR Outliers Imputation - Boxplot for each variable')
numeric_vars = data.select_dtypes(include='number').columns
rows, cols = ds.choose_grid(len(numeric_vars))
fig, axs = plt.subplots(rows, cols, figsize=(cols*ds.HEIGHT, rows*ds.HEIGHT))
i, j = 0, 0
for n in range(len(numeric_vars)):
    axs[i, j].set_title('Boxplot for %s'%numeric_vars[n])
    axs[i, j].boxplot(data[numeric_vars[n]].dropna().values, whis=1.5)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
plt.suptitle('HFCR Outliers Imputation - Boxplot for each variable')
plt.savefig(graphsDir + 'HFCR Outliers Imputation - Boxplot for each variable')
print()