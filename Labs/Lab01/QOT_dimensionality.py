import pandas as pd
import matplotlib.pyplot as plt
import ds_functions as ds

from pandas.plotting import register_matplotlib_converters

graphsDir = './Results/'

register_matplotlib_converters()
data = pd.read_csv('../../Dataset/qsar_oral_toxicity.csv', sep=';', header=None)
print(data.shape)

plt.figure(figsize=(4,2))
values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
ds.bar_chart(values.keys(), values.values(), edgecolor="darkgreen", color="green")
plt.title('Nr of records vs nr of variables', color="green")
plt.savefig(graphsDir + 'QOT Dimensionality - NrRecords.png')

print(data.dtypes)
data.dtypes.to_csv(graphsDir + 'QOT Dimensionality - Types of variables.csv')

cat_vars = data.select_dtypes(include='object')
data[cat_vars.columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
print(data.dtypes)
data.dtypes.to_csv(graphsDir + 'QOT Dimensionality - Types of variables - Improvement.csv')

plt.figure()
mv = {}
for var in data:
    mv[var] = data[var].isna().sum()
ds.bar_chart(mv.keys(), mv.values(), title='Nr of missing values per variable',
               xlabel='variables',
               ylabel='nr missing values')
print(mv.values())
plt.savefig(graphsDir + 'QOT Dimensionality - NrMissingValues.png')