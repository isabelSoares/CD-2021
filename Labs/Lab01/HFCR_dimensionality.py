import pandas as pd
import matplotlib.pyplot as plt
import ds_functions as ds

from pandas.plotting import register_matplotlib_converters

graphsDir = './Results/'

register_matplotlib_converters()
data = pd.read_csv('../../Dataset/heart_failure_clinical_records_dataset.csv')
print(data.shape)

plt.figure(figsize=(4,2))
values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
ds.bar_chart(values.keys(), values.values(), edgecolor="teal", color="turquoise") 
plt.title('Nr of records vs nr of variables', color="teal")
plt.savefig(graphsDir + 'HFCR Dimensionality - NrRecords.png')

print(data.dtypes)
data.dtypes.to_csv(graphsDir + 'HFCR Dimensionality - Types of variables.csv')

cat_vars = data.select_dtypes(include='object')
data[cat_vars.columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
print(data.dtypes)
data.dtypes.to_csv(graphsDir + 'HFCR Dimensionality - Types of variables - Improvement.csv')

plt.figure(figsize=(20,3))
mv = {}
for var in data:
    mv[var] = data[var].isna().sum()
ds.bar_chart(mv.keys(), mv.values(),
               xlabel='variables',
               ylabel='nr missing values',
               color="teal")
print(mv.values())
plt.title('Nr of missing values per variable', color="teal")
plt.savefig(graphsDir + 'HFCR Dimensionality - NrMissingValues.png')