import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import ds_functions as ds
import os

graphsDir = './Results/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)


print('----------------------------')
print('-                          -')
print('-       HFCR Scaling       -')
print('-                          -')
print('----------------------------')

register_matplotlib_converters()
original = pd.read_csv('../../Dataset/heart_failure_clinical_records_dataset.csv')

original['DEATH_EVENT'] = original['DEATH_EVENT'].astype('category')
sb_vars = original.select_dtypes(include='object')
original[sb_vars.columns] = original.select_dtypes(['object']).apply(lambda x: x.astype('category'))

cols_nr = original.select_dtypes(include='number')
# print(cols_nr)
cols_sb = original.select_dtypes(include='category')
# print(cols_sb)

imp_nr = SimpleImputer(strategy='mean', missing_values=np.nan, copy=True)
imp_sb = SimpleImputer(strategy='most_frequent', missing_values='', copy=True)
df_nr = pd.DataFrame(imp_nr.fit_transform(cols_nr), columns=cols_nr.columns)
# df_sb = pd.DataFrame(imp_sb.fit_transform(cols_sb), columns=cols_sb.columns)

transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_nr)
df_nr = pd.DataFrame(transf.transform(df_nr), columns= df_nr.columns)
norm_data_zscore = df_nr.join(cols_sb, how='right')
#norm_data_zscore = df_nr
norm_data_zscore['DEATH_EVENT'] = norm_data_zscore['DEATH_EVENT'].astype('int64')
norm_data_zscore.describe(include='all')

transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_nr)
df_nr = pd.DataFrame(transf.transform(df_nr), columns= df_nr.columns)
norm_data_minmax = df_nr.join(cols_sb, how='right')
#norm_data_minmax = df_nr
norm_data_minmax['DEATH_EVENT'] = norm_data_minmax['DEATH_EVENT'].astype('int64')
norm_data_minmax.describe(include='all')

original['DEATH_EVENT'] = original['DEATH_EVENT'].astype('int64')

fig, axs = plt.subplots(1, 3, figsize=(60,20),squeeze=False)
axs[0, 0].set_title('Original data')
original.boxplot(ax=axs[0, 0])
axs[0, 1].set_title('Z-score normalization')
norm_data_zscore.boxplot(ax=axs[0, 1])
axs[0, 2].set_title('MinMax normalization')
norm_data_minmax.boxplot(ax=axs[0, 2])
# fig.tight_layout()
plt.savefig(graphsDir + 'HFCR Scaling')