import matplotlib.pyplot as plt
import ds_functions as ds
import pandas as pd
import os

from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
graphsDir = './Results/Discretization/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('-------------------------')
print('-                       -')
print('-   QOT Discretization  -')
print('-                       -')
print('-------------------------')

data = pd.read_csv('../Dataset/qsar_oral_toxicity.csv', sep=';', header=None)

print(data)
#print(data.describe(include='all'))
data.describe().to_csv(graphsDir + 'QOT Discretization - Before.csv')
target_collumn = data.pop(1024)

for attribute in data.columns:
    data[attribute] = data[attribute].astype('bool');
print("------------------------------ ONE HOT ENCODER ------------------------------")

new_data = pd.DataFrame()
one_hot_encoder = OneHotEncoder(sparse=False, drop='if_binary')
for column in data.columns:
    one_hot_encoder.fit(data[column].values.reshape(-1, 1))
    feature_names = one_hot_encoder.get_feature_names([str(column)])
    transformed_data = one_hot_encoder.transform(data[column].values.reshape(-1, 1))
    new_data = pd.concat((new_data, pd.DataFrame(transformed_data, columns=feature_names)), 1)

new_data = new_data.join(target_collumn, how='right')
print(new_data)
#print(new_data.describe(include='all'))
new_data.describe().to_csv(graphsDir + 'QOT Discretization - After.csv')
print("------------------------------ ------------------------------")