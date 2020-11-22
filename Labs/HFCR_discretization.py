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
print('-  HFCR Discretization  -')
print('-                       -')
print('-------------------------')

data = pd.read_csv('../Dataset/heart_failure_clinical_records_dataset.csv')

print(data)
#print(data.describe(include='all'))
data.describe().to_csv(graphsDir + 'HFCR Discretization - Before.csv')
target_collumn = data.pop("DEATH_EVENT")

boolean_attributes = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]
for attribute in boolean_attributes:
    data[attribute] = data[attribute].astype('bool');

for strategy in ['uniform', 'quantile']:
    print("------------------------------ DIVIDING IN BINS ------------------------------")
    bins_edges_text = ""
    numeric_columns = data.select_dtypes(include='number').columns
    new_data = pd.DataFrame()
    for column in data.columns:
        if column not in numeric_columns: new_data = pd.concat((new_data, pd.DataFrame(data[column], columns=[column])), 1)
        else :
            est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy=strategy)
            transform_data = est.fit_transform(data[column].values.reshape(-1, 1))
            new_data = pd.concat((new_data, pd.DataFrame(transform_data, columns=[column])), 1)
            bins_edges_text += "Collumn " + column + ": " + str(est.bin_edges_[0]) + "\n"
    file = open(graphsDir + 'HFCR Discretization - After with ' + strategy + '.txt', 'w')
    file.write(bins_edges_text)
    file.close()

    #print(new_data)
    #print(new_data.describe(include='all'))
    print("------------------------------ ONE HOT ENCODER ------------------------------")

    temp_data = new_data
    new_data = pd.DataFrame()
    one_hot_encoder = OneHotEncoder(sparse=False, drop='if_binary')
    for column in temp_data.columns:
        one_hot_encoder.fit(temp_data[column].values.reshape(-1, 1))
        feature_names = one_hot_encoder.get_feature_names([column])
        transformed_data = one_hot_encoder.transform(temp_data[column].values.reshape(-1, 1))
        new_data = pd.concat((new_data, pd.DataFrame(transformed_data, columns=feature_names)), 1)

    new_data = new_data.join(target_collumn, how='right')
    print(new_data)
    #print(new_data.describe(include='all'))
    new_data.describe().to_csv(graphsDir + 'HFCR Discretization - After with ' + strategy + '.csv')
    print("------------------------------ ------------------------------")