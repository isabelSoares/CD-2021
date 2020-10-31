import matplotlib.pyplot as plt
import ds_functions as ds
import pandas as pd
import os

from imblearn.over_sampling import SMOTE

graphsDir = './Results/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

unbal: pd.DataFrame = pd.read_csv('../../Dataset/qsar_oral_toxicity.csv', sep=';', header=None)
target_count = unbal[1024].value_counts()
plt.figure()
plt.title('Class balance')
bars = plt.bar(target_count.index, target_count.values)

i = 0
for rect in bars:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % target_count.values[i], ha='center', va='bottom', fontsize=7)
    i += 1

plt.savefig(graphsDir + 'QOT Balancing - Class Balance Original')

min_class = target_count.idxmin()
ind_min_class = target_count.index.get_loc(min_class)

print('Minority class:', target_count[ind_min_class])
print('Majority class:', target_count[1-ind_min_class])
print('Proportion:', round(target_count[ind_min_class] / target_count[1-ind_min_class], 2), ': 1')

RANDOM_STATE = 42
values = {'Original': [target_count.values[ind_min_class], target_count.values[1-ind_min_class]]}

df_class_min = unbal[unbal[1024] == min_class]
df_class_max = unbal[unbal[1024] != min_class]

df_under = df_class_max.sample(len(df_class_min))
values['UnderSample'] = [target_count.values[ind_min_class], len(df_under)]

df_over = df_class_min.sample(len(df_class_max), replace=True)
values['OverSample'] = [len(df_over), target_count.values[1-ind_min_class]]

smote = SMOTE(sampling_strategy='minority', random_state=RANDOM_STATE)
y = unbal.pop(1024).values
X = unbal.values
smote_X, smote_y = smote.fit_sample(X, y)
smote_target_count = pd.Series(smote_y).value_counts()
values['SMOTE'] = [smote_target_count.values[ind_min_class], smote_target_count.values[1-ind_min_class]]

fig = plt.figure()
ds.multiple_bar_chart([target_count.index[ind_min_class], target_count.index[1-ind_min_class]], values,
                      title='Target', xlabel='frequency', ylabel='Class balance')
plt.savefig(graphsDir + 'QOT Balancing - Class Balanced')