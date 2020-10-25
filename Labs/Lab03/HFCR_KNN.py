import numpy as np
import pandas as pd
import ds_functions as ds
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

graphsDir = './Results/'

data: pd.DataFrame = pd.read_csv('../../Dataset/heart_failure_clinical_records_dataset.csv')

data['DEATH_EVENT'] = data['DEATH_EVENT'].astype('category')
sb_vars = data.select_dtypes(include='object')
data[sb_vars.columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))

cols_nr = data.select_dtypes(include='number')
# print(cols_nr)
cols_sb = data.select_dtypes(include='category')
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

print(norm_data_zscore.dtypes)

y: np.ndarray = norm_data_zscore.pop('DEATH_EVENT').values
X: np.ndarray = norm_data_zscore.values
labels = pd.unique(y)

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
dist = ['manhattan', 'euclidean', 'chebyshev']
values = {}
best = (0, '')
last_best = 0
for d in dist:
    yvalues = []
    for n in nvalues:
        knn = KNeighborsClassifier(n_neighbors=n, metric=d)
        knn.fit(trnX, trnY)
        prdY = knn.predict(tstX)
        yvalues.append(metrics.accuracy_score(tstY, prdY))
        if yvalues[-1] > last_best:
            best = (n, d)
            last_best = yvalues[-1]
    values[d] = yvalues

plt.figure()
ds.multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel='accuracy', percentage=True)
plt.savefig(graphsDir + 'HFCR KNN Variants')
print('Best results with %d neighbors and %s'%(best[0], best[1]))

clf = knn = KNeighborsClassifier(n_neighbors=best[0], metric=best[1])
clf.fit(trnX, trnY)
prd_trn = clf.predict(trnX)
prd_tst = clf.predict(tstX)
ds.plot_evaluation_results(pd.unique(y), trnY, prd_trn, tstY, prd_tst)
plt.suptitle('HFCR KNN - Performance & Confusion matrix - %d neighbors and %s'%(best[0], best[1]))
plt.savefig(graphsDir + 'HFCR KNN - Performance & Confusion matrix')