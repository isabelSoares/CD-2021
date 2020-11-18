import data_preparation_functions as prepfunctions
import matplotlib.pyplot as plt
import ds_functions as ds
import pandas as pd
import os

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

graphsDir = './Results/FeatureSelection/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

features_file = open(graphsDir + 'HFCR Feature Selection - Features', 'w')

data: pd.DataFrame = pd.read_csv('../Dataset/heart_failure_clinical_records_dataset.csv')
datas = prepfunctions.prepare_dataset(data, 'DEATH_EVENT', False, False)
for key, value in datas.items():
    print("Key: ", key)
    dataframe_rec = value.copy()
    subDir = graphsDir + key + '/'
    if not os.path.exists(subDir):
        os.makedirs(subDir)

    data = dataframe_rec.copy()
    y = data.pop('DEATH_EVENT')
    print('Original')
    labels = ['Original']
    values = [data.shape[1]]

    print('VarianceThreshold')
    data = dataframe_rec.copy()
    y = data.pop('DEATH_EVENT')
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    sel.fit_transform(data)
    labels += ['VarianceThreshold']
    values += [data.shape[1]]

    print('Chi Percentile')
    percentile = 80
    data = dataframe_rec.copy()
    y = data.pop('DEATH_EVENT')
    data_new = SelectPercentile(chi2, percentile).fit_transform(data, y)
    labels += ['%d%% Chi2'%percentile]
    values += [data_new.shape[1]]

    # Create the RFE object and compute a cross-validated score.
    print('RFECV')
    data = dataframe_rec.copy()
    y = data.pop('DEATH_EVENT')
    print(list(data.columns))
    print(data.describe())
    print(data.shape)
    svc = SVC(kernel="linear")
    rfecv = RFECV(estimator=svc, step=1, n_jobs=-1, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='accuracy')
    print('Fitting RFECV')
    rfecv.fit(data, y)
    print('Ended Fitting RFECV')
    plt.figure()
    plt.text(0.5, 0.03, 'Optimal number of features : %d' % rfecv.n_features_, fontsize=7, ha='center', va='center')
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.savefig(subDir + 'HFCR Feature Selection - Stratified KFold - ' + key)
    labels += ['RFECV StrKFold']
    values += [rfecv.n_features_]

    new_features = []
    for bool, feature in zip(rfecv.support_, list(data.columns)):
        if bool: new_features.append(feature)
    print(new_features)
    features_file.write(key + ": " + str(new_features) + "\n")

    # Bar Graphs General
    plt.figure()
    plt.title('Nr of Features')
    bars = plt.bar(labels, values)

    i = 0
    for rect in bars:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % values[i], ha='center', va='bottom', fontsize=7)
        i += 1

    plt.savefig(subDir + 'HFCR Feature Selection - ' + key)
    plt.close("all")

features_file.close()