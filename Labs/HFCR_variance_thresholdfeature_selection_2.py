import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ds_functions as ds
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
import data_preparation_functions as prepfunctions
from sklearn.feature_selection import VarianceThreshold

graphsDir = './Results/FeatureSelection/VarianceThreshold/QOT_Clustering/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

features_file = open(graphsDir + 'HFCR Feature Selection - Features', 'w')

data: pd.DataFrame = pd.read_csv('../Dataset/qsar_oral_toxicity.csv', sep=';', header=None)

data.pop(1024)

threshold_list = [0.02, 0.04, 0.07, 0.1, 0.15, 0.2]

#plt.figure()
#plt.title('Nr of Features')
#bars = plt.bar(labels, values)

#i = 0
#for rect in bars:
#    height = rect.get_height()
#    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % values[i], ha='center', va='bottom', fontsize=7)
#    i += 1
#plt.savefig(graphsDir + 'HFCR Feature Selection - VarianceThreshold')


original_data = data.copy()

labels_t = []
values = []
fig, ax = plt.subplots(1, 3, figsize=(3*3, 4), squeeze=False)
count = 0
fig_values_1 = {}
fig_values_2 = {}
#fig_values_3 = {}
fig_values_4 = {}
N_CLUSTERS = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
EPS = [2.5, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for t in threshold_list:
    subDir = graphsDir + 'Threshold = ' + str(t) + '/'
    if not os.path.exists(subDir):
        os.makedirs(subDir)
    labels_t.append(t)
    sel = VarianceThreshold(threshold=t)
    sel.fit_transform(original_data.values)
    f_to_accept = sel.get_support()
    new_features = []
    for i in range(len(f_to_accept)):
        if f_to_accept[i]:
            new_features.append(i)
    print('t = ' + str(t) + ' / n_features = ' + str(len(new_features)))
    features_file.write('t = ' + str(t) + ": " + str(new_features) + "\n")
    values.append(len(new_features))

    num_features = len(new_features)

    data = original_data.copy()[new_features]

    v1 = 0
    v2 = 4

    rows, cols = ds.choose_grid(len(N_CLUSTERS))

    print('QOT Clustering - K-Means')
    sc: list = []
    for n in range(len(N_CLUSTERS)):
        k = N_CLUSTERS[n]
        estimator = KMeans(n_clusters=k)
        estimator.fit(data)
        sc.append(silhouette_score(data, estimator.labels_))

    fig_values_1[num_features] = sc
    #title = 'SC (t=' + str(t) + ')'
    #ds.plot_line(N_CLUSTERS, sc, title=title, xlabel='k', ylabel='SC', ax=ax[0, count], percentage=True)

    print('QOT Clustering - Expectation-Maximization')
    #mse: list = []
    sc: list = []
    for n in range(len(N_CLUSTERS)):
        k = N_CLUSTERS[n]
        estimator = GaussianMixture(n_components=k)
        estimator.fit(data)
        labels = estimator.predict(data)
        #mse.append(ds.compute_mse(data.values, labels, estimator.means_))
        sc.append(silhouette_score(data, labels))

    fig_values_2[num_features] = sc
    #ds.multiple_line_chart(N_CLUSTERS, mse, ax=ax[0, 0], title='KMeans MSE', xlabel='k', ylabel='MSE')
    #ds.multiple_line_chart(N_CLUSTERS, mae, ax=ax[0, 1], title='KMeans MAE', xlabel='k', ylabel='MAE')
    #ds.multiple_line_chart(N_CLUSTERS, db, ax=ax[0, 3], title='KMeans DB', xlabel='k', ylabel='DB')
    #ds.plot_line(N_CLUSTERS, sc, title=title, xlabel='k', ylabel='SC', ax=ax[0, count], percentage=True)
    
    """
    print('QOT Clustering - EPS (Density-based)')
    sc: list = []
    i, j = 0, 0
    for n in range(len(EPS)):
        estimator = DBSCAN(eps=EPS[n], min_samples=2)
        estimator.fit(data)
        labels = estimator.labels_
        k = len(set(labels)) - (1 if -1 in labels else 0)
        if k > 1:
            centers = ds.compute_centroids(data, labels)
            sc.append(silhouette_score(data, labels))
            i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
        else:
            sc.append(0)
    
    fig_values_3[num_features] = sc
    """

    print('QOT Clustering - Hierarchical')
    sc: list = []
    i, j = 0, 0
    for n in range(len(N_CLUSTERS)):
        k = N_CLUSTERS[n]
        estimator = AgglomerativeClustering(n_clusters=k)
        estimator.fit(data)
        labels = estimator.labels_
        centers = ds.compute_centroids(data, labels)
        sc.append(silhouette_score(data, labels))
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)

    fig_values_4[num_features] = sc

    count += 1

#title = 'EM SC (t=' + str(t) + ')'
ds.multiple_line_chart(N_CLUSTERS, fig_values_1, ax=ax[0, 0], title='K-Means', xlabel='k', ylabel='SC', percentage=True)
#title = 'EM MSE (t=' + str(t) + ')'
ds.multiple_line_chart(N_CLUSTERS, fig_values_2, ax=ax[0, 1], title='EM', xlabel='k', ylabel='SC', percentage=True)

#ds.multiple_line_chart(EPS, fig_values_3, ax=ax[0, 2], title='EPS', xlabel='k', ylabel='SC', percentage=True)

ds.multiple_line_chart(N_CLUSTERS, fig_values_4, ax=ax[0, 2], title='Hierarchical', xlabel='k', ylabel='SC', percentage=True)


#plt.suptitle('QOT Clustering - K-Means SC')
#plt.savefig(graphsDir + 'QOT Clustering - K-Means SC')

plt.suptitle('QOT Clustering with Feature Selection - SC Comparison')
plt.savefig(graphsDir + 'QOT Clustering with Feature Selection - SC Comparison')

features_file.close()