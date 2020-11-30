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

threshold_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.24, 0.249]

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

labels = []
values = []
for t in threshold_list:
    subDir = graphsDir + 'Threshold = ' + str(t) + '/'
    if not os.path.exists(subDir):
        os.makedirs(subDir)
    labels.append(t)
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

    data = original_data.copy()[new_features]

    v1 = 0
    v2 = 4

    N_CLUSTERS = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
    rows, cols = ds.choose_grid(len(N_CLUSTERS))

    print('QOT Clustering - K-Means')
    mse: list = []
    mae: list = []
    sc: list = []
    db: list = []
    fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
    i, j = 0, 0
    for n in range(len(N_CLUSTERS)):
        k = N_CLUSTERS[n]
        estimator = KMeans(n_clusters=k)
        estimator.fit(data)
        mse.append(estimator.inertia_)
        mae.append(ds.compute_mae(data.values, estimator.labels_, estimator.cluster_centers_))
        sc.append(silhouette_score(data, estimator.labels_))
        db.append(davies_bouldin_score(data, estimator.labels_))
        ds.plot_clusters(data, v2, v1, estimator.labels_.astype(float), estimator.cluster_centers_, k,
                         f'KMeans k={k}', ax=axs[i,j])
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    plt.suptitle('QOT Clustering - K-Means')
    plt.savefig(subDir + 'QOT Clustering - K-Means')


    print('QOT Clustering - K-Means MSE vs MAE vs SC vs DB')
    fig, ax = plt.subplots(1, 4, figsize=(10, 3), squeeze=False)
    ds.plot_line(N_CLUSTERS, mse, title='KMeans MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
    ds.plot_line(N_CLUSTERS, mae, title='KMeans MAE', xlabel='k', ylabel='MAE', ax=ax[0, 1])
    ds.plot_line(N_CLUSTERS, sc, title='KMeans SC', xlabel='k', ylabel='SC', ax=ax[0, 2], percentage=True)
    ds.plot_line(N_CLUSTERS, db, title='KMeans DB', xlabel='k', ylabel='DB', ax=ax[0, 3])
    plt.suptitle('QOT Clustering - K-Means MSE vs MAE vs SC vs DB')
    plt.savefig(subDir + 'QOT Clustering - K-Means MSE vs MAE vs SC vs DB')

    """
    print('QOT Clustering - Expectation-Maximization')
    mse: list = []
    mae: list = []
    sc: list = []
    db: list = []
    _, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
    i, j = 0, 0
    for n in range(len(N_CLUSTERS)):
        k = N_CLUSTERS[n]
        estimator = GaussianMixture(n_components=k)
        estimator.fit(data)
        labels = estimator.predict(data)
        mse.append(ds.compute_mse(data.values, labels, estimator.means_))
        mae.append(ds.compute_mae(data.values, labels, estimator.means_))
        sc.append(silhouette_score(data, labels))
        db.append(davies_bouldin_score(data, labels))
        ds.plot_clusters(data, v2, v1, labels.astype(float), estimator.means_, k,
                         f'EM k={k}', ax=axs[i,j])
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    plt.suptitle('QOT Clustering - Expectation-Maximization')
    plt.savefig(subDir + 'QOT Clustering - Expectation-Maximization')


    print('QOT Clustering - Expectation-Maximization MSE vs MAE vs SC vs DB')
    fig, ax = plt.subplots(1, 4, figsize=(10, 3), squeeze=False)
    ds.plot_line(N_CLUSTERS, mse, title='EM MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
    ds.plot_line(N_CLUSTERS, mae, title='EM MAE', xlabel='k', ylabel='MAE', ax=ax[0, 1])
    ds.plot_line(N_CLUSTERS, sc, title='EM SC', xlabel='k', ylabel='SC', ax=ax[0, 2], percentage=True)
    ds.plot_line(N_CLUSTERS, db, title='EM DB', xlabel='k', ylabel='DB', ax=ax[0, 3])
    plt.suptitle('QOT Clustering - Expectation-Maximization MSE vs MAE vs SC vs DB')
    plt.savefig(subDir + 'QOT Clustering - Expectation-Maximization MSE vs MAE vs SC vs DB')


    print('QOT Clustering - EPS (Density-based)')
    EPS = [2.5, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    mse: list = []
    mae: list = []
    sc: list = []
    db: list = []
    rows, cols = ds.choose_grid(len(EPS))
    _, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
    i, j = 0, 0
    for n in range(len(EPS)):
        estimator = DBSCAN(eps=EPS[n], min_samples=2)
        estimator.fit(data)
        labels = estimator.labels_
        k = len(set(labels)) - (1 if -1 in labels else 0)
        if k > 1:
            centers = ds.compute_centroids(data, labels)
            mse.append(ds.compute_mse(data.values, labels, centers))
            mae.append(ds.compute_mae(data.values, labels, centers))
            sc.append(silhouette_score(data, labels))
            db.append(davies_bouldin_score(data, labels))
            ds.plot_clusters(data, v2, v1, labels.astype(float), estimator.components_, k,
                             f'DBSCAN eps={EPS[n]} k={k}', ax=axs[i,j])
            i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
        else:
            mse.append(0)
            mae.append(0)
            sc.append(0)
            db.append(0)
    plt.suptitle('QOT Clustering - EPS (Density-based)')
    plt.savefig(subDir + 'QOT Clustering - EPS (Density-based)')


    print('QOT Clustering - EPS (Density-based) MSE vs MAE vs SC vs DB')
    fig, ax = plt.subplots(1, 4, figsize=(10, 3), squeeze=False)
    ds.plot_line(EPS, mse, title='DBSCAN MSE', xlabel='eps', ylabel='MSE', ax=ax[0, 0])
    ds.plot_line(EPS, mae, title='DBSCAN MAE', xlabel='eps', ylabel='MAE', ax=ax[0, 1])
    ds.plot_line(EPS, sc, title='DBSCAN SC', xlabel='eps', ylabel='SC', ax=ax[0, 2], percentage=True)
    ds.plot_line(EPS, db, title='DBSCAN DB', xlabel='eps', ylabel='DB', ax=ax[0, 3])
    plt.suptitle('QOT Clustering - EPS (Density-based) MSE vs MAE vs SC vs DB')
    plt.savefig(subDir + 'QOT Clustering - EPS (Density-based) MSE vs MAE vs SC vs DB')



    print('QOT Clustering - Metric (Density-based)')
    METRICS = ['euclidean', 'cityblock', 'chebyshev', 'cosine', 'jaccard']
    distances = []
    for m in METRICS:
        dist = np.mean(np.mean(squareform(pdist(data.values, metric=m))))
        distances.append(dist)

    print('AVG distances among records', distances)
    distances[0] *= 0.6
    distances[1] = 80
    distances[2] *= 0.6
    distances[3] *= 0.1
    distances[4] *= 0.15
    print('CHOSEN EPS', distances)

    mse: list = []
    mae: list = []
    sc: list = []
    db: list = []
    rows, cols = ds.choose_grid(len(METRICS))
    _, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
    i, j = 0, 0
    for n in range(len(METRICS)):
        estimator = DBSCAN(eps=distances[n], min_samples=2, metric=METRICS[n])
        estimator.fit(data)
        labels = estimator.labels_
        k = len(set(labels)) - (1 if -1 in labels else 0)
        if k > 1:
            centers = ds.compute_centroids(data, labels)
            mse.append(ds.compute_mse(data.values, labels, centers))
            mae.append(ds.compute_mae(data.values, labels, centers))
            sc.append(silhouette_score(data, labels))
            db.append(davies_bouldin_score(data, labels))
            ds.plot_clusters(data, v2, v1, labels.astype(float), estimator.components_, k,
                             f'DBSCAN metric={METRICS[n]} eps={distances[n]:.2f} k={k}', ax=axs[i,j])
        else:
            mse.append(0)
            mae.append(0)
            sc.append(0)
            db.append(0)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    plt.suptitle('QOT Clustering - Metric (Density-based)')
    plt.savefig(subDir + 'QOT Clustering - Metric (Density-based)')


    print('QOT Clustering - Metric (Density-based) MSE vs MAE vs SC vs DB')
    fig, ax = plt.subplots(1, 4, figsize=(10, 3), squeeze=False)
    ds.bar_chart(METRICS, mse, title='DBSCAN MSE', xlabel='metric', ylabel='MSE', ax=ax[0, 0])
    ds.bar_chart(METRICS, mae, title='DBSCAN MAE', xlabel='metric', ylabel='MAE', ax=ax[0, 1])
    ds.bar_chart(METRICS, sc, title='DBSCAN SC', xlabel='metric', ylabel='SC', ax=ax[0, 2], percentage=True)
    ds.bar_chart(METRICS, db, title='DBSCAN DB', xlabel='metric', ylabel='DB', ax=ax[0, 3])
    plt.suptitle('QOT Clustering - Metric (Density-based) MSE vs MAE vs SC vs DB')
    plt.savefig(subDir + 'QOT Clustering - Metric (Density-based) MSE vs MAE vs SC vs DB')



    print('QOT Clustering - Hierarchical')
    mse: list = []
    mae: list = []
    sc: list = []
    db: list = []
    rows, cols = ds.choose_grid(len(N_CLUSTERS))
    _, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
    i, j = 0, 0
    for n in range(len(N_CLUSTERS)):
        k = N_CLUSTERS[n]
        estimator = AgglomerativeClustering(n_clusters=k)
        estimator.fit(data)
        labels = estimator.labels_
        centers = ds.compute_centroids(data, labels)
        mse.append(ds.compute_mse(data.values, labels, centers))
        mae.append(ds.compute_mae(data.values, labels, centers))
        sc.append(silhouette_score(data, labels))
        db.append(davies_bouldin_score(data, labels))
        ds.plot_clusters(data, v2, v1, labels, centers, k,
                         f'Hierarchical k={k}', ax=axs[i,j])
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    plt.suptitle('QOT Clustering - Hierarchical')
    plt.savefig(subDir + 'QOT Clustering - Hierarchical')



    print('QOT Clustering - Hierarchical MSE vs MAE vs SC vs DB')
    fig, ax = plt.subplots(1, 4, figsize=(10, 3), squeeze=False)
    ds.plot_line(N_CLUSTERS, mse, title='Hierarchical MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
    ds.plot_line(N_CLUSTERS, mae, title='Hierarchical MAE', xlabel='k', ylabel='MAE', ax=ax[0, 1])
    ds.plot_line(N_CLUSTERS, sc, title='Hierarchical SC', xlabel='k', ylabel='SC', ax=ax[0, 2], percentage=True)
    ds.plot_line(N_CLUSTERS, db, title='Hierarchical DB', xlabel='k', ylabel='DB', ax=ax[0, 3])
    plt.suptitle('QOT Clustering - Hierarchical MSE vs MAE vs SC vs DB')
    plt.savefig(subDir + 'QOT Clustering - Hierarchical MSE vs MAE vs SC vs DB')


    print('QOT Clustering - Metric (Hierarchical)')
    METRICS = ['euclidean', 'cityblock', 'chebyshev', 'cosine', 'jaccard']
    LINKS = ['complete', 'average']
    k = 3
    values_mse = {}
    values_mae = {}
    values_sc = {}
    values_db = {}
    rows = len(METRICS)
    cols = len(LINKS)
    _, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
    for i in range(len(METRICS)):
        mse: list = []
        mae: list = []
        sc: list = []
        db: list = []
        m = METRICS[i]
        for j in range(len(LINKS)):
            link = LINKS[j]
            estimator = AgglomerativeClustering(n_clusters=k, linkage=link, affinity=m )
            estimator.fit(data)
            labels = estimator.labels_
            centers = ds.compute_centroids(data, labels)
            mse.append(ds.compute_mse(data.values, labels, centers))
            mae.append(ds.compute_mae(data.values, labels, centers))
            sc.append(silhouette_score(data, labels))
            db.append(davies_bouldin_score(data, labels))
            ds.plot_clusters(data, v2, v1, labels, centers, k,
                             f'Hierarchical k={k} metric={m} link={link}', ax=axs[i,j])
        values_mse[m] = mse
        values_mae[m] = mae
        values_sc[m] = sc
        values_db[m] = db
    plt.suptitle('QOT Clustering - Metric (Hierarchical)')
    plt.savefig(subDir + 'QOT Clustering - Metric (Hierarchical)')



    print('QOT Clustering - Metric (Hierarchical) MSE vs MAE vs SC vs DB')
    _, ax = plt.subplots(1, 4, figsize=(10, 3), squeeze=False)
    ds.multiple_bar_chart(LINKS, values_mse, title=f'Hierarchical MSE', xlabel='metric', ylabel='MSE', ax=ax[0, 0])
    ds.multiple_bar_chart(LINKS, values_mae, title=f'Hierarchical MAE', xlabel='metric', ylabel='MAE', ax=ax[0, 1])
    ds.multiple_bar_chart(LINKS, values_sc, title=f'Hierarchical SC', xlabel='metric', ylabel='SC', ax=ax[0, 2], percentage=True)
    ds.multiple_bar_chart(LINKS, values_db, title=f'Hierarchical DB', xlabel='metric', ylabel='DB', ax=ax[0, 3])
    plt.suptitle('QOT Clustering - Metric (Hierarchical) MSE vs MAE vs SC vs DB')
    plt.savefig(subDir + 'QOT Clustering - Metric (Hierarchical) MSE vs MAE vs SC vs DB')
    """

features_file.close()