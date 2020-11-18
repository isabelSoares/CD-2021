import os
import pandas as pd
import matplotlib.pyplot as plt
import ds_functions as ds
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
import data_preparation_functions as prepfunctions

graphsDir = './Results/Clustering/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('--------------------------')
print('-                        -')
print('-     QOT Clustering     -')
print('-                        -')
print('--------------------------')

data: pd.DataFrame = pd.read_csv('../Dataset/qsar_oral_toxicity.csv', sep=';', header=None)
datas = prepfunctions.prepare_dataset(data, 1024, False, False)
featured_datas = prepfunctions.mask_feature_selection(datas, 1024, True, './Results/FeatureSelection/QOT Feature Selection - Features')

count = 0
for key in datas:
    if count < 2:
        count += 1
        continue
    for do_feature_eng in [False]:
        if (do_feature_eng):
            data = featured_datas[key]
            subDir = graphsDir + 'FeatureEng/' +  key + '/'
            if not os.path.exists(subDir):
                os.makedirs(subDir)
        else:
            data = datas[key]
            subDir = graphsDir + key + '/'
            if not os.path.exists(subDir):
                os.makedirs(subDir)

        data.pop(1024)

        v1 = 0
        v2 = 4

        N_CLUSTERS = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
        rows, cols = ds.choose_grid(len(N_CLUSTERS))



        print('QOT Clustering - ' + key + ' - K-Means')
        mse: list = []
        sc: list = []
        fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
        i, j = 0, 0
        for n in range(len(N_CLUSTERS)):
            k = N_CLUSTERS[n]
            estimator = KMeans(n_clusters=k)
            estimator.fit(data)
            mse.append(estimator.inertia_)
            sc.append(silhouette_score(data, estimator.labels_))
            ds.plot_clusters(data, v2, v1, estimator.labels_.astype(float), estimator.cluster_centers_, k,
                             f'KMeans k={k}', ax=axs[i,j])
            i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
        plt.suptitle('QOT Clustering - ' + key + ' - K-Means')
        plt.savefig(subDir + 'QOT Clustering - ' + key + ' - K-Means')



        print('QOT Clustering - ' + key + ' - K-Means MSE vs SC')
        fig, ax = plt.subplots(1, 2, figsize=(6, 3), squeeze=False)
        ds.plot_line(N_CLUSTERS, mse, title='KMeans MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
        ds.plot_line(N_CLUSTERS, sc, title='KMeans SC', xlabel='k', ylabel='SC', ax=ax[0, 1], percentage=True)
        plt.suptitle('QOT Clustering - ' + key + ' - K-Means MSE vs SC')
        plt.savefig(subDir + 'QOT Clustering - ' + key + ' - K-Means MSE vs SC')



        print('QOT Clustering - ' + key + ' - Expectation-Maximization')
        mse: list = []
        sc: list = []
        _, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
        i, j = 0, 0
        for n in range(len(N_CLUSTERS)):
            k = N_CLUSTERS[n]
            estimator = GaussianMixture(n_components=k)
            estimator.fit(data)
            labels = estimator.predict(data)
            mse.append(ds.compute_mse(data.values, labels, estimator.means_))
            sc.append(silhouette_score(data, labels))
            ds.plot_clusters(data, v2, v1, labels.astype(float), estimator.means_, k,
                             f'EM k={k}', ax=axs[i,j])
            i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
        plt.suptitle('QOT Clustering - ' + key + ' - Expectation-Maximization')
        plt.savefig(subDir + 'QOT Clustering - ' + key + ' - Expectation-Maximization')



        print('QOT Clustering - ' + key + ' - Expectation-Maximization MSE vs SC')
        fig, ax = plt.subplots(1, 2, figsize=(6, 3), squeeze=False)
        ds.plot_line(N_CLUSTERS, mse, title='EM MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
        ds.plot_line(N_CLUSTERS, sc, title='EM SC', xlabel='k', ylabel='SC', ax=ax[0, 1], percentage=True)
        plt.suptitle('QOT Clustering - ' + key + ' - Expectation-Maximization MSE vs SC')
        plt.savefig(subDir + 'QOT Clustering - ' + key + ' - Expectation-Maximization MSE vs SC')



        print('QOT Clustering - ' + key + ' - EPS (Density-based)')
        EPS = [2.5, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        mse: list = []
        sc: list = []
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
                sc.append(silhouette_score(data, labels))
                ds.plot_clusters(data, v2, v1, labels.astype(float), estimator.components_, k,
                                 f'DBSCAN eps={EPS[n]} k={k}', ax=axs[i,j])
                i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
            else:
                mse.append(0)
                sc.append(0)
        plt.suptitle('QOT Clustering - ' + key + ' - EPS (Density-based)')
        plt.savefig(subDir + 'QOT Clustering - ' + key + ' - EPS (Density-based)')



        print('QOT Clustering - ' + key + ' - EPS (Density-based) MSE vs SC')
        fig, ax = plt.subplots(1, 2, figsize=(6, 3), squeeze=False)
        ds.plot_line(EPS, mse, title='DBSCAN MSE', xlabel='eps', ylabel='MSE', ax=ax[0, 0])
        ds.plot_line(EPS, sc, title='DBSCAN SC', xlabel='eps', ylabel='SC', ax=ax[0, 1], percentage=True)
        plt.suptitle('QOT Clustering - ' + key + ' - EPS (Density-based) MSE vs SC')
        plt.savefig(subDir + 'QOT Clustering - ' + key + ' - EPS (Density-based) MSE vs SC')



        print('QOT Clustering - ' + key + ' - Metric (Density-based)')
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
        sc: list = []
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
                sc.append(silhouette_score(data, labels))
                ds.plot_clusters(data, v2, v1, labels.astype(float), estimator.components_, k,
                                 f'DBSCAN metric={METRICS[n]} eps={distances[n]:.2f} k={k}', ax=axs[i,j])
            else:
                mse.append(0)
                sc.append(0)
            i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
        plt.suptitle('QOT Clustering - ' + key + ' - Metric (Density-based)')
        plt.savefig(subDir + 'QOT Clustering - ' + key + ' - Metric (Density-based)')



        print('QOT Clustering - ' + key + ' - Metric (Density-based) MSE vs SC')
        fig, ax = plt.subplots(1, 2, figsize=(6, 3), squeeze=False)
        ds.bar_chart(METRICS, mse, title='DBSCAN MSE', xlabel='metric', ylabel='MSE', ax=ax[0, 0])
        ds.bar_chart(METRICS, sc, title='DBSCAN SC', xlabel='metric', ylabel='SC', ax=ax[0, 1], percentage=True)
        plt.suptitle('QOT Clustering - ' + key + ' - Metric (Density-based) MSE vs SC')
        plt.savefig(subDir + 'QOT Clustering - ' + key + ' - Metric (Density-based) MSE vs SC')



        print('QOT Clustering - ' + key + ' - Hierarchical')
        mse: list = []
        sc: list = []
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
            sc.append(silhouette_score(data, labels))
            ds.plot_clusters(data, v2, v1, labels, centers, k,
                             f'Hierarchical k={k}', ax=axs[i,j])
            i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
        plt.suptitle('QOT Clustering - ' + key + ' - Hierarchical')
        plt.savefig(subDir + 'QOT Clustering - ' + key + ' - Hierarchical')



        print('QOT Clustering - ' + key + ' - Hierarchical MSE vs SC')
        fig, ax = plt.subplots(1, 2, figsize=(6, 3), squeeze=False)
        ds.plot_line(N_CLUSTERS, mse, title='Hierarchical MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
        ds.plot_line(N_CLUSTERS, sc, title='Hierarchical SC', xlabel='k', ylabel='SC', ax=ax[0, 1], percentage=True)
        plt.suptitle('QOT Clustering - ' + key + ' - Hierarchical MSE vs SC')
        plt.savefig(subDir + 'QOT Clustering - ' + key + ' - Hierarchical MSE vs SC')



        print('QOT Clustering - ' + key + ' - Metric (Hierarchical)')
        METRICS = ['euclidean', 'cityblock', 'chebyshev', 'cosine', 'jaccard']
        LINKS = ['complete', 'average']
        k = 3
        values_mse = {}
        values_sc = {}
        rows = len(METRICS)
        cols = len(LINKS)
        _, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
        for i in range(len(METRICS)):
            mse: list = []
            sc: list = []
            m = METRICS[i]
            for j in range(len(LINKS)):
                link = LINKS[j]
                estimator = AgglomerativeClustering(n_clusters=k, linkage=link, affinity=m )
                estimator.fit(data)
                labels = estimator.labels_
                centers = ds.compute_centroids(data, labels)
                mse.append(ds.compute_mse(data.values, labels, centers))
                sc.append(silhouette_score(data, labels))
                ds.plot_clusters(data, v2, v1, labels, centers, k,
                                 f'Hierarchical k={k} metric={m} link={link}', ax=axs[i,j])
            values_mse[m] = mse
            values_sc[m] = sc
        plt.suptitle('QOT Clustering - ' + key + ' - Metric (Hierarchical)')
        plt.savefig(subDir + 'QOT Clustering - ' + key + ' - Metric (Hierarchical)')



        print('QOT Clustering - ' + key + ' - Metric (Hierarchical) MSE vs SC')
        _, ax = plt.subplots(1, 2, figsize=(6, 3), squeeze=False)
        ds.multiple_bar_chart(LINKS, values_mse, title=f'Hierarchical MSE', xlabel='metric', ylabel='MSE', ax=ax[0, 0])
        ds.multiple_bar_chart(LINKS, values_sc, title=f'Hierarchical SC', xlabel='metric', ylabel='SC', ax=ax[0, 1], percentage=True)
        plt.suptitle('QOT Clustering - ' + key + ' - Metric (Hierarchical) MSE vs SC')
        plt.savefig(subDir + 'QOT Clustering - ' + key + ' - Metric (Hierarchical) MSE vs SC')

        plt.close("all")