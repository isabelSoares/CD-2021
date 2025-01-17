import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ds_functions as ds
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
import data_preparation_functions as prepfunctions

graphsDir = './Results/Clustering/FeatureExtraction/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('-----------------------------------')
print('-                                 -')
print('-     HFCR Feature Extraction     -')
print('-                                 -')
print('-----------------------------------')

data: pd.DataFrame = pd.read_csv('../Dataset/heart_failure_clinical_records_dataset.csv')
datas = prepfunctions.prepare_dataset(data, 'DEATH_EVENT', True, True)
featured_datas = prepfunctions.mask_feature_selection(datas, 'DEATH_EVENT', False, './Results/FeatureSelection/HFCR Feature Selection - Features')

for key in datas:
    for do_feature_eng in [False, True]:
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

        data.pop('DEATH_EVENT')

        variables = data.columns.values
        eixo_x = 0
        eixo_y = 4
        eixo_z = 7

        plt.figure()
        plt.xlabel(variables[eixo_y])
        plt.ylabel(variables[eixo_z])
        plt.scatter(data.iloc[:, eixo_y], data.iloc[:, eixo_z])



        print('HFCR Feature Extraction - ' + key + ' - PCA')
        mean = (data.mean(axis=0)).tolist()
        centered_data = data - mean
        cov_mtx = centered_data.cov()
        eigvals, eigvecs = np.linalg.eig(cov_mtx)

        pca = PCA()
        pca.fit(centered_data)
        PC = pca.components_
        var = pca.explained_variance_

        # PLOT EXPLAINED VARIANCE RATIO
        fig = plt.figure(figsize=(4, 4))
        plt.title('Explained variance ratio')
        plt.xlabel('PC')
        plt.ylabel('ratio')
        x_values = [str(i) for i in range(1, len(pca.components_) + 1)]
        bwidth = 0.5
        ax = plt.gca()
        ax.set_xticklabels(x_values)
        ax.set_ylim(0.0, 1.0)
        ax.bar(x_values, pca.explained_variance_ratio_, width=bwidth)
        ax.plot(pca.explained_variance_ratio_)
        for i, v in enumerate(pca.explained_variance_ratio_):
            ax.text(i, v+0.05, f'{v*100:.1f}', ha='center', fontweight='bold')
        plt.suptitle('HFCR Feature Extraction - ' + key + ' - PCA')
        plt.savefig(subDir + 'HFCR Feature Extraction - ' + key + ' - PCA')



        print('HFCR Feature Extraction - ' + key + ' - PCA 2')
        transf = pca.transform(data)

        _, axs = plt.subplots(1, 2, figsize=(2*5, 1*5), squeeze=False)
        axs[0,0].set_xlabel(variables[eixo_y])
        axs[0,0].set_ylabel(variables[eixo_z])
        axs[0,0].scatter(data.iloc[:, eixo_y], data.iloc[:, eixo_z])

        axs[0,1].set_xlabel('PC1')
        axs[0,1].set_ylabel('PC2')
        axs[0,1].scatter(transf[:, 0], transf[:, 1])
        plt.suptitle('HFCR Feature Extraction - ' + key + ' - PCA 2')
        plt.savefig(subDir + 'HFCR Feature Extraction - ' + key + ' - PCA 2')



        print('Clustering after PCA')
        data = pd.DataFrame(transf[:,:2], columns=['PC1', 'PC2'])
        eixo_x = 0
        eixo_y = 1

        N_CLUSTERS = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
        rows, cols = ds.choose_grid(len(N_CLUSTERS))

        print('HFCR Clustering - ' + key + ' - K-Means after PCA')
        mse: list = []
        sc: list = []
        _, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
        i, j = 0, 0
        for n in range(len(N_CLUSTERS)):
            k = N_CLUSTERS[n]
            estimator = KMeans(n_clusters=k)
            estimator.fit(data)
            mse.append(estimator.inertia_)
            sc.append(silhouette_score(data, estimator.labels_))
            ds.plot_clusters(data, eixo_x, eixo_y, estimator.labels_.astype(float), estimator.cluster_centers_, k,
                             f'KMeans k={k}', ax=axs[i,j])
            i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
        plt.suptitle('HFCR Clustering - ' + key + ' - K-Means after PCA')
        plt.savefig(subDir + 'HFCR Clustering - ' + key + ' - K-Means after PCA')



        print('HFCR Clustering - ' + key + ' - K-Means after PCA MSE vs SC after PCA')
        fig, ax = plt.subplots(1, 2, figsize=(6, 3), squeeze=False)
        ds.plot_line(N_CLUSTERS, mse, title='KMeans MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
        ds.plot_line(N_CLUSTERS, sc, title='KMeans SC', xlabel='k', ylabel='SC', ax=ax[0, 1], percentage=True)
        plt.suptitle('HFCR Clustering - ' + key + ' - K-Means after PCA MSE vs SC after PCA')
        plt.savefig(subDir + 'HFCR Clustering - ' + key + ' - K-Means after PCA MSE vs SC after PCA')



        print('HFCR Clustering - ' + key + ' - Expectation-Maximization after PCA')
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
            ds.plot_clusters(data, eixo_x, eixo_y, labels.astype(float), estimator.means_, k,
                             f'EM k={k}', ax=axs[i,j])
            i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
        plt.suptitle('HFCR Clustering - ' + key + ' - Expectation-Maximization after PCA')
        plt.savefig(subDir + 'HFCR Clustering - ' + key + ' - Expectation-Maximization after PCA')



        print('HFCR Clustering - ' + key + ' - Expectation-Maximization MSE vs SC after PCA')
        fig, ax = plt.subplots(1, 2, figsize=(6, 3), squeeze=False)
        ds.plot_line(N_CLUSTERS, mse, title='EM MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
        ds.plot_line(N_CLUSTERS, sc, title='EM SC', xlabel='k', ylabel='SC', ax=ax[0, 1], percentage=True)
        plt.suptitle('HFCR Clustering - ' + key + ' - Expectation-Maximization MSE vs SC after PCA')
        plt.savefig(subDir + 'HFCR Clustering - ' + key + ' - Expectation-Maximization MSE vs SC after PCA')



        print('HFCR Clustering - ' + key + ' - EPS (Density-based) after PCA')
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
                ds.plot_clusters(data, eixo_x, eixo_y, labels.astype(float), estimator.components_, k,
                                 f'DBSCAN eps={EPS[n]} k={k}', ax=axs[i,j])
                i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
            else:
                mse.append(0)
                sc.append(0)
        plt.suptitle('HFCR Clustering - ' + key + ' - EPS (Density-based) after PCA')
        plt.savefig(subDir + 'HFCR Clustering - ' + key + ' - EPS (Density-based) after PCA')



        print('HFCR Clustering - ' + key + ' - EPS (Density-based) MSE vs SC after PCA')
        fig, ax = plt.subplots(1, 2, figsize=(6, 3), squeeze=False)
        ds.plot_line(EPS, mse, title='DBSCAN MSE', xlabel='eps', ylabel='MSE', ax=ax[0, 0])
        ds.plot_line(EPS, sc, title='DBSCAN SC', xlabel='eps', ylabel='SC', ax=ax[0, 1], percentage=True)
        plt.suptitle('HFCR Clustering - ' + key + ' - EPS (Density-based) MSE vs SC after PCA')
        plt.savefig(subDir + 'HFCR Clustering - ' + key + ' - EPS (Density-based) MSE vs SC after PCA')



        print('HFCR Clustering - ' + key + ' - Metric (Density-based) after PCA')
        METRICS = ['euclidean', 'cityblock', 'chebyshev', 'cosine', 'jaccard']
        distances = []
        for m in METRICS:
            dist = np.mean(np.mean(squareform(pdist(data.values, metric=m))))
            distances.append(dist)

        print('AVG distances among records', distances)
        distances[0] = 80
        distances[1] = 50
        distances[2] = 80
        distances[3] = 0.0005
        distances[4] = 0.0009
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
                ds.plot_clusters(data, eixo_x, eixo_y, labels.astype(float), estimator.components_, k,
                                 f'DBSCAN metric={METRICS[n]} eps={distances[n]:.2f} k={k}', ax=axs[i,j])
            else:
                print(k)
                mse.append(0)
                sc.append(0)
            i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
        plt.suptitle('HFCR Clustering - ' + key + ' - Metric (Density-based) after PCA')
        plt.savefig(subDir + 'HFCR Clustering - ' + key + ' - Metric (Density-based) after PCA')



        print('HFCR Clustering - ' + key + ' - Metric (Density-based) MSE vs SC after PCA')
        fig, ax = plt.subplots(1, 2, figsize=(6, 3), squeeze=False)
        ds.bar_chart(METRICS, mse, title='DBSCAN MSE', xlabel='metric', ylabel='MSE', ax=ax[0, 0])
        ds.bar_chart(METRICS, sc, title='DBSCAN SC', xlabel='metric', ylabel='SC', ax=ax[0, 1], percentage=True)
        plt.suptitle('HFCR Clustering - ' + key + ' - Metric (Density-based) MSE vs SC after PCA')
        plt.savefig(subDir + 'HFCR Clustering - ' + key + ' - Metric (Density-based) MSE vs SC after PCA')



        print('HFCR Clustering - ' + key + ' - Hierarchical after PCA')
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
            ds.plot_clusters(data, eixo_x, eixo_y, labels, centers, k,
                             f'Hierarchical k={k}', ax=axs[i,j])
            i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
        plt.suptitle('HFCR Clustering - ' + key + ' - Hierarchical after PCA')
        plt.savefig(subDir + 'HFCR Clustering - ' + key + ' - Hierarchical after PCA')



        print('HFCR Clustering - ' + key + ' - Hierarchical MSE vs SC after PCA')
        fig, ax = plt.subplots(1, 2, figsize=(6, 3), squeeze=False)
        ds.plot_line(N_CLUSTERS, mse, title='Hierarchical MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
        ds.plot_line(N_CLUSTERS, sc, title='Hierarchical SC', xlabel='k', ylabel='SC', ax=ax[0, 1], percentage=True)
        plt.suptitle('HFCR Clustering - ' + key + ' - Hierarchical MSE vs SC after PCA')
        plt.savefig(subDir + 'HFCR Clustering - ' + key + ' - Hierarchical MSE vs SC after PCA')



        print('HFCR Clustering - ' + key + ' - Metric (Hierarchical) after PCA')
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
                ds.plot_clusters(data, eixo_x, eixo_y, labels, centers, k,
                                 f'Hierarchical k={k} metric={m} link={link}', ax=axs[i,j])
            values_mse[m] = mse
            values_sc[m] = sc
        plt.suptitle('HFCR Clustering - ' + key + ' - Metric (Hierarchical) after PCA')
        plt.savefig(subDir + 'HFCR Clustering - ' + key + ' - Metric (Hierarchical) after PCA')



        print('HFCR Clustering - ' + key + ' - Metric (Hierarchical) MSE vs SC after PCA')
        _, ax = plt.subplots(1, 2, figsize=(6, 3), squeeze=False)
        ds.multiple_bar_chart(LINKS, values_mse, title=f'Hierarchical MSE', xlabel='metric', ylabel='MSE', ax=ax[0, 0])
        ds.multiple_bar_chart(LINKS, values_sc, title=f'Hierarchical SC', xlabel='metric', ylabel='SC', ax=ax[0, 1], percentage=True)
        plt.suptitle('HFCR Clustering - ' + key + ' - Metric (Hierarchical) MSE vs SC after PCA')
        plt.savefig(subDir + 'HFCR Clustering - ' + key + ' - Metric (Hierarchical) MSE vs SC after PCA')

        plt.close("all")