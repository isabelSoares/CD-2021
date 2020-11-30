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


graphsDir = './Results/Clustering/HFCR/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('---------------------------')
print('-                         -')
print('-     HFCR Clustering     -')
print('-                         -')
print('---------------------------')

data: pd.DataFrame = pd.read_csv('../Dataset/heart_failure_clinical_records_dataset.csv')
original_data = data.copy()
datas_scaled = prepfunctions.prepare_dataset(data, 'DEATH_EVENT', True, True)

scaling_pca = [(False, False), (True, False), (False, True), (True, True)]

def pca_function(data, subDir):
    data.pop('DEATH_EVENT')

    variables = data.columns.values
    eixo_x = 0
    eixo_y = 4
    eixo_z = 7

    plt.figure()
    plt.xlabel(variables[eixo_y])
    plt.ylabel(variables[eixo_z])
    plt.scatter(data.iloc[:, eixo_y], data.iloc[:, eixo_z])



    print('HFCR Feature Extraction - PCA')
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
    plt.suptitle('HFCR Feature Extraction - PCA')
    plt.savefig(subDir + 'HFCR Feature Extraction - PCA')



    print('HFCR Feature Extraction - PCA 2')
    transf = pca.transform(data)

    _, axs = plt.subplots(1, 2, figsize=(2*5, 1*5), squeeze=False)
    axs[0,0].set_xlabel(variables[eixo_y])
    axs[0,0].set_ylabel(variables[eixo_z])
    axs[0,0].scatter(data.iloc[:, eixo_y], data.iloc[:, eixo_z])

    axs[0,1].set_xlabel('PC1')
    axs[0,1].set_ylabel('PC2')
    axs[0,1].scatter(transf[:, 0], transf[:, 1])
    plt.suptitle('HFCR Feature Extraction - PCA')
    plt.savefig(subDir + 'HFCR Feature Extraction - PCA')



    print('Clustering after PCA')
    data = pd.DataFrame(transf[:,:2], columns=['PC1', 'PC2'])
    eixo_x = 0
    eixo_y = 1

    N_CLUSTERS = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
    rows, cols = ds.choose_grid(len(N_CLUSTERS))

    print('HFCR Clustering - K-Means after PCA')
    mse: list = []
    mae: list = []
    sc: list = []
    db: list = []
    _, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
    i, j = 0, 0
    for n in range(len(N_CLUSTERS)):
        k = N_CLUSTERS[n]
        estimator = KMeans(n_clusters=k)
        estimator.fit(data)
        mse.append(estimator.inertia_)
        mae.append(ds.compute_mae(data.values, estimator.labels_, estimator.cluster_centers_))
        sc.append(silhouette_score(data, estimator.labels_))
        db.append(davies_bouldin_score(data, estimator.labels_))
        ds.plot_clusters(data, eixo_x, eixo_y, estimator.labels_.astype(float), estimator.cluster_centers_, k,
                         f'KMeans k={k}', ax=axs[i,j])
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    plt.suptitle('HFCR Clustering - K-Means after PCA')
    plt.savefig(subDir + 'HFCR Clustering - K-Means after PCA')



    print('HFCR Clustering - K-Means after PCA MSE vs MAE vs SC vs DB after PCA')
    fig, ax = plt.subplots(1, 4, figsize=(10, 3), squeeze=False)
    ds.plot_line(N_CLUSTERS, mse, title='KMeans MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
    ds.plot_line(N_CLUSTERS, mae, title='KMeans MAE', xlabel='k', ylabel='MAE', ax=ax[0, 1])
    ds.plot_line(N_CLUSTERS, sc, title='KMeans SC', xlabel='k', ylabel='SC', ax=ax[0, 2], percentage=True)
    ds.plot_line(N_CLUSTERS, db, title='KMeans DB', xlabel='k', ylabel='DB', ax=ax[0, 3])
    plt.suptitle('HFCR Clustering - K-Means after PCA MSE vs MAE vs SC vs DB after PCA')
    plt.savefig(subDir + 'HFCR Clustering - K-Means after PCA MSE vs MAE vs SC vs DB after PCA')



    print('HFCR Clustering - Expectation-Maximization after PCA')
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
        ds.plot_clusters(data, eixo_x, eixo_y, labels.astype(float), estimator.means_, k,
                         f'EM k={k}', ax=axs[i,j])
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    plt.suptitle('HFCR Clustering - Expectation-Maximization after PCA')
    plt.savefig(subDir + 'HFCR Clustering - Expectation-Maximization after PCA')



    print('HFCR Clustering - Expectation-Maximization MSE vs MAE vs SC vs DB after PCA')
    fig, ax = plt.subplots(1, 4, figsize=(10, 3), squeeze=False)
    ds.plot_line(N_CLUSTERS, mse, title='EM MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
    ds.plot_line(N_CLUSTERS, mae, title='EM MAE', xlabel='k', ylabel='MAE', ax=ax[0, 1])
    ds.plot_line(N_CLUSTERS, sc, title='EM SC', xlabel='k', ylabel='SC', ax=ax[0, 2], percentage=True)
    ds.plot_line(N_CLUSTERS, db, title='EM DB', xlabel='k', ylabel='DB', ax=ax[0, 3])
    plt.suptitle('HFCR Clustering - Expectation-Maximization MSE vs MAE vs SC vs DB after PCA')
    plt.savefig(subDir + 'HFCR Clustering - Expectation-Maximization MSE vs MAE vs SC vs DB after PCA')



    print('HFCR Clustering - EPS (Density-based) after PCA')
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
            ds.plot_clusters(data, eixo_x, eixo_y, labels.astype(float), estimator.components_, k,
                             f'DBSCAN eps={EPS[n]} k={k}', ax=axs[i,j])
            i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
        else:
            mse.append(0)
            mae.append(0)
            sc.append(0)
            db.append(0)
    plt.suptitle('HFCR Clustering - EPS (Density-based) after PCA')
    plt.savefig(subDir + 'HFCR Clustering - EPS (Density-based) after PCA')



    print('HFCR Clustering - EPS (Density-based) MSE vs MAE vs SC vs DB after PCA')
    fig, ax = plt.subplots(1, 4, figsize=(10, 3), squeeze=False)
    ds.plot_line(EPS, mse, title='DBSCAN MSE', xlabel='eps', ylabel='MSE', ax=ax[0, 0])
    ds.plot_line(EPS, mae, title='DBSCAN MAE', xlabel='eps', ylabel='MAE', ax=ax[0, 1])
    ds.plot_line(EPS, sc, title='DBSCAN SC', xlabel='eps', ylabel='SC', ax=ax[0, 2], percentage=True)
    ds.plot_line(EPS, db, title='DBSCAN DB', xlabel='eps', ylabel='DB', ax=ax[0, 3])
    plt.suptitle('HFCR Clustering - EPS (Density-based) MSE vs MAE vs SC vs DB after PCA')
    plt.savefig(subDir + 'HFCR Clustering - EPS (Density-based) MSE vs MAE vs SC vs DB after PCA')



    print('HFCR Clustering - Metric (Density-based) after PCA')
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
            ds.plot_clusters(data, eixo_x, eixo_y, labels.astype(float), estimator.components_, k,
                             f'DBSCAN metric={METRICS[n]} eps={distances[n]:.2f} k={k}', ax=axs[i,j])
        else:
            print(k)
            mse.append(0)
            mae.append(0)
            sc.append(0)
            db.append(0)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    plt.suptitle('HFCR Clustering - Metric (Density-based) after PCA')
    plt.savefig(subDir + 'HFCR Clustering - Metric (Density-based) after PCA')



    print('HFCR Clustering - Metric (Density-based) MSE vs MAE vs SC vs DB after PCA')
    fig, ax = plt.subplots(1, 4, figsize=(10, 3), squeeze=False)
    ds.bar_chart(METRICS, mse, title='DBSCAN MSE', xlabel='metric', ylabel='MSE', ax=ax[0, 0])
    ds.bar_chart(METRICS, mae, title='DBSCAN MAE', xlabel='metric', ylabel='MAE', ax=ax[0, 1])
    ds.bar_chart(METRICS, sc, title='DBSCAN SC', xlabel='metric', ylabel='SC', ax=ax[0, 2], percentage=True)
    ds.bar_chart(METRICS, db, title='DBSCAN DB', xlabel='metric', ylabel='DB', ax=ax[0, 3])
    plt.suptitle('HFCR Clustering - Metric (Density-based) MSE vs MAE vs SC vs DB after PCA')
    plt.savefig(subDir + 'HFCR Clustering - Metric (Density-based) MSE vs MAE vs SC vs DB after PCA')



    print('HFCR Clustering - Hierarchical after PCA')
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
        ds.plot_clusters(data, eixo_x, eixo_y, labels, centers, k,
                         f'Hierarchical k={k}', ax=axs[i,j])
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    plt.suptitle('HFCR Clustering - Hierarchical after PCA')
    plt.savefig(subDir + 'HFCR Clustering - Hierarchical after PCA')



    print('HFCR Clustering - Hierarchical MSE vs MAE vs SC vs DB after PCA')
    fig, ax = plt.subplots(1, 4, figsize=(10, 3), squeeze=False)
    ds.plot_line(N_CLUSTERS, mse, title='Hierarchical MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
    ds.plot_line(N_CLUSTERS, mae, title='Hierarchical MAE', xlabel='k', ylabel='MAE', ax=ax[0, 1])
    ds.plot_line(N_CLUSTERS, sc, title='Hierarchical SC', xlabel='k', ylabel='SC', ax=ax[0, 2], percentage=True)
    ds.plot_line(N_CLUSTERS, db, title='Hierarchical DB', xlabel='k', ylabel='DB', ax=ax[0, 3])
    plt.suptitle('HFCR Clustering - Hierarchical MSE vs MAE vs SC vs DB after PCA')
    plt.savefig(subDir + 'HFCR Clustering - Hierarchical MSE vs MAE vs SC vs DB after PCA')



    print('HFCR Clustering - Metric (Hierarchical) after PCA')
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
            ds.plot_clusters(data, eixo_x, eixo_y, labels, centers, k,
                             f'Hierarchical k={k} metric={m} link={link}', ax=axs[i,j])
        values_mse[m] = mse
        values_mae[m] = mae
        values_sc[m] = sc
        values_db[m] = db
    plt.suptitle('HFCR Clustering - Metric (Hierarchical) after PCA')
    plt.savefig(subDir + 'HFCR Clustering - Metric (Hierarchical) after PCA')



    print('HFCR Clustering - Metric (Hierarchical) MSE vs MAE vs SC vs DB after PCA')
    _, ax = plt.subplots(1, 4, figsize=(10, 3), squeeze=False)
    ds.multiple_bar_chart(LINKS, values_mse, title=f'Hierarchical MSE', xlabel='metric', ylabel='MSE', ax=ax[0, 0])
    ds.multiple_bar_chart(LINKS, values_mae, title=f'Hierarchical MAE', xlabel='metric', ylabel='MAE', ax=ax[0, 1])
    ds.multiple_bar_chart(LINKS, values_sc, title=f'Hierarchical SC', xlabel='metric', ylabel='SC', ax=ax[0, 2], percentage=True)
    ds.multiple_bar_chart(LINKS, values_db, title=f'Hierarchical DB', xlabel='metric', ylabel='DB', ax=ax[0, 3])
    plt.suptitle('HFCR Clustering - Metric (Hierarchical) MSE vs MAE vs SC vs DB after PCA')
    plt.savefig(subDir + 'HFCR Clustering - Metric (Hierarchical) MSE vs MAE vs SC vs DB after PCA')

    plt.close("all")
    plt.clf()



for key in ['Original']:
    for scaling, pca in scaling_pca:
        if (scaling, pca) == (False, False):
            data = original_data.copy()
            subDir = graphsDir + 'Original/'
            if not os.path.exists(subDir):
                os.makedirs(subDir)
        elif (scaling, pca) == (True, False):
            data = datas_scaled[key].copy()
            subDir = graphsDir + 'Scaling/'
            if not os.path.exists(subDir):
                os.makedirs(subDir)
        elif (scaling, pca) == (False, True):
            data = original_data.copy()
            subDir = graphsDir + 'PCA/'
            if not os.path.exists(subDir):
            	os.makedirs(subDir)
            print()
            print()
            print('Scaling = ' + str(scaling) + ' and PCA = ' + str(pca))
            print()
            pca_function(data, subDir)
            continue
        elif (scaling, pca) == (True, True):
            data = datas_scaled[key].copy()
            subDir = graphsDir + 'Scaling and PCA/'
            if not os.path.exists(subDir):
                os.makedirs(subDir)
            print()
            print()
            print('Scaling = ' + str(scaling) + ' and PCA = ' + str(pca))
            print()
            pca_function(data, subDir)
            continue
        else:
            break

        print()
        print()
        print('Scaling = ' + str(scaling) + ' and PCA = ' + str(pca))
        print()
        data.pop('DEATH_EVENT')

        v1 = data.columns.get_loc("age")  # age
        v2 = data.columns.get_loc("time") # time

        N_CLUSTERS = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
        rows, cols = ds.choose_grid(len(N_CLUSTERS))

        print('HFCR Clustering - K-Means')
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
        plt.suptitle('HFCR Clustering - K-Means')
        plt.savefig(subDir + 'HFCR Clustering - K-Means')


        print('HFCR Clustering - K-Means MSE vs MAE vs SC vs DB')
        fig, ax = plt.subplots(1, 4, figsize=(10, 3), squeeze=False)
        ds.plot_line(N_CLUSTERS, mse, title='KMeans MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
        ds.plot_line(N_CLUSTERS, mae, title='KMeans MAE', xlabel='k', ylabel='MAE', ax=ax[0, 1])
        ds.plot_line(N_CLUSTERS, sc, title='KMeans SC', xlabel='k', ylabel='SC', ax=ax[0, 2], percentage=True)
        ds.plot_line(N_CLUSTERS, db, title='KMeans DB', xlabel='k', ylabel='DB', ax=ax[0, 3])
        plt.suptitle('HFCR Clustering - K-Means MSE vs MAE vs SC vs DB')
        plt.savefig(subDir + 'HFCR Clustering - K-Means MSE vs MAE vs SC vs DB')


        print('HFCR Clustering - Expectation-Maximization')
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
        plt.suptitle('HFCR Clustering - Expectation-Maximization')
        plt.savefig(subDir + 'HFCR Clustering - Expectation-Maximization')


        print('HFCR Clustering - Expectation-Maximization MSE vs MAE vs SC vs DB')
        fig, ax = plt.subplots(1, 4, figsize=(10, 3), squeeze=False)
        ds.plot_line(N_CLUSTERS, mse, title='EM MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
        ds.plot_line(N_CLUSTERS, mae, title='EM MAE', xlabel='k', ylabel='MAE', ax=ax[0, 1])
        ds.plot_line(N_CLUSTERS, sc, title='EM SC', xlabel='k', ylabel='SC', ax=ax[0, 2], percentage=True)
        ds.plot_line(N_CLUSTERS, db, title='EM DB', xlabel='k', ylabel='DB', ax=ax[0, 3])
        plt.suptitle('HFCR Clustering - Expectation-Maximization MSE vs MAE vs SC vs DB')
        plt.savefig(subDir + 'HFCR Clustering - Expectation-Maximization MSE vs MAE vs SC vs DB')


        print('HFCR Clustering - EPS (Density-based)')
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
        plt.suptitle('HFCR Clustering - EPS (Density-based)')
        plt.savefig(subDir + 'HFCR Clustering - EPS (Density-based)')


        print('HFCR Clustering - EPS (Density-based) MSE vs MAE vs SC vs DB')
        fig, ax = plt.subplots(1, 4, figsize=(10, 3), squeeze=False)
        ds.plot_line(EPS, mse, title='DBSCAN MSE', xlabel='eps', ylabel='MSE', ax=ax[0, 0])
        ds.plot_line(EPS, mae, title='DBSCAN MAE', xlabel='eps', ylabel='MAE', ax=ax[0, 1])
        ds.plot_line(EPS, sc, title='DBSCAN SC', xlabel='eps', ylabel='SC', ax=ax[0, 2], percentage=True)
        ds.plot_line(EPS, db, title='DBSCAN DB', xlabel='eps', ylabel='DB', ax=ax[0, 3])
        plt.suptitle('HFCR Clustering - EPS (Density-based) MSE vs MAE vs SC vs DB')
        plt.savefig(subDir + 'HFCR Clustering - EPS (Density-based) MSE vs MAE vs SC vs DB')



        print('HFCR Clustering - Metric (Density-based)')
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
        plt.suptitle('HFCR Clustering - Metric (Density-based)')
        plt.savefig(subDir + 'HFCR Clustering - Metric (Density-based)')


        print('HFCR Clustering - Metric (Density-based) MSE vs MAE vs SC vs DB')
        fig, ax = plt.subplots(1, 4, figsize=(10, 3), squeeze=False)
        ds.bar_chart(METRICS, mse, title='DBSCAN MSE', xlabel='metric', ylabel='MSE', ax=ax[0, 0])
        ds.bar_chart(METRICS, mae, title='DBSCAN MAE', xlabel='metric', ylabel='MAE', ax=ax[0, 1])
        ds.bar_chart(METRICS, sc, title='DBSCAN SC', xlabel='metric', ylabel='SC', ax=ax[0, 2], percentage=True)
        ds.bar_chart(METRICS, db, title='DBSCAN DB', xlabel='metric', ylabel='DB', ax=ax[0, 3])
        plt.suptitle('HFCR Clustering - Metric (Density-based) MSE vs MAE vs SC vs DB')
        plt.savefig(subDir + 'HFCR Clustering - Metric (Density-based) MSE vs MAE vs SC vs DB')



        print('HFCR Clustering - Hierarchical')
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
        plt.suptitle('HFCR Clustering - Hierarchical')
        plt.savefig(subDir + 'HFCR Clustering - Hierarchical')



        print('HFCR Clustering - Hierarchical MSE vs MAE vs SC vs DB')
        fig, ax = plt.subplots(1, 4, figsize=(10, 3), squeeze=False)
        ds.plot_line(N_CLUSTERS, mse, title='Hierarchical MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
        ds.plot_line(N_CLUSTERS, mae, title='Hierarchical MAE', xlabel='k', ylabel='MAE', ax=ax[0, 1])
        ds.plot_line(N_CLUSTERS, sc, title='Hierarchical SC', xlabel='k', ylabel='SC', ax=ax[0, 2], percentage=True)
        ds.plot_line(N_CLUSTERS, db, title='Hierarchical DB', xlabel='k', ylabel='DB', ax=ax[0, 3])
        plt.suptitle('HFCR Clustering - Hierarchical MSE vs MAE vs SC vs DB')
        plt.savefig(subDir + 'HFCR Clustering - Hierarchical MSE vs MAE vs SC vs DB')


        print('HFCR Clustering - Metric (Hierarchical)')
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
        plt.suptitle('HFCR Clustering - Metric (Hierarchical)')
        plt.savefig(subDir + 'HFCR Clustering - Metric (Hierarchical)')



        print('HFCR Clustering - Metric (Hierarchical) MSE vs MAE vs SC vs DB')
        _, ax = plt.subplots(1, 4, figsize=(10, 3), squeeze=False)
        ds.multiple_bar_chart(LINKS, values_mse, title=f'Hierarchical MSE', xlabel='metric', ylabel='MSE', ax=ax[0, 0])
        ds.multiple_bar_chart(LINKS, values_mae, title=f'Hierarchical MAE', xlabel='metric', ylabel='MAE', ax=ax[0, 1])
        ds.multiple_bar_chart(LINKS, values_sc, title=f'Hierarchical SC', xlabel='metric', ylabel='SC', ax=ax[0, 2], percentage=True)
        ds.multiple_bar_chart(LINKS, values_db, title=f'Hierarchical DB', xlabel='metric', ylabel='DB', ax=ax[0, 3])
        plt.suptitle('HFCR Clustering - Metric (Hierarchical) MSE vs MAE vs SC vs DB')
        plt.savefig(subDir + 'HFCR Clustering - Metric (Hierarchical) MSE vs MAE vs SC vs DB')

        plt.close("all")
        plt.clf()