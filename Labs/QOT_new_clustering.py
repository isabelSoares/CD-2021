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


graphsDir = './Results/Clustering/Comparison/QOT/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('--------------------------')
print('-                        -')
print('-     QOT Clustering     -')
print('-                        -')
print('--------------------------')

data: pd.DataFrame = pd.read_csv('../Dataset/qsar_oral_toxicity.csv', sep=';', header=None)

# Original
original_data = data.copy()
original_data.pop(1024)

sel = VarianceThreshold(threshold=0.2)
sel.fit_transform(original_data.values)
f_to_accept = sel.get_support()
new_features = []
for i in range(len(f_to_accept)):
    if f_to_accept[i]:
        new_features.append(i)

fs_data = original_data.copy()[new_features]

fs_pca = [(False, False), (True, False), (False, True), (True, True)]

N_CLUSTERS = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
EPS = [2.5, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

fig, ax = plt.subplots(2, 3, figsize=(3*3, 4*2), squeeze=False)
fig_values_1 = {}
fig_values_2 = {}
fig_values_3 = {}
fig_values_4 = {}

fig_values_5 = {}
fig_values_6 = {}
fig_values_7 = {}
fig_values_8 = {}

def pca_function(data, subDir, name):
    variables = data.columns.values
    eixo_x = 0
    eixo_y = 4
    eixo_z = 7

    #plt.figure()
    #plt.xlabel(variables[eixo_y])
    #plt.ylabel(variables[eixo_z])
    #plt.scatter(data.iloc[:, eixo_y], data.iloc[:, eixo_z])



    print('QOT Feature Extraction - PCA')
    mean = (data.mean(axis=0)).tolist()
    centered_data = data - mean
    cov_mtx = centered_data.cov()
    eigvals, eigvecs = np.linalg.eig(cov_mtx)

    pca = PCA()
    pca.fit(centered_data)
    PC = pca.components_
    var = pca.explained_variance_

    # PLOT EXPLAINED VARIANCE RATIO
    #fig = plt.figure(figsize=(4, 4))
    #plt.title('Explained variance ratio')
    #plt.xlabel('PC')
    #plt.ylabel('ratio')
    #x_values = [str(i) for i in range(1, len(pca.components_) + 1)]
    #bwidth = 0.5
    #ax = plt.gca()
    #ax.set_xticklabels(x_values)
    #ax.set_ylim(0.0, 1.0)
    #ax.bar(x_values, pca.explained_variance_ratio_, width=bwidth)
    #ax.plot(pca.explained_variance_ratio_)
    #for i, v in enumerate(pca.explained_variance_ratio_):
    #    ax.text(i, v+0.05, f'{v*100:.1f}', ha='center', fontweight='bold')
    #plt.suptitle('QOT Feature Extraction - PCA')
    #plt.savefig(subDir + 'QOT Feature Extraction - PCA')



    print('QOT Feature Extraction - PCA 2')
    transf = pca.transform(data)

    #_, axs = plt.subplots(1, 2, figsize=(2*5, 1*5), squeeze=False)
    #axs[0,0].set_xlabel(variables[eixo_y])
    #axs[0,0].set_ylabel(variables[eixo_z])
    #axs[0,0].scatter(data.iloc[:, eixo_y], data.iloc[:, eixo_z])

    #axs[0,1].set_xlabel('PC1')
    #axs[0,1].set_ylabel('PC2')
    #axs[0,1].scatter(transf[:, 0], transf[:, 1])
    #plt.suptitle('QOT Feature Extraction - PCA')
    #plt.savefig(subDir + 'QOT Feature Extraction - PCA')



    print('Clustering after PCA')
    data = pd.DataFrame(transf[:,:2], columns=['PC1', 'PC2'])
    eixo_x = 0
    eixo_y = 1

    rows, cols = ds.choose_grid(len(N_CLUSTERS))

    print('QOT Clustering - K-Means after PCA')
    sc: list = []
    mse: list = []
    i, j = 0, 0
    for n in range(len(N_CLUSTERS)):
        k = N_CLUSTERS[n]
        estimator = KMeans(n_clusters=k)
        estimator.fit(data)
        sc.append(silhouette_score(data, estimator.labels_))
        mse.append(estimator.inertia_)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)

    fig_values_1[name] = sc
    fig_values_5[name] = mse


    print('QOT Clustering - Expectation-Maximization after PCA')
    sc: list = []
    mse: list = []
    i, j = 0, 0
    for n in range(len(N_CLUSTERS)):
        k = N_CLUSTERS[n]
        estimator = GaussianMixture(n_components=k)
        estimator.fit(data)
        labels = estimator.predict(data)
        sc.append(silhouette_score(data, labels))
        mse.append(ds.compute_mse(data.values, labels, estimator.means_))
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)

    fig_values_2[name] = sc
    fig_values_6[name] = mse

    """
    print('QOT Clustering - EPS (Density-based) after PCA')
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

    fig_values_3[name] = sc
    fig_values_7[name] = mse
    """


    """
    print('QOT Clustering - Metric (Density-based) after PCA')
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
    plt.suptitle('QOT Clustering - Metric (Density-based) after PCA')
    plt.savefig(subDir + 'QOT Clustering - Metric (Density-based) after PCA')



    print('QOT Clustering - Metric (Density-based) MSE vs MAE vs SC vs DB after PCA')
    fig, ax = plt.subplots(1, 4, figsize=(10, 3), squeeze=False)
    ds.bar_chart(METRICS, mse, title='DBSCAN MSE', xlabel='metric', ylabel='MSE', ax=ax[0, 0])
    ds.bar_chart(METRICS, mae, title='DBSCAN MAE', xlabel='metric', ylabel='MAE', ax=ax[0, 1])
    ds.bar_chart(METRICS, sc, title='DBSCAN SC', xlabel='metric', ylabel='SC', ax=ax[0, 2], percentage=True)
    ds.bar_chart(METRICS, db, title='DBSCAN DB', xlabel='metric', ylabel='DB', ax=ax[0, 3])
    plt.suptitle('QOT Clustering - Metric (Density-based) MSE vs MAE vs SC vs DB after PCA')
    plt.savefig(subDir + 'QOT Clustering - Metric (Density-based) MSE vs MAE vs SC vs DB after PCA')
    """


    print('QOT Clustering - Hierarchical after PCA')
    sc: list = []
    mse: list = []
    i, j = 0, 0
    for n in range(len(N_CLUSTERS)):
        k = N_CLUSTERS[n]
        estimator = AgglomerativeClustering(n_clusters=k)
        estimator.fit(data)
        labels = estimator.labels_
        centers = ds.compute_centroids(data, labels)
        sc.append(silhouette_score(data, labels))
        mse.append(ds.compute_mse(data.values, labels, centers))
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)

    fig_values_4[name] = sc
    fig_values_8[name] = mse


    """
    print('QOT Clustering - Metric (Hierarchical) after PCA')
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
    plt.suptitle('QOT Clustering - Metric (Hierarchical) after PCA')
    plt.savefig(subDir + 'QOT Clustering - Metric (Hierarchical) after PCA')



    print('QOT Clustering - Metric (Hierarchical) MSE vs MAE vs SC vs DB after PCA')
    _, ax = plt.subplots(1, 4, figsize=(10, 3), squeeze=False)
    ds.multiple_bar_chart(LINKS, values_mse, title=f'Hierarchical MSE', xlabel='metric', ylabel='MSE', ax=ax[0, 0])
    ds.multiple_bar_chart(LINKS, values_mae, title=f'Hierarchical MAE', xlabel='metric', ylabel='MAE', ax=ax[0, 1])
    ds.multiple_bar_chart(LINKS, values_sc, title=f'Hierarchical SC', xlabel='metric', ylabel='SC', ax=ax[0, 2], percentage=True)
    ds.multiple_bar_chart(LINKS, values_db, title=f'Hierarchical DB', xlabel='metric', ylabel='DB', ax=ax[0, 3])
    plt.suptitle('QOT Clustering - Metric (Hierarchical) MSE vs MAE vs SC vs DB after PCA')
    plt.savefig(subDir + 'QOT Clustering - Metric (Hierarchical) MSE vs MAE vs SC vs DB after PCA')
    """



for fs, pca in fs_pca:
    name = None
    if (fs, pca) == (False, False):
        name = 'Original'
        data = original_data.copy()
        subDir = graphsDir + 'Original/'
        if not os.path.exists(subDir):
            os.makedirs(subDir)
    elif (fs, pca) == (True, False):
        name = 'Only FS'
        data = fs_data.copy()
        subDir = graphsDir + 'FS/'
        if not os.path.exists(subDir):
            os.makedirs(subDir)
    elif (fs, pca) == (False, True):
        name = 'Only PCA'
        data = original_data.copy()
        subDir = graphsDir + 'PCA/'
        if not os.path.exists(subDir):
            os.makedirs(subDir)
        print()
        print()
        print('Feature Selection = ' + str(fs) + ' and PCA = ' + str(pca))
        print()
        pca_function(data, subDir, name)
        continue
    elif (fs, pca) == (True, True):
        name = 'FS + PCA'
        data = fs_data.copy()
        subDir = graphsDir + 'FS and PCA/'
        if not os.path.exists(subDir):
            os.makedirs(subDir)
        print()
        print()
        print('Feature Selection = ' + str(fs) + ' and PCA = ' + str(pca))
        print()
        pca_function(data, subDir, name)
        continue
    else:
        break

    print()
    print()
    print('Feature Selection = ' + str(fs) + ' and PCA = ' + str(pca))
    print()

    v1 = 0
    v2 = 4

    rows, cols = ds.choose_grid(len(N_CLUSTERS))

    print('QOT Clustering - K-Means')
    sc: list = []
    mse: list = []
    i, j = 0, 0
    for n in range(len(N_CLUSTERS)):
        k = N_CLUSTERS[n]
        estimator = KMeans(n_clusters=k)
        estimator.fit(data)
        sc.append(silhouette_score(data, estimator.labels_))
        mse.append(estimator.inertia_)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)


    fig_values_1[name] = sc
    fig_values_5[name] = mse


    print('QOT Clustering - Expectation-Maximization')
    sc: list = []
    mse: list = []
    i, j = 0, 0
    for n in range(len(N_CLUSTERS)):
        k = N_CLUSTERS[n]
        estimator = GaussianMixture(n_components=k)
        estimator.fit(data)
        labels = estimator.predict(data)
        sc.append(silhouette_score(data, labels))
        mse.append(ds.compute_mse(data.values, labels, estimator.means_))
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)

    fig_values_2[name] = sc
    fig_values_6[name] = mse

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
    
    fig_values_3[name] = sc
    fig_values_7[name] = mse
    """


    """
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
    """


    print('QOT Clustering - Hierarchical')
    sc: list = []
    mse: list = []
    i, j = 0, 0
    for n in range(len(N_CLUSTERS)):
        k = N_CLUSTERS[n]
        estimator = AgglomerativeClustering(n_clusters=k)
        estimator.fit(data)
        labels = estimator.labels_
        centers = ds.compute_centroids(data, labels)
        sc.append(silhouette_score(data, labels))
        mse.append(ds.compute_mse(data.values, labels, centers))
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    
    fig_values_4[name] = sc
    fig_values_8[name] = mse

    """
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

ds.multiple_line_chart(N_CLUSTERS, fig_values_1, ax=ax[0, 0], title='K-Means', xlabel='k', ylabel='SC', percentage=True)
ds.multiple_line_chart(N_CLUSTERS, fig_values_2, ax=ax[0, 1], title='EM', xlabel='k', ylabel='SC', percentage=True)
#ds.multiple_line_chart(EPS, fig_values_3, ax=ax[0, 2], title='EPS', xlabel='k', ylabel='SC', percentage=True)
ds.multiple_line_chart(N_CLUSTERS, fig_values_4, ax=ax[0, 2], title='Hierarchical', xlabel='k', ylabel='SC', percentage=True)

ds.multiple_line_chart(N_CLUSTERS, fig_values_5, ax=ax[1, 0], title='K-Means', xlabel='k', ylabel='MSE')
ds.multiple_line_chart(N_CLUSTERS, fig_values_6, ax=ax[1, 1], title='EM', xlabel='k', ylabel='MSE')
#ds.multiple_line_chart(EPS, fig_values_7, ax=ax[1, 2], title='EPS', xlabel='k', ylabel='MSE')
ds.multiple_line_chart(N_CLUSTERS, fig_values_8, ax=ax[1, 2], title='Hierarchical', xlabel='k', ylabel='MSE')

plt.suptitle('HFCR Clustering - SC Comparison')
plt.savefig(graphsDir + 'HFCR Clustering - SC and MSE Comparison')