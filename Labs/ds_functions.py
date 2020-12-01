import itertools
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import sklearn.metrics as metrics
import config as cfg
import datetime as dt
import matplotlib.colors as colors


COLORS = colors.CSS4_COLORS

mdates._reset_epoch_test_example()
mdates.set_epoch('0000-12-31T00:00:00')  # old epoch (pre MPL 3.3)

warnings.simplefilter("ignore")
NR_COLUMNS: int = 3
HEIGHT: int = 4


def choose_grid(nr, graphsPerRow=NR_COLUMNS):
    if nr < graphsPerRow:
        return 1, nr
    else:
        return (nr // graphsPerRow, graphsPerRow) if nr % graphsPerRow == 0 else (nr // graphsPerRow + 1, graphsPerRow)


def set_axes(xvalues: list, ax: plt.Axes = None, title: str = '', xlabel: str = '', ylabel: str = '', percentage=False):
    if ax is None:
        ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(xvalues, fontsize='small', ha='center')

    return ax


def set_locators(xvalues: list, ax: plt.Axes = None):
    if isinstance(xvalues[0], dt.datetime):
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator, defaultfmt='%Y-%m-%d'))
    else:
        ax.set_xticks(xvalues)
        ax.set_xlim(xvalues[0], xvalues[-1])

    return ax


def plot_line(xvalues: list, yvalues: list, ax: plt.Axes = None, title: str = '', xlabel: str = '',
              ylabel: str = '', percentage=False):
    ax = set_axes(xvalues, ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, percentage=percentage)
    ax = set_locators(xvalues, ax=ax)
    ax.plot(xvalues,  yvalues, c=cfg.LINE_COLOR)


def multiple_line_chart(xvalues: list, yvalues: dict, ax: plt.Axes = None, title: str = '',
                        xlabel: str = '', ylabel: str = '', percentage=False):
    ax = set_axes(xvalues, ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, percentage=percentage)
    ax = set_locators(xvalues, ax=ax)

    legend: list = []
    for name, y in yvalues.items():
        ax.plot(xvalues, y)
        legend.append(name)
    ax.legend(legend)


def bar_chart(xvalues: list, yvalues: list, ax: plt.Axes = None, title: str = '',
              xlabel: str = '', ylabel: str = '', percentage=False, edgecolor=cfg.LINE_COLOR, color=cfg.FILL_COLOR):
    ax = set_axes(xvalues, ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, percentage=percentage)
    ax.bar(xvalues, yvalues, edgecolor=edgecolor, color=color)


def multiple_bar_chart(xvalues: list, yvalues: dict, ax: plt.Axes = None, title: str = '',
                       xlabel: str = '', ylabel: str = '', percentage=False):
    ax = set_axes(xvalues, ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, percentage=percentage)

    x = np.arange(len(xvalues))  # the label locations

    width = 0.8 / (len(xvalues)*len(yvalues))
    # the width of the bars
    step = width / len(xvalues)
    i: int = 0
    for metric in yvalues:
        ax.bar(x + i*width, yvalues[metric], width=width, align='center', label=metric)
        i += 1
    ax.set_xticks(x + width/len(xvalues) - step/2)
    ax.legend(fontsize='x-small', title_fontsize='small', loc='lower center')


def plot_confusion_matrix(cnf_matrix: np.ndarray, classes_names: np.ndarray,
                          ax: plt.Axes = None, normalize: bool = False):
    cnf_matrix = np.rot90(cnf_matrix, 2)
    cnf_matrix = np.transpose(cnf_matrix)

    if ax is None:
        ax = plt.gca()
    if normalize:
        total = cnf_matrix.sum(axis=1)[:, np.newaxis]
        cm = cnf_matrix.astype('float') / total
        title = "Normalized confusion matrix"
    else:
        cm = cnf_matrix
        title = 'Confusion matrix'
    np.set_printoptions(precision=2)
    tick_marks = np.arange(0, len(classes_names), 1)
    ax.set_title(title)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes_names[::-1])
    ax.set_yticklabels(classes_names[::-1])
    ax.imshow(np.transpose(cm), interpolation='nearest', cmap=cfg.cmap_blues)

    fmt = '.2f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(i, j, format(cm[i, j], fmt), color='w', horizontalalignment="center")


def plot_evaluation_results(labels: np.ndarray, trn_y, prd_trn, tst_y, prd_tst):
    cnf_mtx_trn = metrics.confusion_matrix(trn_y, prd_trn, labels)
    tn_trn, fp_trn, fn_trn, tp_trn = cnf_mtx_trn.ravel()
    cnf_mtx_tst = metrics.confusion_matrix(tst_y, prd_tst, labels)
    tn_tst, fp_tst, fn_tst, tp_tst = cnf_mtx_tst.ravel()

    evaluation = {'Accuracy': [(tn_trn + tp_trn) / (tn_trn + tp_trn + fp_trn + fn_trn),
                               (tn_tst + tp_tst) / (tn_tst + tp_tst + fp_tst + fn_tst)],
                  'Recall': [tp_trn / (tp_trn + fn_trn), tp_tst / (tp_tst + fn_tst)],
                  'Specificity': [tn_trn / (tn_trn + fp_trn), tn_tst / (tn_tst + fp_tst)],
                  'Precision': [tp_trn / (tp_trn + fp_trn), tp_tst / (tp_tst + fp_tst)]}

    fig, axs = plt.subplots(1, 2, figsize=(2 * HEIGHT, HEIGHT))
    multiple_bar_chart(['Train', 'Test'], evaluation, ax=axs[0], title="Model's performance over Train and Test sets")
    plot_confusion_matrix(cnf_mtx_tst, labels, ax=axs[1])

def plot_evaluation_results_kfold(labels: np.ndarray, trn_y_lst, prd_trn_lst, tst_y_lst, prd_tst_lst):
    n_folds = len(trn_y_lst)
    k_evaluation = np.array([[0.0,0.0], [0.0,0.0], [0.0,0.0], [0.0,0.0]])
    k_confusion_matrix = np.array([[0,0], [0,0]])
    for ki in range(n_folds):
        cnf_mtx_trn = metrics.confusion_matrix(trn_y_lst[ki], prd_trn_lst[ki], labels)
        tn_trn, fp_trn, fn_trn, tp_trn = cnf_mtx_trn.ravel()
        cnf_mtx_tst = metrics.confusion_matrix(tst_y_lst[ki], prd_tst_lst[ki], labels)
        tn_tst, fp_tst, fn_tst, tp_tst = cnf_mtx_tst.ravel()

        k_evaluation += np.array([[(tn_trn + tp_trn) / (tn_trn + tp_trn + fp_trn + fn_trn),(tn_tst + tp_tst) / (tn_tst + tp_tst + fp_tst + fn_tst)],
                                [tp_trn / (tp_trn + fn_trn), tp_tst / (tp_tst + fn_tst)],
                                [tn_trn / (tn_trn + fp_trn), tn_tst / (tn_tst + fp_tst)],
                                [tp_trn / (tp_trn + fp_trn), tp_tst / (tp_tst + fp_tst)]])

        k_confusion_matrix += np.array([[tn_tst, fp_tst], [fn_tst, tp_tst]])

    evaluation = {'Accuracy': list(k_evaluation[0]/n_folds),
                  'Recall': list(k_evaluation[1]/n_folds),
                  'Specificity': list(k_evaluation[2]/n_folds),
                  'Precision': list(k_evaluation[3]/n_folds)}

    confusion_matrix = k_confusion_matrix/n_folds

    fig, axs = plt.subplots(1, 2, figsize=(2 * HEIGHT, HEIGHT))
    multiple_bar_chart(['Train', 'Test'], evaluation, ax=axs[0], title="Model's performance over Train and Test sets")
    plot_confusion_matrix(np.round(confusion_matrix).astype(int), labels, ax=axs[1])

def plot_roc_chart(models: dict, tstX: np.ndarray, tstY: np.ndarray, ax: plt.Axes = None, target: str = 'class'):
    if ax is None:
        ax = plt.gca()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('FP rate')
    ax.set_ylabel('TP rate')
    ax.set_title('ROC chart for %s' % target)

    ax.plot([0, 1], [0, 1], color='navy', label='random', linewidth=1, linestyle='--',  marker='')
    for clf in models.keys():
        metrics.plot_roc_curve(models[clf], tstX, tstY, ax=ax, marker='', linewidth=1)
    ax.legend(loc="lower right")

def plot_clusters(data, var1st, var2nd, clusters, centers, n_clusters: int, title: str,  ax: plt.Axes = None):
    if ax is None:
        ax = plt.gca()
    ax.scatter(data.iloc[:, var1st], data.iloc[:, var2nd], c=clusters, alpha=0.5)
    for k, col in zip(range(n_clusters), COLORS):
        cluster_center = centers[k]
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
    ax.set_title(title)
    ax.set_xlabel('var' + str(var1st))
    ax.set_ylabel('var' + str(var2nd))


def compute_centroids(data: pd.DataFrame, labels: np.ndarray) -> list:
    n_vars = data.shape[1]
    ext_data = pd.concat([data, pd.DataFrame(labels)], axis=1)
    ext_data.columns = list(data.columns) + ['cluster']
    clusters = pd.unique(labels)
    n_clusters = len(clusters)
    centers = [0] * n_clusters
    for k in range(-1, n_clusters):
        if k != -1:
            cluster = ext_data[ext_data['cluster'] == k]
            centers[k] = list(cluster.sum(axis=0))
            centers[k] = [centers[k][j]/len(cluster) if len(cluster) > 0 else 0 for j in range(n_vars)]
        else:
            centers[k] = [0]*n_vars

    return centers


def compute_mse(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    n = len(X)
    centroid_per_record = [centroids[labels[i]] for i in range(n)]
    partial = X - centroid_per_record
    partial = list(partial * partial)
    partial = [sum(el) for el in partial]
    partial = sum(partial)
    return math.sqrt(partial / n)

def compute_mae(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    n = len(X)
    centroid_per_record = [centroids[labels[i]] for i in range(n)]
    partial = abs(X - centroid_per_record)
    partial = [sum(el) for el in partial]
    partial = sum(partial)
    return partial / n