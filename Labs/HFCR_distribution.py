import pandas as pd
import matplotlib.pyplot as plt
import ds_functions as ds
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
graphsDir = './Results/'
print('-------------------------')
print('-                       -')
print('-   HFCR Distribution   -')
print('-                       -')
print('-------------------------')

print('HFCR Distribution - Numeric variables description')
data = pd.read_csv('../../Dataset/heart_failure_clinical_records_dataset.csv')
print(data.describe())
data.describe().to_csv(graphsDir + 'HFCR Distribution - Numeric variables description.csv')
print()

print('HFCR Distribution - Boxplot')
data.boxplot(rot=45, figsize=(9,12), whis=1.5)
plt.suptitle('HFCR Distribution - Boxplot')
plt.savefig(graphsDir + 'HFCR Distribution - Boxplot')
print()

print('HFCR Distribution - Boxplot for each variable')
numeric_vars = data.select_dtypes(include='number').columns
rows, cols = ds.choose_grid(len(numeric_vars))
fig, axs = plt.subplots(rows, cols, figsize=(cols*ds.HEIGHT, rows*ds.HEIGHT))
i, j = 0, 0
for n in range(len(numeric_vars)):
    axs[i, j].set_title('Boxplot for %s'%numeric_vars[n])
    axs[i, j].boxplot(data[numeric_vars[n]].dropna().values, whis=1.5)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
plt.suptitle('HFCR Distribution - Boxplot for each variable')
plt.savefig(graphsDir + 'HFCR Distribution - Boxplot for each variable')
print()

print('HFCR Distribution - Histograms')
fig, axs = plt.subplots(rows, cols, figsize=(cols*ds.HEIGHT, rows*ds.HEIGHT))
i, j = 0, 0
for n in range(len(numeric_vars)):
    axs[i, j].set_title('Histogram for %s'%numeric_vars[n])
    axs[i, j].set_xlabel(numeric_vars[n])
    axs[i, j].set_ylabel("nr records")
    axs[i, j].hist(data[numeric_vars[n]].dropna().values, 'auto')
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
plt.suptitle('HFCR Distribution - Histograms')
plt.savefig(graphsDir + 'HFCR Distribution - Histograms')
print()

import seaborn as sns
print('HFCR Distribution - Histograms with the best fit')
fig, axs = plt.subplots(rows, cols, figsize=(cols*ds.HEIGHT, rows*ds.HEIGHT))
i, j = 0, 0
for n in range(len(numeric_vars)):
    axs[i, j].set_title('Histogram with trend for %s'%numeric_vars[n])
    sns.distplot(data[numeric_vars[n]].dropna().values, norm_hist=True, ax=axs[i, j], axlabel=numeric_vars[n])
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
plt.suptitle('HFCR Distribution - Histograms with the best fit')
plt.savefig(graphsDir + 'HFCR Distribution - Histograms with the best fit')
print()

import scipy.stats as _stats
import numpy as np

print('HFCR Distribution - Histograms compared to known distributions')
def compute_known_distributions(x_values: list) -> dict:
    distributions = dict()
    # Gaussian
    mean, sigma = _stats.norm.fit(x_values)
    distributions['Normal(%.1f,%.2f)'%(mean,sigma)] = _stats.norm.pdf(x_values, mean, sigma)
    # Exponential
    loc, scale = _stats.expon.fit(x_values)
    distributions['Exp(%.2f)'%(1/scale)] = _stats.expon.pdf(x_values, loc, scale)
    # LogNorm
    sigma, loc, scale = _stats.lognorm.fit(x_values)
    distributions['LogNor(%.1f,%.2f)'%(np.log(scale),sigma)] = _stats.lognorm.pdf(x_values, sigma, loc, scale)
    return distributions

def histogram_with_distributions(ax: plt.Axes, series: pd.Series, var: str):
    values = series.sort_values().values
    ax.hist(values, 20, density=True)
    distributions = compute_known_distributions(values)
    ds.multiple_line_chart(values, distributions, ax=ax, title='Best fit for %s'%var, xlabel=var, ylabel='')

fig, axs = plt.subplots(rows, cols, figsize=(cols*ds.HEIGHT, rows*ds.HEIGHT))
i, j = 0, 0
for n in range(len(numeric_vars)):
    histogram_with_distributions(axs[i, j], data[numeric_vars[n]].dropna(), numeric_vars[n])
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
plt.suptitle('HFCR Distribution - Histograms compared to known distributions')
plt.savefig(graphsDir + 'HFCR Distribution - Histograms compared to known distributions')

print()
print("---   There are no symbolic variables   ---")