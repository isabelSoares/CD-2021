import pandas as pd
import matplotlib.pyplot as plt
import ds_functions as ds
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
graphsDir = './Results/'
print('------------------------')
print('-                      -')
print('-   QOT Distribution   -')
print('-                      -')
print('------------------------')
print()


print('QOT Distribution - Numeric variables description')

data = pd.read_csv('../../Dataset/qsar_oral_toxicity.csv', sep=';', header=None)

print(data.describe())
data.describe().to_csv(graphsDir + 'QOT Distribution - Numeric variables description.csv')
print()


print('QOT Distribution - Boxplot')
data.boxplot(rot=45, figsize=(150,3))
plt.suptitle('QOT Distribution - Boxplot')
plt.savefig(graphsDir + 'QOT Distribution - Boxplot')
print()

"""

numeric_vars = data.select_dtypes(include='number').columns
rows, cols = ds.choose_grid(len(numeric_vars)//10, 18)
height_fix = ds.HEIGHT/1.7

print('QOT Distribution - Boxplot for each variable')
fig, axs = plt.subplots(rows, cols, figsize=(cols*height_fix, rows*height_fix))
i, j = 0, 0
for n in range(len(numeric_vars)):
    axs[i, j].set_title('Boxplot for %s'%numeric_vars[n])
    axs[i, j].boxplot(data[numeric_vars[n]].dropna().values)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
plt.suptitle('QOT Distribution - Boxplot for each variable')
plt.savefig(graphsDir + 'QOT Distribution - Boxplot for each variable')
print()

print('QOT Distribution - Histograms')
fig, axs = plt.subplots(rows, cols, figsize=(cols*height_fix, rows*height_fix))
i, j = 0, 0
for n in range(len(numeric_vars)):
    axs[i, j].set_title('Histogram for %s'%numeric_vars[n])
    axs[i, j].set_xlabel(numeric_vars[n])
    axs[i, j].set_ylabel("nr records")
    axs[i, j].hist(data[numeric_vars[n]].dropna().values, 'auto')
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
plt.suptitle('QOT Distribution - Histograms')
plt.savefig(graphsDir + 'QOT Distribution - Histograms')
print()


print('QOT Distribution - Histograms with the best fit')
import seaborn as sns
fig, axs = plt.subplots(rows, cols, figsize=(cols*height_fix, rows*height_fix))
i, j = 0, 0
for n in range(len(numeric_vars)):
    axs[i, j].set_title('Histogram with trend for %s'%numeric_vars[n])
    sns.distplot(data[numeric_vars[n]].dropna().values, norm_hist=True, ax=axs[i, j], axlabel=numeric_vars[n])
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
plt.suptitle('QOT Distribution - Histograms with the best fit')
plt.savefig(graphsDir + 'QOT Distribution - Histograms with the best fit')
print()


print('QOT Distribution - Histograms compared to known distributions')
import scipy.stats as _stats
import numpy as np
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

fig, axs = plt.subplots(rows, cols, figsize=(cols*height_fix, rows*height_fix))
i, j = 0, 0
for n in range(len(numeric_vars)//10):
    histogram_with_distributions(axs[i, j], data[numeric_vars[n]].dropna(), numeric_vars[n])
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
plt.suptitle('QOT Distribution - Histograms compared to known distributions')
plt.savefig(graphsDir + 'QOT Distribution - Histograms compared to known distributions')
print()

"""

print('QOT Distribution - Object variables description')
print(data.describe(include='object'))
data.describe(include='object').to_csv(graphsDir + 'QOT Distribution - Object variables description.csv')
print()

print('Object variables values description')
symbolic_vars = data.select_dtypes(include='object').columns
for v in symbolic_vars:
    print(v, data[v].unique())
print()



print('QOT Distribution (Object) - Histogram')
rows, cols = ds.choose_grid(len(symbolic_vars))
fig, axs = plt.subplots(rows, cols, figsize=(cols*ds.HEIGHT, rows*ds.HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(symbolic_vars)):
    counts = data[symbolic_vars[n]].value_counts()
    ds.bar_chart(counts.index.to_list(), counts.values, ax=axs[i, j], title='Histogram for %s'%symbolic_vars[n],
                 xlabel=symbolic_vars[n], ylabel='nr records')
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
plt.suptitle('QOT Distribution (Object) - Histogram')
plt.savefig(graphsDir + 'QOT Distribution (Object) - Histogram')



