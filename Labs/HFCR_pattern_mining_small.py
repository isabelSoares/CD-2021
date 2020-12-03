import data_preparation_functions as prepfunctions
import mlxtend.frequent_patterns as pm
import matplotlib.pyplot as plt
import ds_functions as ds
import pandas as pd
import os

graphsDir = './Results/Pattern Mining/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

bin_strategies = ['Uniform', 'Quantile', 'Kmeans']
n_bins = [3, 5, 10]

values = {
    'with 3 bins': [18200, 20500, 20000],
    'with 5 bins': [18200, 12000, 18000],
    'with 10 bins': [12500, 5400, 10500],
}

plt.figure(figsize=(7,7))
ds.multiple_bar_chart(bin_strategies, values, ylabel='Number of Patterns')
plt.suptitle('Number of Patterns for a support of 0.01')
plt.savefig(graphsDir + 'HFCR Number of Patterns')

values = {
    'with 3 bins': [30, 59, 29],
    'with 5 bins': [45, 81, 40],
    'with 10 bins': [52, 50, 40],
}

plt.figure(figsize=(7,7))
ds.multiple_bar_chart(bin_strategies, values, ylabel='Lift Score')
plt.suptitle('Lift of the top 10% for a support of 0.01')
plt.savefig(graphsDir + 'HFCR Lift Score')

bin_strategies = ['0.25', '0.57']
