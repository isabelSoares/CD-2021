import os
import pandas as pd
import matplotlib.pyplot as plt
import ts_functions as ts


graphsDir = './Results/Covid19_Diff/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

fig, axs = plt.subplots(2, 1, figsize=(10, 2*4))

data = pd.read_csv('../Dataset/covid19_pt.csv', index_col='Date', sep=',', decimal='.',
                   parse_dates=True, infer_datetime_format=True)
data = data.sort_values(by='Date')

ts.plot_series(data, ax=axs[0], x_label='Date', y_label='deaths', title='Original (covid19)')

print('Covid19 - Differentiation')

data = pd.read_csv('../Dataset/covid19_pt.csv', index_col='Date', sep=',', decimal='.',
                   parse_dates=True, infer_datetime_format=True)
data = data.diff()
data['deaths'][0] = 0
data = data.sort_values(by='Date')

ts.plot_series(data, ax=axs[1], x_label='Date', y_label='deaths', title='Diff (covid19)')
plt.suptitle('Covid19 - Diff')
plt.savefig(graphsDir + 'Covid19 - Diff')