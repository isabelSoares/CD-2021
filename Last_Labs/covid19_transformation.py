import os
import pandas as pd
import matplotlib.pyplot as plt
import ts_functions as ts

data = pd.read_csv('../Dataset/covid19_pt.csv', index_col='Date', sep=',', decimal='.',
                   parse_dates=True, infer_datetime_format=True)
data = data.sort_values(by='Date') 

graphsDir = './Results/Transformation/Smoothing/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('Covid19 - Smoothing')

plt.figure(figsize=(3*ts.HEIGHT, ts.HEIGHT/2))
ts.plot_series(data, x_label='Date', y_label='deaths', title='COVID19 original')
plt.xticks(rotation = 45)
plt.suptitle('Covid19 - win_size = 10')
plt.savefig(graphsDir + 'Covid19 - win_size = 10')

WIN_SIZE = 100
rolling = data.rolling(window=WIN_SIZE)
smooth_df = rolling.mean()
plt.figure(figsize=(3*ts.HEIGHT, ts.HEIGHT/2))
ts.plot_series(smooth_df, title=f'Smoothing (win_size={WIN_SIZE})', x_label='Date', y_label='deaths')
plt.xticks(rotation = 45)
plt.suptitle('Covid19 - win_size = 100')
plt.savefig(graphsDir + 'Covid19 - win_size = 100')


graphsDir = './Results/Transformation/Aggregation/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('Covid19 - Aggregation')

def aggregate_by(data: pd.Series, index_var: str, period: str, title: str = '', x_label: str = '', y_label: str = ''):
    index = data.index.to_period(period)
    agg_df = data.copy().groupby(index).mean()
    agg_df[index_var] = index.drop_duplicates().to_timestamp()
    agg_df.set_index(index_var, drop=True, inplace=True)
    ts.plot_series(agg_df, title=title, x_label=x_label, y_label=y_label)

plt.figure(figsize=(3*ts.HEIGHT, ts.HEIGHT))
aggregate_by(data, 'Date', 'D', title='Daily deaths', x_label='Date', y_label='deaths')
plt.xticks(rotation = 45)
plt.suptitle('Covid19 - Daily')
plt.savefig(graphsDir + 'Covid19 - Daily')

plt.figure(figsize=(3*ts.HEIGHT, ts.HEIGHT))
aggregate_by(data, 'Date', 'W', title='Weekly deaths', x_label='Date', y_label='deaths')
plt.xticks(rotation = 45)
plt.suptitle('Covid19 - Weekly')
plt.savefig(graphsDir + 'Covid19 - Weekly')

plt.figure(figsize=(3*ts.HEIGHT, ts.HEIGHT))
aggregate_by(data, 'Date', 'M', title='Monthly deaths', x_label='Date', y_label='deaths')
plt.xticks(rotation = 45)
plt.suptitle('Covid19 - Monthly')
plt.savefig(graphsDir + 'Covid19 - Monthly')


graphsDir = './Results/Transformation/Differentiation/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('Covid19 - Differentiation')

diff_df = data.diff()
plt.figure(figsize=(3*ts.HEIGHT, ts.HEIGHT))
ts.plot_series(diff_df, title='Differentiation', x_label='Date', y_label='deaths')
plt.xticks(rotation = 45)
plt.suptitle('Covid19 - Differentiation')
plt.savefig(graphsDir + 'Covid19 - Differentiation')

print('Covid19 - Change of Space - TODO')
