import os
import pandas as pd
import matplotlib.pyplot as plt
import ts_functions as ts

data = pd.read_csv('../Dataset/deaths_pt.csv', index_col='start_date', sep=',', decimal='.',
                   parse_dates=True, infer_datetime_format=True)

graphsDir = './Results/Transformation/Smoothing/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('Deaths - Smoothing')

plt.figure(figsize=(3*ts.HEIGHT, ts.HEIGHT/2))
ts.plot_series(data, x_label='timestamp', y_label='consumption', title='DEATHS original')
plt.xticks(rotation = 45)
plt.suptitle('Deaths - win_size = 10')
plt.savefig(graphsDir + 'Deaths - win_size = 10')

WIN_SIZE = 100
rolling = data.rolling(window=WIN_SIZE)
smooth_df = rolling.mean()
plt.figure(figsize=(3*ts.HEIGHT, ts.HEIGHT/2))
ts.plot_series(smooth_df, title=f'Smoothing (win_size={WIN_SIZE})', x_label='timestamp', y_label='consumption')
plt.xticks(rotation = 45)
plt.suptitle('Deaths - win_size = 100')
plt.savefig(graphsDir + 'Deaths - win_size = 100')


graphsDir = './Results/Transformation/Aggregation/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('Deaths - Aggregation')

def aggregate_by(data: pd.Series, index_var: str, period: str, title: str = '', x_label: str = '', y_label: str = ''):
    index = data.index.to_period(period)
    agg_df = data.copy().groupby(index).mean()
    agg_df[index_var] = index.drop_duplicates().to_timestamp()
    agg_df.set_index(index_var, drop=True, inplace=True)
    ts.plot_series(agg_df, title=title, x_label=x_label, y_label=y_label)

plt.figure(figsize=(3*ts.HEIGHT, ts.HEIGHT))
aggregate_by(data, 'timestamp', 'D', title='Daily consumptions', x_label='timestamp', y_label='consumption')
plt.xticks(rotation = 45)
plt.suptitle('Deaths - Daily')
plt.savefig(graphsDir + 'Deaths - Daily')

plt.figure(figsize=(3*ts.HEIGHT, ts.HEIGHT))
aggregate_by(data, 'timestamp', 'W', title='Weekly consumptions', x_label='timestamp', y_label='consumption')
plt.xticks(rotation = 45)
plt.suptitle('Deaths - Weekly')
plt.savefig(graphsDir + 'Deaths - Weekly')

plt.figure(figsize=(3*ts.HEIGHT, ts.HEIGHT))
aggregate_by(data, 'timestamp', 'M', title='Monthly consumptions', x_label='timestamp', y_label='consumption')
plt.xticks(rotation = 45)
plt.suptitle('Deaths - Monthly')
plt.savefig(graphsDir + 'Deaths - Monthly')


graphsDir = './Results/Transformation/Differentiation/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('Deaths - Differentiation')

diff_df = data.diff()
plt.figure(figsize=(3*ts.HEIGHT, ts.HEIGHT))
ts.plot_series(diff_df, title='Differentiation', x_label='timestamp', y_label='consumption')
plt.xticks(rotation = 45)
plt.suptitle('Deaths - Differentiation')
plt.savefig(graphsDir + 'Deaths - Differentiation')

print('Deaths - Change of Space - TODO')
