import os
import pandas as pd
import matplotlib.pyplot as plt
import ts_functions as ts
import numpy as np

data = pd.read_csv('../Dataset/deaths_pt.csv', index_col='start_date', sep=',', decimal='.',
                   parse_dates=True, infer_datetime_format=True)

graphsDir = './Results/Profiling/Dimensionality/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('Deaths - Dimensionality')

print(" - Nr. Records = ", data.shape[0])
print(" - First timestamp", data.index[0])
print(" - Last timestamp", data.index[-1])
plt.figure(figsize=(3*ts.HEIGHT, ts.HEIGHT))
ts.plot_series(data, x_label='timestamp', y_label='consumption', title='DEATHS')
plt.xticks(rotation = 45)
plt.suptitle('Deaths - Dimensionality')
plt.savefig(graphsDir + 'Deaths - Dimensionality')


graphsDir = './Results/Profiling/Granularity/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('Deaths - Granularity')

day_df = data.copy().groupby(data.index.date).mean()
plt.figure(figsize=(3*ts.HEIGHT, ts.HEIGHT))
ts.plot_series(day_df, title='Daily consumptions', x_label='timestamp', y_label='consumption')
plt.xticks(rotation = 45)
plt.suptitle('Deaths - Daily')
plt.savefig(graphsDir + 'Deaths - Daily')

index = data.index.to_period('W')
week_df = data.copy().groupby(index).mean()
week_df['timestamp'] = index.drop_duplicates().to_timestamp()
week_df.set_index('timestamp', drop=True, inplace=True)
plt.figure(figsize=(3*ts.HEIGHT, ts.HEIGHT))
ts.plot_series(week_df, title='Weekly consumptions', x_label='timestamp', y_label='consumption')
plt.xticks(rotation = 45)
plt.suptitle('Deaths - Weekly')
plt.savefig(graphsDir + 'Deaths - Weekly')

index = data.index.to_period('M')
month_df = data.copy().groupby(index).mean()
month_df['timestamp'] = index.drop_duplicates().to_timestamp()
month_df.set_index('timestamp', drop=True, inplace=True)
plt.figure(figsize=(3*ts.HEIGHT, ts.HEIGHT))
ts.plot_series(month_df, title='Monthly consumptions', x_label='timestamp', y_label='consumption')
plt.suptitle('Deaths - Monthly')
plt.savefig(graphsDir + 'Deaths - Monthly')

index = data.index.to_period('Q')
quarter_df = data.copy().groupby(index).mean()
quarter_df['timestamp'] = index.drop_duplicates().to_timestamp()
quarter_df.set_index('timestamp', drop=True, inplace=True)
plt.figure(figsize=(3*ts.HEIGHT, ts.HEIGHT))
ts.plot_series(quarter_df, title='Quarterly consumptions', x_label='timestamp', y_label='consumption')
plt.suptitle('Deaths - Quarterly')
plt.savefig(graphsDir + 'Deaths - Quarterly')


graphsDir = './Results/Profiling/Distribution/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('Deaths - Distribution')

index = data.index.to_period('W')
week_df = data.copy().groupby(index).sum()
week_df['timestamp'] = index.drop_duplicates().to_timestamp()
week_df.set_index('timestamp', drop=True, inplace=True)
_, axs = plt.subplots(1, 2, figsize=(2*ts.HEIGHT, ts.HEIGHT/2))
axs[0].grid(False)
axs[0].set_axis_off()
axs[0].set_title('HOURLY', fontweight="bold")
axs[0].text(0, 0, str(data.describe()))
axs[1].grid(False)
axs[1].set_axis_off()
axs[1].set_title('WEEKLY', fontweight="bold")
axs[1].text(0, 0, str(week_df.describe()))

_, axs = plt.subplots(1, 2, figsize=(2*ts.HEIGHT, ts.HEIGHT))
data.boxplot(ax=axs[0])
week_df.boxplot(ax=axs[1])
plt.suptitle('Deaths - 5-Number Summary')
plt.savefig(graphsDir + 'Deaths - 5-Number Summary')

bins = (10, 25, 50)
_, axs = plt.subplots(1, len(bins), figsize=(len(bins)*ts.HEIGHT, ts.HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for hourly deaths %d bins'%bins[j])
    axs[j].set_xlabel('consumption')
    axs[j].set_ylabel('Nr records')
    axs[j].hist(data.values, bins=bins[j])
plt.suptitle('Deaths - (Hourly) Variables Distribution')
plt.savefig(graphsDir + 'Deaths - (Hourly) Variables Distribution')

_, axs = plt.subplots(1, len(bins), figsize=(len(bins)*ts.HEIGHT, ts.HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for weekly deaths %d bins'%bins[j])
    axs[j].set_xlabel('consumption')
    axs[j].set_ylabel('Nr records')
    axs[j].hist(week_df.values, bins=bins[j])
plt.suptitle('Deaths - (Weekly) Variables Distribution')
plt.savefig(graphsDir + 'Deaths - (Weekly) Variables Distribution')


graphsDir = './Results/Profiling/Stationarity/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('Deaths - Stationarity')

dt_series = pd.Series(data['deaths'])

mean_line = pd.Series(np.ones(len(dt_series.values)) * dt_series.mean(), index=dt_series.index)
series = {'Deaths': dt_series, 'mean': mean_line}
plt.figure(figsize=(3*ts.HEIGHT, ts.HEIGHT))
ts.plot_series(series, x_label='timestamp', y_label='consumption', title='Stationary study', show_std=True)
plt.suptitle('Deaths - Stationarity 1')
plt.savefig(graphsDir + 'Deaths - Stationarity 1')

BINS = 10
line = []
n = len(dt_series)
for i in range(BINS):
    b = dt_series[i*n//BINS:(i+1)*n//BINS]
    mean = [b.mean()] * (n//BINS)
    line += mean
line += [line[-1]] * (n - len(line))
mean_line = pd.Series(line, index=dt_series.index)
series = {'deaths': dt_series, 'mean': mean_line}
plt.figure(figsize=(3*ts.HEIGHT, ts.HEIGHT))
ts.plot_series(series, x_label='time', y_label='consumptions', title='Stationary study', show_std=True)
plt.suptitle('Deaths - Stationarity 2')
plt.savefig(graphsDir + 'Deaths - Stationarity 2')