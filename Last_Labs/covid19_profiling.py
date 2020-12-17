import os
import pandas as pd
import matplotlib.pyplot as plt
import ts_functions as ts
import numpy as np

data = pd.read_csv('../Dataset/covid19_pt.csv', index_col='Date', sep=',', decimal='.',
                   parse_dates=True, infer_datetime_format=True)
data = data.diff()
data['deaths'][0] = 0
data = data.sort_values(by='Date')

graphsDir = './Results/Profiling/Dimensionality/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('Covid19 - Dimensionality')

print(" - Nr. Records = ", data.shape[0])
print(" - First Date", data.index[0])
print(" - Last Date", data.index[-1])
plt.figure(figsize=(3*ts.HEIGHT, ts.HEIGHT))
ts.plot_series(data, x_label='Date', y_label='deaths', title='COVID19')
plt.xticks(rotation = 45)
plt.suptitle('Covid19 - Dimensionality')
plt.savefig(graphsDir + 'Covid19 - Dimensionality')


graphsDir = './Results/Profiling/Granularity/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('Covid19 - Granularity')

day_df = data.copy().groupby(data.index.date).mean()
plt.figure(figsize=(3*ts.HEIGHT, ts.HEIGHT))
ts.plot_series(day_df, title='Daily deaths', x_label='Date', y_label='deaths')
plt.xticks(rotation = 45)
plt.suptitle('Covid19 - Daily')
plt.savefig(graphsDir + 'Covid19 - Daily')

index = data.index.to_period('W')
week_df = data.copy().groupby(index).mean()
week_df['Date'] = index.drop_duplicates().to_timestamp()
week_df.set_index('Date', drop=True, inplace=True)
plt.figure(figsize=(3*ts.HEIGHT, ts.HEIGHT))
ts.plot_series(week_df, title='Weekly deaths', x_label='Date', y_label='deaths')
plt.xticks(rotation = 45)
plt.suptitle('Covid19 - Weekly')
plt.savefig(graphsDir + 'Covid19 - Weekly')

index = data.index.to_period('M')
month_df = data.copy().groupby(index).mean()
month_df['Date'] = index.drop_duplicates().to_timestamp()
month_df.set_index('Date', drop=True, inplace=True)
plt.figure(figsize=(3*ts.HEIGHT, ts.HEIGHT))
ts.plot_series(month_df, title='Monthly deaths', x_label='Date', y_label='deaths')
plt.suptitle('Covid19 - Monthly')
plt.savefig(graphsDir + 'Covid19 - Monthly')

index = data.index.to_period('Q')
quarter_df = data.copy().groupby(index).mean()
quarter_df['Date'] = index.drop_duplicates().to_timestamp()
quarter_df.set_index('Date', drop=True, inplace=True)
plt.figure(figsize=(3*ts.HEIGHT, ts.HEIGHT))
ts.plot_series(quarter_df, title='Quarterly deaths', x_label='Date', y_label='deaths')
plt.suptitle('Covid19 - Quarterly')
plt.savefig(graphsDir + 'Covid19 - Quarterly')


graphsDir = './Results/Profiling/Distribution/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('Covid19 - Distribution')

index = data.index.to_period('W')
week_df = data.copy().groupby(index).sum()
week_df['Date'] = index.drop_duplicates().to_timestamp()
week_df.set_index('Date', drop=True, inplace=True)
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
plt.suptitle('Covid19 - 5-Number Summary')
plt.savefig(graphsDir + 'Covid19 - 5-Number Summary')

bins = (10, 25, 50)
_, axs = plt.subplots(1, len(bins), figsize=(len(bins)*ts.HEIGHT, ts.HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for hourly deaths %d bins'%bins[j])
    axs[j].set_xlabel('deaths')
    axs[j].set_ylabel('Nr records')
    axs[j].hist(data.values, bins=bins[j])
plt.suptitle('Covid19 - (Hourly) Variables Distribution')
plt.savefig(graphsDir + 'Covid19 - (Hourly) Variables Distribution')

_, axs = plt.subplots(1, len(bins), figsize=(len(bins)*ts.HEIGHT, ts.HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for weekly deaths %d bins'%bins[j])
    axs[j].set_xlabel('deaths')
    axs[j].set_ylabel('Nr records')
    axs[j].hist(week_df.values, bins=bins[j])
plt.suptitle('Covid19 - (Weekly) Variables Distribution')
plt.savefig(graphsDir + 'Covid19 - (Weekly) Variables Distribution')


graphsDir = './Results/Profiling/Stationarity/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('Covid19 - Stationarity')

dt_series = pd.Series(data['deaths'])

mean_line = pd.Series(np.ones(len(dt_series.values)) * dt_series.mean(), index=dt_series.index)
series = {'covid19': dt_series, 'mean': mean_line}
plt.figure(figsize=(3*ts.HEIGHT, ts.HEIGHT))
ts.plot_series(series, x_label='Date', y_label='deaths', title='Stationary study', show_std=True)
plt.suptitle('Covid19 - Stationarity 1')
plt.savefig(graphsDir + 'Covid19 - Stationarity 1')

BINS = 10
line = []
n = len(dt_series)
for i in range(BINS):
    b = dt_series[i*n//BINS:(i+1)*n//BINS]
    mean = [b.mean()] * (n//BINS)
    line += mean
line += [line[-1]] * (n - len(line))
mean_line = pd.Series(line, index=dt_series.index)
series = {'covid19': dt_series, 'mean': mean_line}
plt.figure(figsize=(3*ts.HEIGHT, ts.HEIGHT))
ts.plot_series(series, x_label='time', y_label='deaths', title='Stationary study', show_std=True)
plt.suptitle('Covid19 - Stationarity 2')
plt.savefig(graphsDir + 'Covid19 - Stationarity 2')