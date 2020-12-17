import os
import pandas as pd
import matplotlib.pyplot as plt
import ts_functions as ts
import ds_functions as ds
import statsmodels.tsa.seasonal as seasonal
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from math import sqrt

data = pd.read_csv('../Dataset/deaths_pt.csv', index_col='start_date', sep=',', decimal='.',
                   parse_dates=True, infer_datetime_format=True)
data = data.sort_values(by='start_date')
data = data.resample('W').first()
data = data.fillna(method='bfill').fillna(method='ffill')

graphsDir = './Results/Forecasting/Original/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('Deaths - Original')

x_label='start_date'
y_label='deaths'
plt.figure(figsize=(3*ts.HEIGHT, ts.HEIGHT/2))
ts.plot_series(data, x_label=x_label, y_label=y_label, title='DEATHS original')
plt.xticks(rotation = 45)
plt.suptitle('Deaths - Original')
plt.savefig(graphsDir + 'Deaths - Original')


FIG_WIDTH, FIG_HEIGHT = 3*ts.HEIGHT, ts.HEIGHT/2

def plot_components(series: pd.Series, comps: seasonal.DecomposeResult, x_label: str = 'time', y_label:str =''):
    lst = [('Observed', series), ('trend', comps.trend), ('seasonal', comps.seasonal), ('residual', comps.resid)]
    _, axs = plt.subplots(len(lst), 1, figsize=(3*ts.HEIGHT, ts.HEIGHT*len(lst)))
    for i in range(len(lst)):
        axs[i].set_title(lst[i][0])
        axs[i].set_ylabel(y_label)
        axs[i].set_xlabel(x_label)
        axs[i].plot(lst[i][1])

decomposition = seasonal.seasonal_decompose(data, model = "add")
plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
plot_components(data, decomposition, x_label='start_date', y_label='deaths')
plt.savefig(graphsDir + 'Deaths - Observed vs Trend vs Seasonal vs Residual')

graphsDir = './Results/Forecasting/ARIMA/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('Deaths - ARIMA')

df = data
model = ARIMA(df, order=(2,0,2))
results = model.fit()
plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
plt.plot(df)
plt.plot(results.fittedvalues)
plt.savefig(graphsDir + 'Deaths - ARIMA 1')


results.plot_diagnostics(figsize=(FIG_WIDTH, 4*FIG_HEIGHT))
plt.savefig(graphsDir + 'Deaths - Diagnostics')



_, axs = plt.subplots(2, 3, figsize=(FIG_WIDTH, 2*FIG_HEIGHT))
params = (1, 2, 3)
for d in (0, 1):
    rmse = {}
    mse = {}
    mae = {}
    for p in params:
        rmse_lst = []
        mse_lst = []
        mae_lst = []
        for q in params:
            mod = ARIMA(df, order=(p, d, q))
            results = mod.fit()
            rmse_lst.append(sqrt(results.mse))
            mse_lst.append(results.mse)
            mae_lst.append(results.mae)
        rmse[p] = rmse_lst
        mse[p] = mse_lst
        mae[p] = mae_lst
    ds.multiple_line_chart(params, rmse, ax=axs[d, 0], title=f'RMSE with d={d}', xlabel='p', ylabel='rmse')
    ds.multiple_line_chart(params, mse, ax=axs[d, 1], title=f'MSE with d={d}', xlabel='p', ylabel='mse')
    ds.multiple_line_chart(params, mae, ax=axs[d, 2], title=f'MAE with d={d}', xlabel='p', ylabel='mae')
plt.savefig(graphsDir + 'Deaths - RMSE and MSE and MAE')




def plot_forecasting(train: pd.Series, test: pd.Series, pred,
                     ax: plt.Axes=None, x_label: str = 'time', y_label:str =''):
    if ax is None:
        ax = plt.gca()
    ax.plot(train, label='train')
    ax.plot(test, label='test')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.plot(pred.index, pred.values, label='predicted', color='r')
    plt.legend()

p, d, q = 2, 1, 2
n = len(df)
train = df[:n*9//10]
test = df[n*9//10+1:]

mod = ARIMA(train, order=(p, d, q))
mod = mod.fit()
pred = mod.predict(start = len(train), end = len(df)-1)

plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
plot_forecasting(train, test, pred, x_label=x_label, y_label=y_label)
plt.savefig(graphsDir + 'Deaths - Train vs Test vs Predicted')





fig, axs = plt.subplots(3, 1, figsize=(FIG_WIDTH, 3*FIG_HEIGHT))
fig.suptitle(f'ARIMA predictions (p={p},d={d},q={q})')


# PREDICT 2019
print(df)
train = df.loc[(df.index >= '2015-01-01') & (df.index < '2019-01-01')]
test = df.loc[(df.index >= '2019-01-01') & (df.index < '2020-01-01')]
mod = ARIMA(train, order=(p, d, q))
mod = mod.fit()
pred = mod.predict(start = '2019-01-01', end = '2020-01-01')
plot_forecasting(train, test, pred, ax=axs[0], x_label=x_label, y_label=y_label)

# PREDICT 2020
train = data.loc[(df.index >= '2015-01-01') & (df.index < '2020-01-01')]
test = data.loc[(df.index >= '2020-01-01') & (df.index < '2020-12-01')]
mod = ARIMA(train, order=(p, d, q))
mod = mod.fit()
pred = mod.predict(start = '2020-01-01', end = '2020-12-01')
plot_forecasting(train, test, pred, ax=axs[1], x_label=x_label, y_label=y_label)


# PREDICT LAST WEEKS

'''
mask_train = (df.index >= '2015-01-01') & (df.index < '2020-12-01')
train = data.loc[mask_train]
mod = ARIMA(train, order=(p, d, q))
mod = mod.fit()
'''
pred = mod.predict(start = '2020-12-01', end = '2020-12-27')
plot_forecasting(train, [], pred, ax=axs[2], x_label=x_label, y_label=y_label)


plt.savefig(graphsDir + 'Deaths - ARIMA 2')

'''
2020-09-06 00:00:00
2020-09-13 00:00:00
2020-09-20 00:00:00
2020-09-27 00:00:00
2020-10-04 00:00:00
2020-10-11 00:00:00
2020-10-18 00:00:00
2020-10-25 00:00:00
2020-11-01 00:00:00
2020-11-08 00:00:00
2020-11-15 00:00:00
2020-11-22 00:00:00
2020-11-29 00:00:00
2020-12-06 00:00:00
2020-12-13 00:00:00
2020-12-20 00:00:00
2020-12-27 00:00:00
'''