import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ts_functions as ts
import matrixprofile as mp

data = pd.read_csv('../Dataset/covid19_pt.csv')
#data = data.sort_values(by='Date')

graphsDir = './Results/Motif Discovery/Original/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('Covid19 - Original')

index_var = 'Date'
variable = 'deaths'
data[index_var] = pd.to_datetime(data[index_var])
data = data.set_index(index_var)
data = data.diff()
data['deaths'][0] = 0
data = data.sort_index()

FIG_WIDTH, FIG_HEIGHT = 3*ts.HEIGHT, ts.HEIGHT/2

plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
ts.plot_series(data, x_label='Date', y_label='deaths', title='COVID19 original')
plt.xticks(rotation = 45)
plt.suptitle('Covid19 - Original')
plt.savefig(graphsDir + 'Covid19 - Original')

graphsDir = './Results/Motif Discovery/Matrix Profile/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('Covid19 - Matrix Profile')

all_windows = [
    ('6 Days', 6),
    ('Week', 7),
    ('2 Weeks', 2 * 7),
    ('Month', 30),
    ('Quarter', 3 * 30),
]

def compute_matrix_profiles(df: pd.DataFrame, windows: list) :
    profiles = {}
    for label, size in windows:
        key = '{} Profile'.format(label)
        profiles[key] = mp.compute(df[variable].values, size)
    return profiles

def plot_signal_data(profiles: dict, windows: list):
    _, axes = plt.subplots(len(windows), 1, sharex=True, figsize=(FIG_WIDTH, len(windows)*FIG_HEIGHT))
    for ax_idx, window in enumerate(windows):
        key = '{} Profile'.format(window[0])
        axes[ax_idx].plot(profiles[key]['mp'])
        axes[ax_idx].set_title(key)

    plt.xlabel(index_var)
    plt.tight_layout()
    plt.suptitle('Covid19 - Profile')
    plt.savefig(graphsDir + 'Covid19 - Profile')

all_profiles = compute_matrix_profiles(data, all_windows)
plot_signal_data(all_profiles, all_windows)

def compute_all_profiles(profiles: dict, windows: list, k: int, type: str='motifs'):
    discover_function = mp.discover.motifs
    if type == 'discords':
        discover_function = mp.discover.discords

    for label, size in windows:
        key = '{} Profile'.format(label)
        profiles[key] = discover_function(profiles[key], k=k, exclusion_zone=size)


graphsDir = './Results/Motif Discovery/Motifs/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('Covid19 - Motifs')


compute_all_profiles(all_profiles, all_windows, k=10, type='motifs')

def show_profile(profile, title, type):
    lst_figs = mp.visualize(profile)
    for i in range(len(lst_figs)-1):
        plt.close(lst_figs[i])
    plt.suptitle('Covid19 - ' + type + ' - ' + title)
    plt.savefig(graphsDir + 'Covid19 - ' + type + ' - ' + title)

title = all_windows[0][0]+' Profile'
show_profile(all_profiles[title], title, 'Motifs')
#print(all_profiles[title])

title = all_windows[1][0]+' Profile'
show_profile(all_profiles[title], title, 'Motifs')

title = all_windows[2][0]+' Profile'
show_profile(all_profiles[title], title, 'Motifs')

title = all_windows[3][0]+' Profile'
show_profile(all_profiles[title], title, 'Motifs')

title = all_windows[4][0]+' Profile'
show_profile(all_profiles[title], title, 'Motifs')
"""
compute_all_profiles(all_profiles, all_windows, k=5, type='discords')

title = all_windows[0][0]+' Profile'
show_profile(all_profiles[title], title, 'Discords')

title = all_windows[3][0]+' Profile'
show_profile(all_profiles[title], title, 'Discords')
"""