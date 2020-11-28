import data_preparation_functions as prepfunctions
import mlxtend.frequent_patterns as pm
import matplotlib.pyplot as plt
import ds_functions as ds
import pandas as pd
import os

graphsDir = './Results/Pattern Mining/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

data: pd.DataFrame = pd.read_csv('../Dataset/qsar_oral_toxicity.csv', sep=';', header=None)
data.pop(1024)
data = prepfunctions.dummification(data, data.columns)

MIN_SUP: float = 0.25
var_min_sup =[0.70]
# IF WITH MIN_SUP = 0.57
'''
for i in range(70, -10, -10):
    var_min_sup.append(round(MIN_SUP + i / 1000, 2))
'''
# IF WITH MIN_SUP = 0.25
step = 350
while(round(MIN_SUP + step / 1000, 2) > MIN_SUP):
    if (not (round(MIN_SUP + step / 1000, 2) in var_min_sup)):
        var_min_sup.append(round(MIN_SUP + step / 1000, 2))
    step = step * 0.80
var_min_sup.append(MIN_SUP)

print(var_min_sup)
graphsDir = graphsDir + str(round(MIN_SUP, 2)) + '/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

plt.figure()
patterns: pd.DataFrame = pm.apriori(data, min_support=MIN_SUP, use_colnames=True, verbose=True, max_len=3)
print(len(patterns),'patterns')
nr_patterns = []
for sup in var_min_sup:
    pat = patterns[patterns['support']>=sup]
    nr_patterns.append(len(pat))

plt.figure(figsize=(6, 4))
ds.plot_line(var_min_sup, nr_patterns, title='Nr Patterns x Support', xlabel='support', ylabel='Nr Patterns')
plt.savefig(graphsDir + 'QOT Pattern Mining - Nr Patterns x Support')

MIN_CONF: float = 0.1
rules = pm.association_rules(patterns, metric='confidence', min_threshold=MIN_CONF*5, support_only=False)
print(f'\tfound {len(rules)} rules')

def plot_top_rules(rules: pd.DataFrame, metric: str, per_metric: str) -> None:
    _, ax = plt.subplots(figsize=(10, 3))
    ax.grid(False)
    ax.set_axis_off()
    ax.set_title(f'TOP 10 per Min {per_metric} - {metric}', fontweight="bold")
    text = ''
    cols = ['antecedents', 'consequents']
    rules[cols] = rules[cols].applymap(lambda x: tuple(x))
    for i in range(len(rules)):
        rule = rules.iloc[i]
        text += f"{rule['antecedents']} ==> {rule['consequents']}"
        text += f"(s: {rule['support']:.2f}, c: {rule['confidence']:.2f}, lift: {rule['lift']:.2f})\n"
    ax.text(0, 0, text)
    plt.savefig(graphsDir + 'QOT Pattern Mining - Metric ' + metric + ' - Per metric ' + per_metric + ' - Association Rules top rules')

def analyse_per_metric(rules: pd.DataFrame, metric: str, metric_values: list) -> list:
    print(f'Analyse per {metric}...')
    conf = {'avg': [], 'top25%': [], 'top10': []}
    lift = {'avg': [], 'top25%': [], 'top10': []}
    leverage = {'avg': [], 'top25%': [], 'top10': []}
    top_conf = []
    top_lift = []
    top_leverage = []
    nr_rules = []
    for m in metric_values:
        rs = rules[rules[metric] >= m]
        nr_rules.append(len(rs))
        conf['avg'].append(rs['confidence'].mean(axis=0))
        lift['avg'].append(rs['lift'].mean(axis=0))
        leverage['avg'].append(rs['leverage'].mean(axis=0))

        top_conf = rs.nlargest(int(0.25*len(rs)), 'confidence')
        conf['top25%'].append(top_conf['confidence'].mean(axis=0))
        top_lift = rs.nlargest(int(0.25*len(rs)), 'lift')
        lift['top25%'].append(top_lift['lift'].mean(axis=0))
        top_leverage = rs.nlargest(int(0.25*len(rs)), 'leverage')
        leverage['top25%'].append(top_leverage['leverage'].mean(axis=0))

        top_conf = rs.nlargest(10, 'confidence')
        conf['top10'].append(top_conf['confidence'].mean(axis=0))
        top_lift = rs.nlargest(10, 'lift')
        lift['top10'].append(top_lift['lift'].mean(axis=0))
        top_leverage = rs.nlargest(10, 'leverage')
        leverage['top10'].append(top_leverage['leverage'].mean(axis=0))

    _, axs = plt.subplots(2, 2, figsize=(10, 5), squeeze=False)
    ds.multiple_line_chart(metric_values, conf, ax=axs[0, 0], title=f'Avg Confidence x {metric}',
                           xlabel=metric, ylabel='Avg confidence')
    ds.multiple_line_chart(metric_values, lift, ax=axs[0, 1], title=f'Avg Lift x {metric}',
                           xlabel=metric, ylabel='Avg lift')
    ds.multiple_line_chart(metric_values, leverage, ax=axs[1, 0], title=f'Avg Leverage x {metric}',
                           xlabel=metric, ylabel='Avg leverage')
    plt.savefig(graphsDir + 'QOT Pattern Mining - Association Rules analyse per ' + metric)

    plot_top_rules(top_conf, 'confidence', metric)
    plot_top_rules(top_lift, 'lift', metric)
    plot_top_rules(top_leverage, 'leverage', metric)

    return nr_rules

nr_rules_sp = analyse_per_metric(rules, 'support', var_min_sup)
plt.figure(figsize=(6, 4))
ds.plot_line(var_min_sup, nr_rules_sp, title='Nr Rules x Support', xlabel='support', ylabel='Nr. rules', percentage=False)
plt.savefig(graphsDir + 'QOT Pattern Mining - Nr Rules x Support')

var_min_conf = [round(i * MIN_CONF, 2) for i in range(10, 5, -1)]
nr_rules_cf = analyse_per_metric(rules, 'confidence', var_min_conf)
plt.figure(figsize=(6, 4))
ds.plot_line(var_min_conf, nr_rules_cf, title='Nr Rules x Confidence', xlabel='confidence', ylabel='Nr Rules', percentage=False)    
plt.savefig(graphsDir + 'QOT Pattern Mining - Nr Rules x Confidence')