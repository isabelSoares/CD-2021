import data_preparation_functions as prepfunctions
import mlxtend.frequent_patterns as pm
import matplotlib.pyplot as plt
import ds_functions as ds
import pandas as pd
import os

graphsDir = './Results/Pattern Mining/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

perm_data: pd.DataFrame = pd.read_csv('../Dataset/heart_failure_clinical_records_dataset.csv')
perm_data.pop("DEATH_EVENT")
boolean_attributes = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]

bin_strategies = ['uniform', 'quantile', 'kmeans']
n_bins = [3, 5, 10]

def plot_top_rules(rules: pd.DataFrame, metric: str, per_metric: str, save_dir: str) -> None:
    _, ax = plt.subplots(figsize=(21, 3))
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
    plt.savefig(save_dir + 'HFCR Pattern Mining - Metric ' + metric + ' - Per metric ' + per_metric + ' - Association Rules top rules')

def analyse_per_metric(rules: pd.DataFrame, metric: str, metric_values: list, save_dir: str) -> list:
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

    _, axs = plt.subplots(2, 2, figsize=(20, 10), squeeze=False)
    ds.multiple_line_chart(metric_values, conf, ax=axs[0, 0], title=f'Avg Confidence x {metric}',
                           xlabel=metric, ylabel='Avg confidence')
    ds.multiple_line_chart(metric_values, lift, ax=axs[0, 1], title=f'Avg Lift x {metric}',
                           xlabel=metric, ylabel='Avg lift')
    ds.multiple_line_chart(metric_values, leverage, ax=axs[1, 0], title=f'Avg Leverage x {metric}',
                           xlabel=metric, ylabel='Avg leverage')
    plt.savefig(save_dir + 'HFCR Pattern Mining - Association Rules analyse per ' + metric)

    plot_top_rules(top_conf, 'confidence', metric, subDir)
    plot_top_rules(top_lift, 'lift', metric, subDir)
    plot_top_rules(top_leverage, 'leverage', metric, subDir)

    return nr_rules

for strategie in bin_strategies:
    for bins in n_bins:
        print(strategie + " with " + str(bins) + " bins")
        subDir = './Results/Pattern Mining/' + strategie + '/' + str(bins) + '/'
        if not os.path.exists(subDir):
            os.makedirs(subDir)

        data = prepfunctions.dummification(perm_data.copy(deep=True), boolean_attributes, bins, strategie)

        MIN_SUP: float = 0.001
        var_min_sup =[0.2, 0.1] + [round(i*MIN_SUP, 2) for i  in range(100, 0, -10)]

        plt.figure()
        patterns: pd.DataFrame = pm.apriori(data, min_support=MIN_SUP, use_colnames=True, verbose=True)
        print(len(patterns),'patterns')
        nr_patterns = []
        for sup in var_min_sup:
            pat = patterns[patterns['support']>=sup]
            nr_patterns.append(len(pat))

        plt.figure(figsize=(6, 4))
        ds.plot_line(var_min_sup, nr_patterns, title='Nr Patterns x Support', xlabel='support', ylabel='Nr Patterns')
        plt.savefig(subDir + 'HFCR Pattern Mining - Nr Patterns x Support')

        MIN_CONF: float = 0.1
        rules = pm.association_rules(patterns, metric='confidence', min_threshold=MIN_CONF*5, support_only=False)
        print(f'\tfound {len(rules)} rules')

        nr_rules_sp = analyse_per_metric(rules, 'support', var_min_sup, subDir)
        plt.figure(figsize=(6, 4))
        ds.plot_line(var_min_sup, nr_rules_sp, title='Nr Rules x Support', xlabel='support', ylabel='Nr. rules', percentage=False)
        plt.savefig(subDir + 'HFCR Pattern Mining - Nr Rules x Support')

        var_min_conf = [round(i * MIN_CONF, 2) for i in range(10, 5, -1)]
        nr_rules_cf = analyse_per_metric(rules, 'confidence', var_min_conf, subDir)
        plt.figure(figsize=(6, 4))
        ds.plot_line(var_min_conf, nr_rules_cf, title='Nr Rules x Confidence', xlabel='confidence', ylabel='Nr Rules', percentage=False)
        plt.savefig(subDir + 'HFCR Pattern Mining - Nr Rules x Confidence')