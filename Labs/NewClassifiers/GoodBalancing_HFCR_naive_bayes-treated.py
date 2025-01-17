import os
import numpy as np
import pandas as pd
import ds_functions as ds
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import data_preparation_functions as prepfunctions
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
import matplotlib.colors as clrs

from cycler import cycler

graphsDir = './Results/Naive Bayes/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('--------------------------------------')
print('-                                    -')
print('-     HFCR Naive Bayes - Treated     -')
print('-                                    -')
print('--------------------------------------')

data: pd.DataFrame = pd.read_csv('../../Dataset/heart_failure_clinical_records_dataset.csv')

data_outliers = prepfunctions.new_prepare_dataset(data.copy(), 'DEATH_EVENT', False, True)

all_datas = [data, data_outliers]
all_datas_index = [(0, 2), (1, 3)]
all_datas_splits = [{}, {}, {}, {}]
datas_splits_scaling_featureselection = []
c = 0
for dt in all_datas:
    y: np.ndarray = dt.copy().pop('DEATH_EVENT').values
    X: np.ndarray = dt.copy().values
    labels = [0,1]

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    trn_x_b_lst = {'Original':[], 'UnderSample':[], 'OverSample':[], 'SMOTE':[]}
    trn_y_b_lst = {'Original':[], 'UnderSample':[], 'OverSample':[], 'SMOTE':[]}
    tst_x_b_lst = {'Original':[], 'UnderSample':[], 'OverSample':[], 'SMOTE':[]}
    tst_y_b_lst = {'Original':[], 'UnderSample':[], 'OverSample':[], 'SMOTE':[]}
    trn_x_fs_lst = {'Original':[], 'UnderSample':[], 'OverSample':[], 'SMOTE':[]}
    trn_y_fs_lst = {'Original':[], 'UnderSample':[], 'OverSample':[], 'SMOTE':[]}
    tst_x_fs_lst = {'Original':[], 'UnderSample':[], 'OverSample':[], 'SMOTE':[]}
    tst_y_fs_lst = {'Original':[], 'UnderSample':[], 'OverSample':[], 'SMOTE':[]}
    for train_i, test_i in skf.split(X, y):
        trn_data = prepfunctions.data_balancing(dt.iloc[train_i].copy(), 'DEATH_EVENT')
        tst_data = {'Original':dt.iloc[test_i].copy(), 'UnderSample':dt.iloc[test_i].copy(), 'OverSample':dt.iloc[test_i].copy(), 'SMOTE':dt.iloc[test_i].copy()}

        trn_data_fs = prepfunctions.mask_feature_selection(trn_data.copy(), 'DEATH_EVENT', False, './Results/FeatureSelection/HFCR Feature Selection - Features')
        tst_data_fs = prepfunctions.mask_feature_selection(tst_data.copy(), 'DEATH_EVENT', False, './Results/FeatureSelection/HFCR Feature Selection - Features')

        for category in ['Original', 'UnderSample', 'OverSample', 'SMOTE']:
            # Train
            trn_y_b: np.ndarray = trn_data[category].pop('DEATH_EVENT').values
            trn_X_b: np.ndarray = trn_data[category].values
            # Test
            tst_y_b: np.ndarray = tst_data[category].pop('DEATH_EVENT').values
            tst_X_b: np.ndarray = tst_data[category].values

            trn_x_b_lst[category].append(trn_X_b.copy())
            trn_y_b_lst[category].append(trn_y_b.copy())
            tst_x_b_lst[category].append(tst_X_b.copy())
            tst_y_b_lst[category].append(tst_y_b.copy())

        for category in ['Original', 'UnderSample', 'OverSample', 'SMOTE']:
            # Train
            trn_y_fs: np.ndarray = trn_data_fs[category].pop('DEATH_EVENT').values
            trn_X_fs: np.ndarray = trn_data_fs[category].values
            # Test
            tst_y_fs: np.ndarray = tst_data_fs[category].pop('DEATH_EVENT').values
            tst_X_fs: np.ndarray = tst_data_fs[category].values

            trn_x_fs_lst[category].append(trn_X_fs.copy())
            trn_y_fs_lst[category].append(trn_y_fs.copy())
            tst_x_fs_lst[category].append(tst_X_fs.copy())
            tst_y_fs_lst[category].append(tst_y_fs.copy())

    final_lst_b = {}
    final_lst_fs = {}
    for category in ['Original', 'UnderSample', 'OverSample', 'SMOTE']:
        final_lst_b[category] = [trn_x_b_lst[category], trn_y_b_lst[category], tst_x_b_lst[category], tst_y_b_lst[category]]
        final_lst_fs[category] = [trn_x_fs_lst[category], trn_y_fs_lst[category], tst_x_fs_lst[category], tst_y_fs_lst[category]]

    all_datas_splits[all_datas_index[c][0]] = final_lst_b.copy()
    
    all_datas_splits[all_datas_index[c][1]] = final_lst_fs.copy()

    c += 1

#datas_outliers_featureselection = prepfunctions.mask_feature_selection(datas_outliers.copy(), 'DEATH_EVENT', False, './Results/FeatureSelection/HFCR Feature Selection - Features')

#datas_outliers_scaling_featureselection = prepfunctions.mask_feature_selection(datas_outliers_scaling.copy(), 'DEATH_EVENT', False, './Results/FeatureSelection/HFCR Feature Selection - Features')

#datas_scaling_featureselection = prepfunctions.mask_feature_selection(datas_scaling.copy(), 'DEATH_EVENT', False, './Results/FeatureSelection/HFCR Feature Selection - Features')

#datas_featureselection = prepfunctions.mask_feature_selection(datas.copy(), 'DEATH_EVENT', False, './Results/FeatureSelection/HFCR Feature Selection - Features')

#all_datas = [datas, datas_outliers, datas_scaling, datas_featureselection, datas_outliers_scaling, datas_outliers_featureselection, datas_outliers_scaling_featureselection]
all_datas_names = ['', ' - No Outliers', ' - Feature Selection', ' - No Outliers & Feature Selection']

accuracies = {}
for key in ['Original', 'UnderSample', 'OverSample', 'SMOTE']:
    best_accuracy = -1
    last_accuracy = -1
    offset = 2
    count = 0
    for dt in range(len(all_datas_splits)):
        if(dt != count): continue
        trn_x_lst = all_datas_splits[dt][key][0]
        trn_y_lst = all_datas_splits[dt][key][1]
        tst_x_lst = all_datas_splits[dt][key][2]
        tst_y_lst = all_datas_splits[dt][key][3]
        if(all_datas_names[count] == ''):
            subDir = graphsDir + key + '/' + 'First' + '/'
        else:
            subDir = graphsDir + key + '/' + all_datas_names[count] + '/'
        if not os.path.exists(subDir):
            os.makedirs(subDir)

        print('HFCR Naive Bayes - Performance & Confusion matrix')

        clf = GaussianNB()
        prd_trn_lst = []
        prd_tst_lst = []
        test_accuracy = 0
        train_accuracy = 0
        for trn_X, trn_y, tst_X, tst_y in zip(trn_x_lst, trn_y_lst, tst_x_lst, tst_y_lst):
            clf.fit(trn_X, trn_y)
            prd_trn = clf.predict(trn_X)
            prd_tst = clf.predict(tst_X)

            train_accuracy += metrics.accuracy_score(trn_y, prd_trn)
            test_accuracy += metrics.accuracy_score(tst_y, prd_tst)

            prd_trn_lst.append(prd_trn)
            prd_tst_lst.append(prd_tst)

        if(count == 0): text = key
        else: text = all_datas_names[count] + ' - ' + key
        accuracies[text] = [train_accuracy/n_splits, test_accuracy/n_splits]

        last_accuracy = test_accuracy/n_splits

        ds.plot_evaluation_results_kfold([0,1], trn_y_lst, prd_trn_lst, tst_y_lst, prd_tst_lst)
        plt.suptitle('HFCR Naive Bayes - ' + key + ' - Performance & Confusion matrix')
        plt.savefig(subDir + 'HFCR Naive Bayes - ' + key + ' - Performance & Confusion matrix')


        print('HFCR Naive Bayes - Comparison of Naive Bayes Models')
        estimators = {#'GaussianNB': GaussianNB(),
              'MultinomialNB': MultinomialNB(),
              'BernoulyNB': BernoulliNB()}

        xvalues = []
        yvalues = []
        xvalues.append('GaussianNB')
        yvalues.append(test_accuracy/n_splits)
        accuracy = 0
        for clf in estimators:
            xvalues.append(clf)
            for trn_X, trn_y, tst_X, tst_y in zip(trn_x_lst, trn_y_lst, tst_x_lst, tst_y_lst):
                estimators[clf].fit(trn_X, trn_y)
                prd_tst = estimators[clf].predict(tst_X)

                accuracy += metrics.accuracy_score(tst_y, prd_tst)

            yvalues.append(accuracy/n_splits)

        plt.figure(figsize=(7,7))
        ds.bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
        plt.suptitle(subDir + 'HFCR Naive Bayes - ' + key + ' - Comparison of Naive Bayes Models')
        plt.savefig(subDir + 'HFCR Naive Bayes - ' + key + ' - Comparison of Naive Bayes Models')

        if(offset == 1):
            break
        if(last_accuracy > best_accuracy and best_accuracy != -1):
            best_accuracy = last_accuracy
            last_accuracy = -1
            count += offset
            offset -= 1
        elif(best_accuracy == -1):
            best_accuracy = last_accuracy
            count += 1
        else:
            count += 1
            offset -= 1


        plt.close("all")
        plt.clf()

for a in accuracies:
    print(str(accuracies[a]) + ' <-- ' + a)

plt.style.use('dslabs.mplstyle')

my_palette = {'yellow': '#ECD474', 'pale orange': '#E9AE4E', 'salmon': '#E2A36B', 'orange': '#F79522', 'dark orange': '#D7725E',
              'pale acqua': '#92C4AF', 'acqua': '#64B29E', 'marine': '#3D9EA9', 'green': '#45BF7E', 'pale green': '#A2F0AC', 'olive': '#99C244',
              'pale blue': '#BDDDE0', 'blue2': '#199ED5', 'blue3': '#1DAFE5', 'dark blue': '#0C70A2',
              'pale pink': '#D077AC', 'pink': '#EA4799', 'lavender': '#E09FD5', 'lilac': '#B081B9', 'purple': '#923E97',
              'white': '#FFFFFF', 'light grey': '#D2D3D4', 'grey': '#939598', 'black': '#000000'}

colors_pale = [my_palette['salmon'], my_palette['blue2'], my_palette['acqua']]
colors_soft = [my_palette['dark orange'], my_palette['dark blue'], my_palette['green']]
colors_live = [my_palette['orange'], my_palette['blue3'], my_palette['olive']]
blues = [my_palette['pale blue'], my_palette['blue2'], my_palette['blue3'], my_palette['dark blue']]
oranges = [my_palette['pale orange'], my_palette['salmon'], my_palette['orange'], my_palette['dark orange']]
cmap_orange = clrs.LinearSegmentedColormap.from_list("myCMPOrange", oranges)
cmap_blues = clrs.LinearSegmentedColormap.from_list("myCMPBlues", blues)

LINE_COLOR = my_palette['dark blue']
FILL_COLOR = my_palette['pale blue']
DOT_COLOR = my_palette['blue3']
ACTIVE_COLORS = [my_palette['blue3'], my_palette['blue2'], my_palette['dark blue'],
                 my_palette['yellow'], my_palette['pale orange'], my_palette['dark orange'],
                 my_palette['lavender'], my_palette['pale pink'], my_palette['purple'],
                 my_palette['pale green'], my_palette['olive'], my_palette['green'], my_palette['acqua']]

alpha = 0.3
plt.rcParams['axes.prop_cycle'] = cycler('color', ACTIVE_COLORS)

plt.rcParams['text.color'] = LINE_COLOR
plt.rcParams['patch.edgecolor'] = LINE_COLOR
plt.rcParams['patch.facecolor'] = FILL_COLOR
plt.rcParams['axes.facecolor'] = my_palette['white']
plt.rcParams['axes.edgecolor'] = my_palette['grey']
plt.rcParams['axes.labelcolor'] = my_palette['grey']
plt.rcParams['xtick.color'] = my_palette['grey']
plt.rcParams['ytick.color'] = my_palette['grey']

plt.rcParams['grid.color'] = my_palette['light grey']

plt.rcParams['boxplot.boxprops.color'] = FILL_COLOR
plt.rcParams['boxplot.capprops.color'] = LINE_COLOR
plt.rcParams['boxplot.flierprops.color'] = my_palette['pink']
plt.rcParams['boxplot.flierprops.markeredgecolor'] = FILL_COLOR
plt.rcParams['boxplot.flierprops.markerfacecolor'] = FILL_COLOR
plt.rcParams['boxplot.whiskerprops.color'] = LINE_COLOR
plt.rcParams['boxplot.meanprops.color'] = my_palette['purple']
plt.rcParams['boxplot.meanprops.markeredgecolor'] = my_palette['purple']
plt.rcParams['boxplot.meanprops.markerfacecolor'] = my_palette['purple']
plt.rcParams['boxplot.medianprops.color'] = my_palette['green']


plt.rcParams['axes.prop_cycle'] = cycler('color', ACTIVE_COLORS)

plt.figure(figsize=(7,7))
ds.multiple_bar_chart(['Train', 'Test'], accuracies, ylabel='Accuracy')
plt.suptitle('HFCR Accuracy Comparison')
plt.savefig(graphsDir + 'HFCR Accuracy Comparison')