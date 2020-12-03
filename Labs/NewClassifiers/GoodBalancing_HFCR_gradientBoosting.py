import data_preparation_functions as prepfunctions
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import ds_functions as ds
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as clrs

from cycler import cycler

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

graphsDir = './Results/GradientBoosting/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

data: pd.DataFrame = pd.read_csv('../../Dataset/heart_failure_clinical_records_dataset.csv')
#data = prepfunctions.new_prepare_dataset(data.copy(), 'DEATH_EVENT', False, False)

data_outliers = prepfunctions.new_prepare_dataset(data.copy(), 'DEATH_EVENT', False, True)
data_outliers_scaling = prepfunctions.new_prepare_dataset(data.copy(), 'DEATH_EVENT', True, True)
data_scaling = prepfunctions.new_prepare_dataset(data.copy(), 'DEATH_EVENT', True, False)

all_datas = [data, data_outliers, data_scaling, data_outliers_scaling]
all_datas_index = [(0, 3), (1, 5), (2,-1), (4, 6)]
all_datas_splits = [{}, {}, {}, {}, {}, {}, {}]
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
    
    if(c == 2):
        datas_splits_scaling_featureselection = final_lst_fs.copy()
        c += 1
        continue
    all_datas_splits[all_datas_index[c][1]] = final_lst_fs.copy()

    c += 1

#datas_outliers_featureselection = prepfunctions.mask_feature_selection(datas_outliers.copy(), 'DEATH_EVENT', False, './Results/FeatureSelection/HFCR Feature Selection - Features')

#datas_outliers_scaling_featureselection = prepfunctions.mask_feature_selection(datas_outliers_scaling.copy(), 'DEATH_EVENT', False, './Results/FeatureSelection/HFCR Feature Selection - Features')

#datas_scaling_featureselection = prepfunctions.mask_feature_selection(datas_scaling.copy(), 'DEATH_EVENT', False, './Results/FeatureSelection/HFCR Feature Selection - Features')

#datas_featureselection = prepfunctions.mask_feature_selection(datas.copy(), 'DEATH_EVENT', False, './Results/FeatureSelection/HFCR Feature Selection - Features')

#all_datas = [datas, datas_outliers, datas_scaling, datas_featureselection, datas_outliers_scaling, datas_outliers_featureselection, datas_outliers_scaling_featureselection]
all_datas_names = ['', ' - No Outliers', ' - Scaling', ' - Feature Selection', ' - No Outliers & Scaling', ' - No Outliers & Feature Selection', ' - No Outliers, Scaling & Feature Selection']
provisorio_data_scaling = ' - Scaling & Feature Selection'

best_accuracies = {}

for key in ['Original', 'UnderSample', 'OverSample', 'SMOTE']:
    last_name = 'None'
    best_accuracy = -1
    last_accuracy = -1
    offset = 3
    count = 0
    for dt in range(len(all_datas_splits)):
        if(dt != count): continue
        #data = all_datas[dt][key]
        trn_x_lst = all_datas_splits[dt][key][0]
        trn_y_lst = all_datas_splits[dt][key][1]
        tst_x_lst = all_datas_splits[dt][key][2]
        tst_y_lst = all_datas_splits[dt][key][3]
        if(last_name == ' - Scaling' and offset == 1):
            #data = datas_scaling_featureselection.copy()[key]
            trn_x_lst = datas_splits_scaling_featureselection[key][0]
            trn_y_lst = datas_splits_scaling_featureselection[key][1]
            tst_x_lst = datas_splits_scaling_featureselection[key][2]
            tst_y_lst = datas_splits_scaling_featureselection[key][3]
            subDir = graphsDir + key + '/' + provisorio_data_scaling + '/'
            last_name = provisorio_data_scaling
        elif(all_datas_names[count] == ''):
            subDir = graphsDir + key + '/' + 'First' + '/'
            last_name = all_datas_names[count]
        else:
            subDir = graphsDir + key + '/' + all_datas_names[count] + '/'
            last_name = all_datas_names[count]
        if not os.path.exists(subDir):
            os.makedirs(subDir)

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(current_time, ": Key: ", key)

        criterions = ['friedman_mse', 'mse', 'mae']
        n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
        max_depths = [5, 10, 25]
        learning_rate = [.1, .3, .5, .7, .9]
        max_features = [None, 'auto', 'sqrt', 'log2']

        values_by_criteria = {}
        for criterion in criterions:
            criterionDir = subDir + criterion + '/'
            if not os.path.exists(criterionDir):
                os.makedirs(criterionDir)

            best = ('', '', 0, 0)
            last_best = 0
            best_tree = None

            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(current_time, ": Criterion: ", criterion)

            cols = len(max_depths)
            rows = len(max_features)
            plt.figure()
            fig, axs = plt.subplots(rows, cols, figsize=(cols*(ds.HEIGHT * 2), rows * (ds.HEIGHT + 3)), squeeze=False)
            overfitting_values = {}
            for w in range(len(max_features)):
                max_feat = max_features[w]
                max_feat_string = max_feat
                if (max_feat == None): max_feat_string = 'None'
                overfitting_values[max_feat_string] = {}

                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(current_time, ": Max Features: ", max_feat_string)

                for k in range(len(max_depths)):
                    d = max_depths[k]
                    values = {}
                    overfitting_values[max_feat_string][d] = {}

                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    print(current_time, ": D: ", d)
                    for lr in learning_rate:
                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")
                        print(current_time, ": Lr: ", lr)
                        yvalues = []
                        train_acc_values = []
                        test_acc_values = []
                        for n in n_estimators:
                            prd_trn_lst = []
                            prd_tst_lst = []
                            test_accuracy = 0
                            train_accuracy = 0
                            for trn_X, trn_y, tst_X, tst_y in zip(trn_x_lst, trn_y_lst, tst_x_lst, tst_y_lst):
                            	gb = GradientBoostingClassifier(criterion=criterion, max_features=max_feat, n_estimators=n, max_depth=d, learning_rate=lr)
                            	gb.fit(trn_X, trn_y)
                            	prd_tst = gb.predict(tst_X)
                            	prd_trn = gb.predict(trn_X)

                            	train_accuracy += metrics.accuracy_score(trn_y, prd_trn)
                            	test_accuracy += metrics.accuracy_score(tst_y, prd_tst)

                            	prd_trn_lst.append(prd_trn)
                            	prd_tst_lst.append(prd_tst)

                            test_accuracy /= n_splits
                            train_accuracy /= n_splits

                            yvalues.append(test_accuracy)
                            train_acc_values.append(train_accuracy)
                            test_acc_values.append(test_accuracy)
                            if yvalues[-1] > last_best:
                                best = (max_feat_string, d, lr, n)
                                best_model = (prd_trn_lst, prd_tst_lst)
                                last_best = yvalues[-1]
                                last_best_train = train_acc_values[-1]
                                values_by_criteria[criterion] = [last_best_train, last_best]
                                best_tree = gb
                        
                        values[lr] = yvalues
                        overfitting_values[max_feat_string][d][lr] = {}
                        overfitting_values[max_feat_string][d][lr]['train'] = train_acc_values
                        overfitting_values[max_feat_string][d][lr]['test'] = test_acc_values

                    ds.multiple_line_chart(n_estimators, values, ax=axs[w, k], title='Gradient Boorsting with max_features=%s max_depth=%d'%(max_feat_string, d),
                                        xlabel='nr estimators', ylabel='accuracy', percentage=True)
            
            print('Best results with max_features=%s, depth=%d, learning rate=%1.2f and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[2], best[3], last_best))
            fig.text(0.5, 0.03, 'Best results with max_features=%s, depth=%d, learning rate=%1.2f and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[2], best[3], last_best), fontsize=7, ha='center', va='center')
            plt.suptitle('HFCR Gradient Boosting - ' + key + ' - parameters')
            plt.savefig(criterionDir + 'HFCR Gradient Boosting - ' + key + ' - parameters')

            if(count == 0): text = key
            else: text = last_name + ' - ' + key
            if ((text not in best_accuracies.keys()) or (best_accuracies[text][1] < last_best)):
                best_accuracies[text] = [last_best_train, last_best]
                last_accuracy = last_best

            plt.figure()
            fig, axs = plt.subplots(len(max_depths), len(learning_rate), figsize=(32, 8), squeeze=False)
            for i in range(len(max_depths)):
                d = max_depths[i]
                for j in range(len(learning_rate)):
                    lr = learning_rate[j]
                    ds.multiple_line_chart(n_estimators, overfitting_values[best[0]][d][lr], ax=axs[i, j], title='Overfitting for max_depth = %d, with learning rate = %1.2f'%(d, lr), xlabel='n_estimators', ylabel='accuracy', percentage=True)
            
            plt.subplots_adjust(hspace=0.5)
            plt.suptitle('HFCR Overfitting - Gradient Boosting with max_features=%s'%best[0])
            plt.savefig(criterionDir + 'HFCR Overfitting - Gradient Boosting')
            
            prd_trn_lst = best_model[0]
            prd_tst_lst = best_model[1]

            ds.plot_evaluation_results_kfold([0,1], trn_y_lst, prd_trn_lst, tst_y_lst, prd_tst_lst)
            plt.suptitle('HFCR Gradient Boosting - ' + key + ' - Performance & Confusion matrix')
            plt.savefig(criterionDir + 'HFCR Gradient Boosting - ' + key + ' - Performance & Confusion matrix')

            plt.close("all")
            plt.clf()

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

        plt.figure(figsize=(7,7))
        ds.multiple_bar_chart(['Train', 'Test'], values_by_criteria, ylabel='Accuracy')
        plt.suptitle('HFCR Gradient Boosting Criteria')
        plt.savefig(subDir + 'HFCR Gradient Boosting Criteria')
        
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
ACTIVE_COLORS = [my_palette['pale blue'], my_palette['blue3'], my_palette['blue2'], my_palette['dark blue'],
                 my_palette['yellow'], my_palette['pale orange'], my_palette['salmon'], my_palette['dark orange'],
                 my_palette['lavender'], my_palette['pale pink'], my_palette['lilac'], my_palette['purple'],
                 my_palette['pale green'], my_palette['green'], my_palette['olive'], my_palette['marine']]

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
ds.multiple_bar_chart(['Train', 'Test'], best_accuracies, ylabel='Accuracy')
plt.suptitle('HFCR Accuracy Comparison')
plt.savefig(graphsDir + 'HFCR Accuracy Comparison')