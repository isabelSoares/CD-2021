import os
import numpy as np
import pandas as pd
import ds_functions as ds
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import data_preparation_functions as prepfunctions
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import StratifiedKFold

graphsDir = './Results/Naive Bayes/'
if not os.path.exists(graphsDir):
    os.makedirs(graphsDir)

print('--------------------------------------')
print('-                                    -')
print('-     HFCR Naive Bayes - Treated     -')
print('-                                    -')
print('--------------------------------------')

data: pd.DataFrame = pd.read_csv('../Dataset/heart_failure_clinical_records_dataset.csv')
datas = prepfunctions.prepare_dataset(data, 'DEATH_EVENT', False, True)
featured_datas = prepfunctions.mask_feature_selection(datas, 'DEATH_EVENT', False, './Results/FeatureSelection/HFCR Feature Selection - Features')
best_accuracies = {}

for key in datas:
    for do_feature_eng in [False, True]:
        if (do_feature_eng):
            data = featured_datas[key]
            subDir = graphsDir + 'FeatureEng/' +  key + '/'
            if not os.path.exists(subDir):
                os.makedirs(subDir)
        else:
            data = datas[key]
            subDir = graphsDir + key + '/'
            if not os.path.exists(subDir):
                os.makedirs(subDir)

        print('HFCR Naive Bayes - Performance & Confusion matrix')
        y: np.ndarray = data.pop('DEATH_EVENT').values
        X: np.ndarray = data.values
        labels = pd.unique(y)
        
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        splitIterator = iter(skf.split(X, y))

        clf = GaussianNB()
        best_iteration_accuracy = 0
        model_sets = ([], [], [], [])
        for model in splitIterator:
            trnX = X[model[0]]
            trnY = y[model[0]]
            tstX = X[model[1]]
            tstY = y[model[1]]

            clf.fit(trnX, trnY)
            prd_trn = clf.predict(trnX)
            prd_tst = clf.predict(tstX)

            iteration_accuracy = metrics.accuracy_score(tstY, prd_tst)

            if iteration_accuracy > best_iteration_accuracy:
                best_train_accuracy = metrics.accuracy_score(trnY, prd_trn)
                best_iteration_accuracy = iteration_accuracy
                model_sets = (trnY, prd_trn, tstY, prd_tst)

        text = key
        if (do_feature_eng): text += ' with FS'
        best_accuracies[text] = [best_train_accuracy, best_iteration_accuracy]

        ds.plot_evaluation_results(pd.unique(y), model_sets[0], model_sets[1], model_sets[2], model_sets[3])
        plt.suptitle('HFCR Naive Bayes - ' + key + ' - Performance & Confusion matrix')
        plt.savefig(subDir + 'HFCR Naive Bayes - ' + key + ' - Performance & Confusion matrix')


        print('HFCR Naive Bayes - Comparison of Naive Bayes Models')
        estimators = {'GaussianNB': GaussianNB(),
                    'MultinomialNB': MultinomialNB(),
                    'BernoulyNB': BernoulliNB()}

        xvalues = []
        yvalues = []
        for clf in estimators:
            xvalues.append(clf)
            splitIterator = iter(skf.split(X, y))

            best_iteration_accuracy = 0
            for model in splitIterator:
                trnX = X[model[0]]
                trnY = y[model[0]]
                tstX = X[model[1]]
                tstY = y[model[1]]

                estimators[clf].fit(trnX, trnY)
                prd_tst = estimators[clf].predict(tstX)

                iteration_accuracy = metrics.accuracy_score(tstY, prd_tst)

                if iteration_accuracy > best_iteration_accuracy:
                    best_iteration_accuracy = iteration_accuracy

            yvalues.append(best_iteration_accuracy)

        plt.figure(figsize=(7,7))
        ds.bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
        plt.suptitle(subDir + 'HFCR Naive Bayes - ' + key + ' - Comparison of Naive Bayes Models')
        plt.savefig(subDir + 'HFCR Naive Bayes - ' + key + ' - Comparison of Naive Bayes Models')


        plt.close("all")

plt.figure(figsize=(7,7))
ds.multiple_bar_chart(['Train', 'Test'], best_accuracies, ylabel='Accuracy')
plt.suptitle('HFCR Sampling & Feature Selection')
plt.savefig(graphsDir + 'HFCR Sampling & Feature Selection')