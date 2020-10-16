import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ds_functions as ds
from sklearn.model_selection import StratifiedKFold

graphsDir = './Results/'

data: pd.DataFrame = pd.read_csv('../../Dataset/qsar_oral_toxicity.csv', sep=';', header=None)

trnX, tstX, trnY, tstY = StratifiedKFold(n_splits = 2)

#values['Train'] = [len(np.delete(trnY, np.argwhere(trnY==negative))), len(np.delete(trnY, np.argwhere(trnY==positive)))]
#values['Test'] = [len(np.delete(tstY, np.argwhere(tstY==negative))), len(np.delete(tstY, np.argwhere(tstY==positive)))]

plt.figure(figsize=(7,7))
#ds.multiple_bar_chart([positive, negative], values, title='Data distribution per dataset')
plt.suptitle('QOT Training Strategies')
plt.savefig(graphsDir + 'QOT Training Strategies')