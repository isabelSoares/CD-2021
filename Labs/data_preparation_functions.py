import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

def prepare_dataset(data, target, do_scaling, do_outliers):
    # Outliers Removal with Winsorization
    if (do_outliers): data = outliers_removal(data, target)
    # Scaling
    if (do_scaling): data = scaling(data, target, True)
    # Data Balancing
    datas = data_balancing(data, target)

    return datas

def outliers_removal(data, target):
    # Winsorization
    for var in data:
        if var == target:
            break
        if var == "anaemia" or var == "diabetes" or var == "high_blood_pressure" or var == "sex" or var == "smoking":
            continue
        q1 = data[var].quantile(0.25)
        q3 = data[var].quantile(0.75)
        iqr = q3 - q1
        lower_limit = q1 -  1.5*iqr
        higher_limit = q3 + 1.5*iqr

        acceptable_values = data.loc[(data[var] >= lower_limit) & (data[var] <= higher_limit)]

        var_mean = acceptable_values[var].mean()
        max_value = acceptable_values[var].max()
        min_value = acceptable_values[var].min()


        data.loc[(data[var] < min_value), var] = min_value
        data.loc[(data[var] > max_value), var] = max_value
    
    return data

def scaling(data, target, z_score = True):
    target_collumn = data.pop(target)

    cols_nr = data.select_dtypes(include='number')
    #cols_sb = data.select_dtypes(include='category')

    imp_nr = SimpleImputer(strategy='mean', missing_values=np.nan, copy=True)
    imp_sb = SimpleImputer(strategy='most_frequent', missing_values='', copy=True)
    df_nr = pd.DataFrame(imp_nr.fit_transform(cols_nr), columns=cols_nr.columns)
    #df_sb = pd.DataFrame(imp_sb.fit_transform(cols_sb), columns=cols_sb.columns)

    if (z_score):
        # Using Normalization with Z-score
        transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_nr)
        df_nr = pd.DataFrame(transf.transform(df_nr), columns= df_nr.columns)
        #norm_data = df_nr.join(cols_sb, how='right')
        norm_data = df_nr.join(target_collumn, how='right')
    else:
        # Using Min - Max
        transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_nr)
        df_nr = pd.DataFrame(transf.transform(df_nr), columns= df_nr.columns)
        #norm_data = df_nr.join(cols_sb, how='right')
        norm_data = df_nr.join(target_collumn, how='right')
    
    return norm_data

def data_balancing(data, target):
    datas = {}
    datas['Original'] = data.copy()

    target_count = data[target].value_counts()
    min_class = target_count.idxmin()
    ind_min_class = target_count.index.get_loc(min_class)
    df_class_min = data[data[target] == min_class]
    df_class_max = data[data[target] != min_class]

    # By Undersampling
    new_df = df_class_min.copy()
    df_under = df_class_max.copy().sample(len(df_class_min))
    new_df = pd.concat([new_df, df_under], sort=False).sort_index()
    new_df = new_df.reset_index(drop=True)
    datas['UnderSample'] = new_df

    #By Oversampling
    new_df = df_class_max.copy()
    df_over = df_class_min.copy().sample(len(df_class_max), replace=True)
    new_df = pd.concat([new_df, df_over], sort=False).sort_index()
    new_df = new_df.reset_index(drop=True)
    datas['OverSample'] = new_df.copy()

    #By SMOTE
    RANDOM_STATE = 42
    smote = SMOTE(sampling_strategy='minority', random_state=RANDOM_STATE)
    y = data.pop(target).values
    X = data.values
    smote_X, smote_y = smote.fit_sample(X, y)
    df_smote = pd.DataFrame(smote_X, columns=data.columns)
    df_smote[target] = smote_y
    df_smote = new_df.reset_index(drop=True)
    datas['SMOTE'] = df_smote.copy()

    return datas

def mask_feature_selection(datas, target, features_are_numbers, mask_file):
    features_file = open(mask_file, 'r')
    lines = features_file.readlines()
    new_datas = {}
  
    count = 0
    for line in lines:
        if (line == "\n"): break
        #print(line)
        line = line.strip()
        #print(line)
        divided = line.split(sep = ": ")

        key = divided[0]
        list_of_features = (((divided[1])[1:-1]).replace("'", "")).split(sep = ", ")
        if (features_are_numbers): list_of_features = [ int(x) for x in list_of_features ]
        list_of_features.append(target)
        
        data = datas[key].copy()
        target_collumn = data[target]
        new_data = data[list_of_features]
        print(key, ": ", data.shape, " -> ", new_data.shape)
        new_datas[key] = new_data
    
    return new_datas

def dummification(data, boolean_attributes = [], bins = 0, strategy = 'uniform'):
    for attribute in boolean_attributes:
        data[attribute] = data[attribute].astype('bool')

    numeric_columns = data.select_dtypes(include='number').columns
    new_data = pd.DataFrame()
    for column in data.columns:
        if column not in numeric_columns: new_data = pd.concat((new_data, pd.DataFrame(data[column], columns=[column])), 1)
        else :
            est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=strategy)
            transform_data = est.fit_transform(data[column].values.reshape(-1, 1))
            new_data = pd.concat((new_data, pd.DataFrame(transform_data, columns=[column])), 1)

    temp_data = new_data
    new_data = pd.DataFrame()
    one_hot_encoder = OneHotEncoder(sparse=False, drop='if_binary')
    for column in temp_data.columns:
        one_hot_encoder.fit(temp_data[column].values.reshape(-1, 1))
        feature_names = one_hot_encoder.get_feature_names([str(column)])
        transformed_data = one_hot_encoder.transform(temp_data[column].values.reshape(-1, 1))
        new_data = pd.concat((new_data, pd.DataFrame(transformed_data, columns=feature_names)), 1)
    
    return new_data