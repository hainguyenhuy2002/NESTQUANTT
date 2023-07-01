import numpy as np
import pandas as pd
import lightgbm as lgb
from utils import *

def get_return(df, time, name):
    dff = df.copy()
    dff[f'RETURN_{name}_{time}'] = np.log(dff[f'CLOSE_{name}'] / dff[f'CLOSE_{name}'].shift(time))
    return dff

def get_vola(df, time, isReturn, name):
    dff = df.copy()
    if isReturn == False:
        dff = get_return(dff, 1)
    dff[f"std_{time}"] = dff[f'RETURN_{name}_1'].rolling(window=time).std()
    dff["std_long"] = dff[f'RETURN_{name}_1'].rolling(window=336).std()
    dff[f"VOLABILITY_{name}_{time}"]  = dff[f"std_{time}"]/ dff["std_long"]

    dff = dff.drop(f"std_{time}", axis = 1)
    dff = dff.drop("std_long", axis = 1)

    return dff


def preprocess_df(df, name, start, end):
    for col in df.columns:
        if (col == 'YEAR_AND_MONTH') | (col =="SYMBOL"):
            df = df.drop(col, axis = 1)
        else:
            df[f'{col}_{name}'] = df[col]
            df = df.drop(col, axis = 1)

    for i in range(start, end):
        df = get_return(df, i, name)
        if i == 1:
            continue
        else:
            df = get_vola(df, i, True, name)

    return df


# Original dataframe
def get_dupp(dff):
# Convert the 'date' column to datetime type
    dff['DATE'] = pd.to_datetime(dff['DATE'])

    # Create a new dataframe to store the expanded rows
    expanded_dff = pd.DataFrame()

    # Iterate over each row in the original dataframe
    for index, row in dff.iterrows():
        date = row['DATE']
        values = row[1:]  # Exclude the 'date' column

        # Create a datetime range for the 24 hours of the day
        hour_range = pd.date_range(date, periods=24, freq='H')

        # Create a temporary dataframe with the expanded rows
        temp_dff = pd.DataFrame(hour_range, columns=['DATE'])

        # Duplicate the values for each hour
        for column, value in zip(dff.columns[1:], values):
            temp_dff[column] = value

        # Append the temporary dataframe to the expanded dataframe
        expanded_dff = expanded_dff.append(temp_dff, ignore_index=True)
    return expanded_dff


def check_cor(df, rangee, delta, time_lst, target_feature):
    corr_tuple = tuple()



    for i in range(rangee, len(time_lst)- delta, delta):
        df_tmp = df[(df.OPEN_TIME >= time_lst[i-rangee])&(df.OPEN_TIME < time_lst[i-delta])]
        correlation_values = df_tmp.corrwith(df_tmp[target_feature]).drop(target_feature)

        correlation_array = correlation_values.values
        contains_nan = np.isnan(correlation_array).any()
        if contains_nan:
            continue
        corr_tuple+= (correlation_array,)

    # Vertically stack the arrays
    stacked_array = np.vstack(corr_tuple)

    # Calculate the median array
    median_array = np.median(stacked_array, axis=0)
    df_corr = pd.DataFrame(correlation_values).reset_index()
    df_corr.drop(0, axis=1)
    df_corr["median_corr"] = median_array.flatten()
    df_corr["absolute_corr"] = df_corr["median_corr"].apply(abs)
    df_corr = df_corr.sort_values("absolute_corr", ascending=False)
    return df_corr



def check_featureImportant_slow(df, rangee, delta, time_lst, target_feature):
    blockPrint()
    models = []
    t = 0
    for i in range(rangee, len(time_lst), delta):
        t+=1
        tmp_train_df = df[(df.OPEN_TIME >= time_lst[i-rangee])&(df.OPEN_TIME < time_lst[i-delta])]
        x_trainn = tmp_train_df.drop([f'{target_feature}','OPEN_TIME'],axis=1)
        y_trainn = tmp_train_df[f"{target_feature}"]

        tmp_valid_df = df[(df.OPEN_TIME >= time_lst[i-delta])&(df.OPEN_TIME < time_lst[i])]
        x_validd = tmp_valid_df.drop([f'{target_feature}','OPEN_TIME'],axis=1)
        y_validd = tmp_valid_df[f"{target_feature}"]

        train_data = lgb.Dataset(x_trainn, label=pd.DataFrame(y_trainn), params={'verbose': -1})
        valid_data = lgb.Dataset(pd.DataFrame(x_validd), label=pd.DataFrame(y_validd), params={'verbose': -1}, reference=train_data)

        """
        optimizable
        """

        param = { 
            'boosting_type': 'goss',
            'max_depth': 4,
            'num_leaves': 15,
            'learning_rate': 0.08,
            'objective': "regression",
            'early_stopping_rounds': 64,
            'metric': 'mse',
            'num_boost_round': 100,
            'num_iterations': 256
        #     'bagging_fraction': 0.8
        }
        if  t == 1:
            model = lgb.train(
                param,
                train_data,
                valid_sets=[train_data, valid_data],
                valid_names=['train', 'valid'],
                verbose_eval=False)
        else:
            model = lgb.train(
                param,
                train_data,
                valid_sets=[train_data, valid_data],
                valid_names=['train', 'valid'],
                verbose_eval=False,
                init_model = models[-1])

        #model_predicted_scores.append([(model.predict(x_trainn)-y_trainn).abs().mean(), (model.predict(x_validd)-y_validd).abs().mean(), (model.predict(x_testt)-y_testt).abs().mean()])

        models.append(model)

    

    feat_imp = pd.DataFrame([model.feature_name(), model.feature_importance("gain")]).T
    feat_imp.columns=['Name', 'Feature Importance']
    feat = feat_imp.sort_values("Feature Importance", ascending=False)
    return feat


def check_featureImportant_fast(df, rangee, delta, time_lst, target_feature):
    blockPrint()
    models = []
    t = 0
    for i in range(rangee, len(time_lst), delta):
        t+=1
        tmp_train_df = df[(df.OPEN_TIME >= time_lst[i-rangee])&(df.OPEN_TIME < time_lst[i-delta])]
        x_trainn = tmp_train_df.drop([f'{target_feature}','OPEN_TIME'],axis=1)
        y_trainn = tmp_train_df[f"{target_feature}"]

        tmp_valid_df = df[(df.OPEN_TIME >= time_lst[i-delta])&(df.OPEN_TIME < time_lst[i])]
        x_validd = tmp_valid_df.drop([f'{target_feature}','OPEN_TIME'],axis=1)
        y_validd = tmp_valid_df[f"{target_feature}"]

        train_data = lgb.Dataset(x_trainn, label=pd.DataFrame(y_trainn), params={'verbose': -1})
        valid_data = lgb.Dataset(pd.DataFrame(x_validd), label=pd.DataFrame(y_validd), params={'verbose': -1}, reference=train_data)

        """
        optimizable
        """

        param = { 
            'boosting_type': 'goss',
            'max_depth': 4,
            'num_leaves': 15,
            'learning_rate': 0.08,
            'objective': "regression",
            'early_stopping_rounds': 64,
            'metric': 'mse',
            'num_boost_round': 100,
            'num_iterations': 256
        #     'bagging_fraction': 0.8
        }
        model = lgb.train(
                param,
                train_data,
                valid_sets=[train_data, valid_data],
                valid_names=['train', 'valid'],
                verbose_eval=False)

        #model_predicted_scores.append([(model.predict(x_trainn)-y_trainn).abs().mean(), (model.predict(x_validd)-y_validd).abs().mean(), (model.predict(x_testt)-y_testt).abs().mean()])

        models.append(model)
    important_tuple = tuple()
    for i in models:
        feat_imp = pd.DataFrame([i.feature_name(), i.feature_importance("gain")]).T
        feat_imp.columns=['Name', 'Feature Importance']
        feat = feat_imp.sort_values("Name", ascending=False)
        array_important = feat['Feature Importance'].values
        important_tuple+= (array_important,)

    stacked_array = np.vstack(important_tuple)

    # Calculate the median array
    median_array = np.median(stacked_array, axis=0)
    df_importancee = pd.DataFrame(feat).reset_index()
    df_importancee.drop("Feature Importance", axis=1)
    df_importancee["median_importance"] = median_array.flatten()
    # df_importancee["absolute_corr"] = df_importancee["median_corr"].apply(abs)
    df_importancee = df_importancee.sort_values("median_importance", ascending=False)



    return df_importancee,important_tuple