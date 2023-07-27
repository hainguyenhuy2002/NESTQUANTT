import os
import pandas as pd
import lightgbm as lgb
import numpy as np
def get_immediate_subfolder_paths(folder_path):
    subfolder_paths = []
    
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            subfolder_paths.append(subfolder_path)
    
    return subfolder_paths

def get_TIME(x):
    return x - x%3600000

def clean_data(df):
    dff = df.copy() 
    dff["OPEN_TIME"] = dff["OPEN_TIME"].apply(get_TIME)

    dff = dff.groupby("OPEN_TIME").agg({"OPEN": "first", "HIGH": "max","LOW":"min", "CLOSE": "last","VOLUME":"sum"})  
    return dff

#Merge external data
def join_data(df_external,df_target):
    df_final = df_target.copy()
    df_final = df_final.join(df_external, how = 'left')
    
    df_final = df_final.reset_index()
    
    return df_final


def get_clean_ImportanceFeat(ImportanceFeatDf):
    
    lst_feat  = ImportanceFeatDf.Name.tolist()
    final_feat = []
    check_feat = []
    for i in lst_feat:
        lst_i = i.split('_')[0:2]
        if lst_i not in check_feat:
            check_feat.append(lst_i)
            final_feat.append(i)
    return final_feat
    
       
        
def get_finalDataframe(df_lst, df_target, is_realtime):
    df_final =join_data(df_lst, df_target)
    volumn_col = []
    for i in df_final.columns:
        lst_i = i.split('_')
        if "VOLUME" in lst_i:
            volumn_col.append(i)


    # Fill NaN values with 0 in the specified columns
    df_final[volumn_col] = df_final[volumn_col].fillna(0)
    if is_realtime:
        df_final_withouLabel = df_final[["OPEN_TIME","LABEL_BTC"]]
        df_finall = df_final.drop("LABEL_BTC", axis = 1) 
        df_finall = df_finall.bfill().ffill()
        final = df_finall.merge(df_final_withouLabel, on="OPEN_TIME", how = "left")
    else:
        final = df_final.bfill().ffill()

    return final


# def check_feature_important_withouRolling(dff,time_train,target_feature):
#     tmp_train_df = dff[(dff.OPEN_TIME >= time_train[1])]
#     x_trainn = tmp_train_df.drop([f'{target_feature}','OPEN_TIME'],axis=1)
#     y_trainn = tmp_train_df[f"{target_feature}"]
    
    


#     # tmp_valid_df = dff[(dff.OPEN_TIME >= time_train[i + 26])&(dff.OPEN_TIME < time_train[i +26+delta])]
#     # x_validd = tmp_valid_df.drop([f'{target_feature}','OPEN_TIME'],axis=1)
#     # y_validd = tmp_valid_df[f"{target_feature}"]
    

#     train_data = lgb.Dataset(x_trainn, label=pd.DataFrame(y_trainn), params={'verbose': -1})
#     #valid_data = lgb.Dataset(pd.DataFrame(x_validd), label=pd.DataFrame(y_validd), params={'verbose': -1}, reference=train_data)

    

#     """
#     optimizable
#     """

#     param = { 
#         'boosting_type': 'goss',
#         'max_depth': 4,
#         'num_leaves': 15,
#         'learning_rate': 0.08,
#         'objective': "regression",
#         'metric': 'mse',
#         'num_boost_round': 100,
#         'num_iterations': 128,
#     #     'bagging_fraction': 0.8
#     }

#     model = lgb.train(
#     param,
#     train_data, 
#     verbose_eval=False)
    
#     feat_imp = pd.DataFrame([model.feature_name(), model.feature_importance("gain")]).T
#     feat_imp.columns=['Name', 'Feature Importance']
#     feat = feat_imp.sort_values("Feature Importance", ascending=False)
#     return feat

def check_feature_important_withouRolling(dff,time_train,target_feature):
    tmp_train_df = dff[(dff.OPEN_TIME >= time_train[1])]
    x_trainn = tmp_train_df.drop([f'{target_feature}','OPEN_TIME'],axis=1)
    y_trainn = tmp_train_df[f"{target_feature}"]
    
    


    # tmp_valid_df = dff[(dff.OPEN_TIME >= time_train[i + 26])&(dff.OPEN_TIME < time_train[i +26+delta])]
    # x_validd = tmp_valid_df.drop([f'{target_feature}','OPEN_TIME'],axis=1)
    # y_validd = tmp_valid_df[f"{target_feature}"]
    
    
    

    train_data = lgb.Dataset(x_trainn, label=pd.DataFrame(y_trainn), params={'verbose': -1})
    #valid_data = lgb.Dataset(pd.DataFrame(x_validd), label=pd.DataFrame(y_validd), params={'verbose': -1}, reference=train_data)

    

    """
    optimizable
    """

    param = { 
        'boosting_type': 'goss',
        'max_depth': 4,
        'num_leaves': 15,
        'learning_rate': 0.08,
        'objective': "regression",
        'metric': 'mse',
        'num_boost_round': 100,
        'num_iterations': 128,
        'device':'cpu'
    #     'bagging_fraction': 0.8
    }
    model = lgb.train(
    param,
    train_data,
    verbose_eval=False)
    feat_imp = pd.DataFrame([model.feature_name(), model.feature_importance("gain")]).T
    feat_imp.columns=['Name', 'Feature Importance']
    feat = feat_imp.sort_values("Feature Importance", ascending=False)
    return feat





def get_feat_importance(df_finall, feat_important_score, start_time_real,end_time_real ):
    df_final = df_finall.copy()
    time_lst = df_finall[(df_finall["OPEN_TIME"]>=start_time_real) & ( (df_finall["OPEN_TIME"]<= end_time_real))].OPEN_TIME.tolist() 
    df_feat_importance = check_feature_important_withouRolling(df_final, time_lst, 'LABEL_BTC')
    df_feat_importance = df_feat_importance[df_feat_importance["Feature Importance"] >feat_important_score]
    final_feat =get_clean_ImportanceFeat(df_feat_importance)
    # df_feat = df_feat_importance[df_feat_importance["Name"].isin(final_feat)]
    final_feat.append("OPEN_TIME")
    final_feat.append("LABEL_BTC")
    df_finall = df_final[final_feat]
    return df_finall



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
