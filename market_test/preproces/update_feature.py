import os
import gc
import math
import sklearn
import numpy as np
import pandas as pd
import lightgbm as lgb
import sys

from sklearn.model_selection import train_test_split
from datetime import datetime

from preproces.preprocess import *
from preproces.feat_importance import *
from IPython.utils import io



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



stocks_paths = []
fx_paths = []
Fred_paths = []


folder_path = '/home/ubuntu/data'
subfolders = get_immediate_subfolder_paths(folder_path)

for subfolder in subfolders:
    if "FRED" in subfolder[-6:]:
        for i in get_immediate_subfolder_paths(subfolder):
            Fred_paths.append(i)
            
    if "FX" in subfolder[-6:]:
        for i in get_immediate_subfolder_paths(subfolder):
            fx_paths.append(i)
            
    if "Stocks" in subfolder[-6:]:
        for i in get_immediate_subfolder_paths(subfolder):
            stocks_paths.append(i)


def get_whole_feature_importance(df_BTCLabel, df_SPY_clean, paths):    
    df_BTCLabel = preprocess_df(df_BTCLabel, "BTC")
    df_SPY_clean_prep = preprocess_df(df_SPY_clean,"SPY")
    df_SPY_clean_prep = df_SPY_clean_prep.dropna()
    df_final =get_finalDataframe(df_SPY_clean_prep,df_BTCLabel)
    #df_fin = get_diff(df_final, "BTC", "SPY")
    df_feat =get_feat_importance(df_final, 0.1)  #final important feature with BTC and SPY
    del df_SPY_clean_prep
    del df_SPY_clean

    for path in range(len(paths)):
     
                    
        
        for i in paths[path]:
            df_externals = pd.read_parquet(i).reset_index()
            df_externals_clean = clean_data(df_externals)
            df_externals_clean_prep = preprocess_df(df_externals_clean,i.split('/')[-1])
            df_externals_clean_prep = df_externals_clean_prep.dropna()
            

            df_external = get_finalDataframe(df_externals_clean_prep,df_BTCLabel)
            df_feats =get_feat_importance(df_external, 0.1)


            external_cols = []
            for i in df_feats.columns:
                if "BTC" not in i.split("_"):
                    external_cols.append(i)

            df_new_feats = df_external[external_cols]
            del df_externals
            df_feat = df_feat.merge(df_new_feats, on = "OPEN_TIME", how= "left")
            del df_new_feats
            
        
    
    return df_feat

if __name__ == "__main__":
    """
    BTC preprocessing
    """ 
    df_BTC = pd.read_parquet("/home/ubuntu/data/Crypto/BTCUSDT").reset_index()
    df_Label = pd.read_parquet("/home/ubuntu/data/Label/LABEL_BTCUSDT").reset_index()
    df_BTC_clean = clean_data(df_BTC)

    df_BTCLabel = df_BTC_clean.merge(df_Label, on= "OPEN_TIME",how='inner')
    df_BTCLabel = df_BTCLabel.drop(columns=["SYMBOL"])
    df_BTCLabel = df_BTCLabel.dropna()
    df_BTCLabel = df_BTCLabel.set_index("OPEN_TIME")
    """
    SPY
    """
    df_SPY = pd.read_parquet("/home/ubuntu/data/Stocks/SPY").reset_index()
    df_SPY_clean = clean_data(df_SPY)

    Fred_lst =["T1YFF","SOFR","DCOILBRENTEU","CPFF","BAA10Y"]


    paths = [fx_paths,stocks_paths]
    df_final_feat = get_whole_feature_importance(df_BTCLabel, df_SPY_clean, paths)

    Fred_lst =["T1YFF","SOFR","DCOILBRENTEU","CPFF","BAA10Y"]

    print(Fred_lst)
    df_final = df_final_feat.copy()
    for i in Fred_paths:
        if i.split('/')[-1] in Fred_lst:
            df_freds = pd.read_parquet(i)
            df_freds = get_dupp(df_freds)
            df_freds["OPEN_TIME"] = df_freds["DATE"].apply(lambda x: int(datetime.timestamp(x)) *1000)
            df_freds = df_freds.drop("DATE", axis= 1)
            df_freds[f"VALUE_{i.split('/')[-1]}"] = df_freds["VALUE"]
            df_freds = df_freds.drop("VALUE", axis =1)
            df_final = df_final.merge(df_freds, on = "OPEN_TIME", how = "left").bfill().ffill()
        

    time_lst = df_final.OPEN_TIME.tolist()[46918:] #1/1/2023
    print(time_lst)
    df_feat_importance = check_feature_important_withouRolling(df_final, time_lst, 'LABEL_BTC')
    feat = df_feat_importance[df_feat_importance["Feature Importance"]> 0]
    name_feat = feat.Name.tolist()
    name_feat.append("LABEL_BTC")
    name_feat.append("OPEN_TIME")
    df_final = df_final[name_feat]
    df_final.to_csv("/home/ubuntu/nestquant/market_test/trainData/final_data.csv")
    print("sucessful_save")
