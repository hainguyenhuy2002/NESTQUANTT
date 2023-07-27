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

def get_whole_feature_importance(df_BTCLabel,df_SPY_clean, paths, start_time,  start_time_real,end_time_real):    
    df_BTCLabel = preprocess_df(df_BTCLabel, "BTC")
    print(df_BTCLabel)
    df_SPY_clean_prep = preprocess_df(df_SPY_clean,"SPY")
    df_SPY_clean_prep = df_SPY_clean_prep.dropna()
    df_BTCLabel1 = df_BTCLabel.dropna()
    
    df_final =get_finalDataframe(df_SPY_clean_prep,df_BTCLabel1, False)
    df_final_notLabel = get_finalDataframe(df_SPY_clean_prep,df_BTCLabel, True)

    #Feature important
    df_feat =get_feat_importance(df_final, 0, start_time_real,end_time_real)  #final important feature with BTC and SPY
    col_lst = list(df_feat.columns)
    #col_lst.remove('LABEL_BTC')

    df_final_notLabel = df_final_notLabel[col_lst]
    print(df_feat)
    print(df_final_notLabel)
    del df_SPY_clean_prep
    del df_SPY_clean


    for path in range(len(paths)):   
        for i in paths[path]:
            df_externals = pd.read_parquet(i).reset_index()
            df_externals_clean = clean_data(df_externals).reset_index()
            df_externals_clean = df_externals_clean[(df_externals_clean["OPEN_TIME"]>= start_time)]
            df_externals_clean = df_externals_clean.set_index("OPEN_TIME")

            if path == 0:
                name = i.split('/')[-1][-6:]
                print(name)
            else:
                name = i.split('/')[-1]
            df_externals_clean_prep = preprocess_df(df_externals_clean,name)
            df_externals_clean_prep = df_externals_clean_prep.dropna()
            

            df_external = get_finalDataframe(df_externals_clean_prep,df_BTCLabel1, False)
            df_external_notLabel = get_finalDataframe(df_externals_clean_prep,df_BTCLabel, True)


            df_feats =get_feat_importance(df_external, 0, start_time_real,end_time_real)

            external_cols = []
            for i in df_feats.columns:
                if "BTC" not in i.split("_"):
                    external_cols.append(i)

            df_new_feats = df_external[external_cols]
            df_external_notLabel = df_external_notLabel[external_cols]
            del df_externals
            df_feat = df_feat.merge(df_new_feats, on = "OPEN_TIME", how= "left")
            df_final_notLabel = df_final_notLabel.merge(df_external_notLabel, on = "OPEN_TIME", how = "left")
            del df_new_feats
            
        
    
    return df_feat, df_final_notLabel


def update_whole_feature_importance(df_BTCLabel,df_SPY_clean, paths, feat_list):    
    df_BTCLabel = preprocess_df(df_BTCLabel, "BTC")
    df_SPY_clean_prep = preprocess_df(df_SPY_clean,"SPY")
    df_SPY_clean_prep = df_SPY_clean_prep.dropna()
    
    df_final_notLabel = get_finalDataframe(df_SPY_clean_prep,df_BTCLabel, True)

    valid_columns_to_keep = [col for col in feat_list if col in df_final_notLabel.columns]


    df_final_notLabel = df_final_notLabel[valid_columns_to_keep]
    del df_SPY_clean_prep
    del df_SPY_clean


    for path in range(len(paths)):  
        print(path) 
        for i in paths[path]:
            if i.split('/') =='SPY':
                continue
            df_externals = pd.read_parquet(i).reset_index()
            df_externals_clean = clean_data(df_externals)
            if path ==0 :
                name = i.split('/')[-1][-6:]
                print(name)
            else:
                name = i.split('/')[-1]

            df_externals_clean_prep = preprocess_df(df_externals_clean,name)
            df_externals_clean_prep = df_externals_clean_prep.dropna()
            

            df_external_notLabel = get_finalDataframe(df_externals_clean_prep,df_BTCLabel, True)
            valid_columns_to_keep = [col for col in feat_list if col in df_external_notLabel.columns and 'BTC' not in col.split("_")]



            df_external_notLabel = df_external_notLabel[valid_columns_to_keep]
            del df_externals
            df_final_notLabel = df_final_notLabel.merge(df_external_notLabel, on = "OPEN_TIME", how = "left")
            
        
    
    return df_final_notLabel

def get_and_update_feature(is_updated, range, LabelTime, delta):
        start_time = LabelTime - ((range+1000)*3600000)
        start_time_real = LabelTime - ((range)*3600000)
        end_time_real = LabelTime + delta*3600000
        stocks_paths = []
        fx_paths = []
        Fred_paths = []


        folder_path = '/kaggle/input/nq2023/data'
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
        """
        BTC preprocessing
        """ 
        df_BTC = pd.read_parquet('/kaggle/input/nq2023/data/Crypto/BTCUSDT').reset_index()
        
        df_Label = pd.read_parquet('/kaggle/input/nq2023/data/Label/LABEL_BTCUSDT').reset_index()
        df_BTC_clean = clean_data(df_BTC)
        df_BTCLabel = df_BTC_clean.merge(df_Label, on= "OPEN_TIME",how='left')
        df_BTCLabel = df_BTCLabel.drop(columns=["SYMBOL"])
        df_BTCLabel = df_BTCLabel[(df_BTCLabel["OPEN_TIME"]>= start_time)]
        #df_BTCLabel = df_BTCLabel.dropna()
        df_BTCLabel = df_BTCLabel.set_index("OPEN_TIME")
        """
        SPY
        """
        df_SPY = pd.read_parquet('/kaggle/input/nq2023/data/Stocks/SPY').reset_index()
        df_SPY_clean = clean_data(df_SPY).reset_index()
        df_SPY_clean = df_SPY_clean[(df_SPY_clean["OPEN_TIME"]>= start_time)] 
        df_SPY_clean.set_index("OPEN_TIME")
        Fred_lst =["T1YFF","SOFR","DCOILBRENTEU","CPFF","BAA10Y"]

        if is_updated == False:
            paths = [fx_paths,stocks_paths]
            df_feat, df_final_notLabel = get_whole_feature_importance(df_BTCLabel, df_SPY_clean, paths, start_time,  start_time_real,end_time_real)


            print(Fred_lst)
            for i in Fred_paths:
                if i.split('/')[-1] in Fred_lst:
                    df_freds = pd.read_parquet(i)
                    df_freds = get_dupp(df_freds)
                    df_freds["OPEN_TIME"] = df_freds["DATE"].apply(lambda x: int(datetime.timestamp(x)) *1000)
                    df_freds = df_freds.drop("DATE", axis= 1)
                    df_freds[f"VALUE_{i.split('/')[-1]}"] = df_freds["VALUE"]
                    df_freds = df_freds.drop("VALUE", axis =1)
                    df_feat = df_feat.merge(df_freds, on = "OPEN_TIME", how = "left").bfill().ffill()
                    Label = df_final_notLabel[["OPEN_TIME", "LABEL_BTC"]]
                    WithouLabel = df_final_notLabel.drop("LABEL_BTC", axis = 1)
                    WithouLabel = WithouLabel.merge(df_freds, on = "OPEN_TIME", how = "left").bfill().ffill()
                    df_final_notLabel = Label.merge(WithouLabel, on = "OPEN_TIME", how = "left")


            time_lst = df_feat[(df_feat["OPEN_TIME"]>=start_time_real) & ( (df_feat["OPEN_TIME"]<= end_time_real))].OPEN_TIME.tolist() #1/1/2023
            print(time_lst)
            df_feat_importance = check_feature_important_withouRolling(df_feat, time_lst, 'LABEL_BTC')
            feat = df_feat_importance[df_feat_importance["Feature Importance"]> 0]
            name_feat = feat.Name.tolist()
            name_feat.append("LABEL_BTC")
            name_feat.append("OPEN_TIME")
            df_final_notLabel = df_final_notLabel[name_feat]


            df_final_notLabel.to_csv('/kaggle/working/realtime_feature.csv')
            print("sucessful_save")

        if is_updated == True:
            old_df = pd.read_csv("/home/ubuntu/nestquant/market_test/realtimeData/realtime_feature.csv")
            lst_feat = list(old_df.columns)


            paths = [fx_paths,stocks_paths]
            #df_feat, df_final_notLabel = get_whole_feature_importance(df_BTCLabel, df_SPY_clean, paths)
            df_final_notLabel = update_whole_feature_importance(df_BTCLabel,df_SPY_clean, paths, lst_feat)
            print(df_final_notLabel)
            df_final_notLabel = df_final_notLabel.drop("LABEL_BTC", axis = 1)

            df_Label =  pd.read_parquet("/home/ubuntu/data/Label/LABEL_BTCUSDT").reset_index()
            df_Label = df_Label.rename(columns={"LABEL":"LABEL_BTC"}).drop("SYMBOL", axis = 1)

            df_updated = df_final_notLabel.merge(df_Label, on= "OPEN_TIME", how ="left")
            df_updated.to_csv("/home/ubuntu/nestquant/market_test/realtimeData/realtime_feature.csv")
 






