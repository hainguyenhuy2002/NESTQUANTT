
import pandas as pd
import lightgbm as lgb
import pickle

from preproces.preprocess import *



def train(df,rangee, num_boost_round):

    tmp_train_df = df.iloc[-rangee:]
    x_trainn = tmp_train_df.drop(['LABEL_BTC','OPEN_TIME'],axis=1)
    y_trainn = tmp_train_df["LABEL_BTC"]

    train_data = lgb.Dataset(x_trainn, label=pd.DataFrame(y_trainn), params={'verbose': -1})
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
        'num_boost_round': num_boost_round,
        'num_iterations': 128,
    #     'bagging_fraction': 0.8
    }
    model = lgb.train(
    param,
    train_data, 
    verbose_eval=False)
    return model



