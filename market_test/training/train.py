
import pandas as pd
import lightgbm as lgb
import pickle

from preproces.preprocess import *
from src.submit import Submission
from training.predict import *


def train(df,rangee, num_boost_round, Labeltime):

    tmp_train_df = df[(df["OPEN_TIME"]>= (Labeltime -rangee*3600000)) & (df["OPEN_TIME"]<=(Labeltime))]
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

import statistics

def calculate_mean(list_of_dicts, key):
    values = [d[key] for d in list_of_dicts if key in d]
    if not values:
        return None
    return statistics.mean(values)



