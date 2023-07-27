
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


def get_score(trial):
    rangee = trial.suggest_int('rangee', 600, 1000, 20) 
    num_boost_round = trial.suggest_int('num_boost_round', 80, 150, 10) 
    model = train(rangee,num_boost_round)


    s = Submission(api_key='svx8ZNYrgMNyuithrHdnLEAkn7OzlBKp8h5rzy2e')
    data_set = submit.to_dict('records')
    timestamp = s.submit(True, data=data_set, symbol='BTC')
    #all_rec = s.get_submission_time(is_backtest=True, symbol='BTC')['BTCUSDT']
    results = s.get_result(is_backtest=True, submission_time=int(timestamp), symbol='BTC')
    s.delete_record(is_backtest=True, submission_time=int(timestamp), symbol='BTC')

    with open("/kaggle/working/model_{}.pickle".format(trial.number), "wb") as fout:
        pickle.dump(model, fout)
    
    submit.to_csv(f"/kaggle/working/submit_{trial.number}.csv")
    
    Movement_score = "MOVEMENT_SCORE"
    mean_movement = calculate_mean(results['Movement Score'], Movement_score)

    Correlation_score = "CORRELATION"
    mean_correlation = calculate_mean(results['Correlation'], Correlation_score)

    trueContribution_score = "TRUE_CONTRIBUTION"
    mean_trueContribution = calculate_mean(results['True Contribution'], trueContribution_score)

    Overall_score = (2*mean_movement + 2* mean_correlation + mean_trueContribution)/5
    return Overall_score
