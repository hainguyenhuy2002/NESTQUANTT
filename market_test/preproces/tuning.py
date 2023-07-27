import os
import sys
from turtle import shape
import pandas as pd
import itertools

import sys
import pytz

from datetime import datetime

from src.crawl import Crawler


from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from preproces.preprocess import *
from preproces.feat_importance import *
from preproces.get_feature import *
#from preproces.update_feature import *
from training.train import *
from training.predict import *
from IPython.utils import io

sys.path.append(os.path.dirname (os .getcwd ( ) ) )

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from src.submit import Submission

import statistics
import optuna




def calculate_mean(list_of_dicts, key):
    values = [d[key] for d in list_of_dicts if key in d]
    if not values:
        return None
    return statistics.mean(values)

start =1690210800000
end = 1690297200000
def get_score(trial):
    rangee = trial.suggest_int('rangee', 336, 336, 0) 
    data_set = pipeline(start,end,rangee,4)
    s = Submission(api_key='svx8ZNYrgMNyuithrHdnLEAkn7OzlBKp8h5rzy2e')
    timestamp = s.submit(True, data=data_set, symbol='BTC')
    # all_rec = s.get_submission_time(is_backtest=True, symbol='BTC')['BTCUSDT']
    results = s.get_result(is_backtest=True, submission_time=int(timestamp), symbol='BTC')

    s.delete_record(is_backtest=True, submission_time=int(timestamp), symbol='BTC')
    
    Movement_score = "MOVEMENT_SCORE"
    mean_movement = calculate_mean(results['Movement Score'], Movement_score)

    Correlation_score = "CORRELATION"
    mean_correlation = calculate_mean(results['Correlation'], Correlation_score)

    trueContribution_score = "TRUE_CONTRIBUTION"
    mean_trueContribution = calculate_mean(results['True Contribution'], trueContribution_score)

    Overall_score = (2*mean_movement + 2* mean_correlation + mean_trueContribution)/5
    return Overall_score



def optimization(n):
        # define custom values to search on:
        study = optuna.create_study(study_name="Optimization over given values",direction='maximize')
        study.optimize(get_score, n_trials = n)
        best_params = study.best_params
       
        best_metric = study.best_value
        return best_params, best_metric, study

def pipeline(start, end, rangee, delta):
    submitt = []

    for i in range(start, end, 3600000):
        current_hour_in_vietnam = ((i%(3600000*24))/3600000) +7 +25+2
        print("-----------------------------------------------------------------------------------------------------------------------------------------------")
        print("Start to preprocessing data")
        print("Keep goingggg 🫠 🫠")

        num_boost_round = 100
        with io.capture_output() as captured:


            if current_hour_in_vietnam%delta ==0:
                get_and_update_feature(False, rangee,i, delta)
                
            # elif current_hour_in_vietnam%4 !=2:
            #     get_and_update_feature(True)
        testting = pd.read_csv('/kaggle/working/realtime_feature.csv')
        
        
        print("....")
        if current_hour_in_vietnam%delta==0:
            print("Sucessful get preprocessed data with shape of: "+ str(testting.shape))
        
        elif current_hour_in_vietnam%delta!=0:
            print("successful updating data with shape of: "+ str(testting.shape))
        print("-----------------------------------------------------------------------------------------------------------------------------------------------")
        #training

        print("Start to train ...")
        

        path = '/kaggle/working/realtime_feature.csv'
        df_final = pd.read_csv(path)
        df_final = df_final.dropna()
        dff = df_final.copy()
        dff = dff.drop("Unnamed: 0", axis = 1)

        if current_hour_in_vietnam%delta ==0:

            model = train(dff,rangee,num_boost_round, i)
            file = f'/kaggle/working/trained_model.pkl'  ##get name in here
            pickle.dump(model, open(file, 'wb'))
            print("....")

            print("Sucessful trained data with the shape of: "+ str(testting.shape) + " 😋 😗")
            print("-----------------------------------------------------------------------------------------------------------------------------------------------")
        

    
        print("Time to predict and submit my result hehe 🥲 🥲")
        
        data = pd.read_csv('/kaggle/working/realtime_feature.csv')
        filename = '/kaggle/working/trained_model.pkl'
        model = pickle.load(open(filename, 'rb')) 
        #print(data)
        submit = get_predict(model, data, i)
        data_set = submit.to_dict('records')
        submitt.append(data_set)
        print(data_set)
        # s = Submission(api_key='svx8ZNYrgMNyuithrHdnLEAkn7OzlBKp8h5rzy2e')
        # timestamp = s.submit(False, data=data_set, symbol='BTC')
        # print("Sucessful submit to the system: 🥳 🥳 ")
        # print("Submission time: " + str(timestamp))

        print("-----------------------------------------------------------------------------------------------------------------------------------------------")

    flat_list = list(itertools.chain(*submitt))
    print(flat_list)
    return flat_list


