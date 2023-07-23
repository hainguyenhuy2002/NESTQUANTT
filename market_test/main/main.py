import os
import sys
from turtle import shape
import pandas as pd

import sys
import pytz

from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '..'))

from src.crawl import Crawler


from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from preproces.preprocess import *
from preproces.feat_importance import *
from preproces.get_feature import *
from preproces.update_feature import *
from training.train import *
from training.predict import *
from IPython.utils import io

sys.path.append(os.path.dirname (os .getcwd ( ) ) )

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from src.submit import Submission




if __name__ == "__main__":
    from datetime import datetime

    # Set the time zone to Vietnamese time (Indochina Time)
    vietnam_timezone = pytz.timezone('Asia/Ho_Chi_Minh')

    # Get the current time in the Vietnamese time zone
    current_time_in_vietnam = datetime.now(vietnam_timezone)

    # Get the hour from the current time
    current_hour_in_vietnam = current_time_in_vietnam.hour
    # print("current hours: "+ str(current_hour_in_vietnam)) 

    print("-----------------------------------------------------------------------------------------------------------------------------------------------")
    print("Start to crawlling ...")
    print("Please wait ... 🙄 🙄")
    print("-----------------------------------------------------------------------------------------------------------------------------------------------")
    
    #crawlinng
    stocks_lst = ["AAPL","AMZN","AVGO","BRK.B","GOOG","JNJ","JPM","LLY","META","MSFT","NVDA","QQQ","TSLA","UNH","V","WMT","XOM","SPY"]
    fx_lst = ["AUDUSD","EURUSD","GBPUSD","USDJPY","XAUUSD"]
    fred_lst = ["T1YFF","SOFR","DCOILBRENTEU","CPFF","BAA10Y"]
    crawler = Crawler(api_key=os.getenv('API_KEY')) # Put your API key in .env file
    for i in stocks_lst:
        crawler.download_historical_data(category="stocks", symbol=f"{i}", location='./data/Stocks')
    for i in fx_lst:
        crawler.download_historical_data(category="fx", symbol=f"{i}", location='./data/FX')
    for i in fred_lst:
        crawler.download_historical_data(category="fred", symbol=f"{i}", location='./data/FRED')
    #print("Lastest data: ", crawler.get_lastest_data(category="crypto", symbol="BTCUSDT"))
    
    crawler.download_historical_data(category="crypto", symbol="BTC", location='./data/Crypto')
    crawler.download_historical_data(category="label", symbol="BTC", location='./data/Label')
    print("sucessful crawling data")
    print("-----------------------------------------------------------------------------------------------------------------------------------------------")
    #preprocessing and getting data
    print("-----------------------------------------------------------------------------------------------------------------------------------------------")
    print("Start to preprocessing data")
    print("Keep goingggg 🫠 🫠")
    with io.capture_output() as captured:

        if current_hour_in_vietnam%6 ==0:
            get_and_update_feature(False)
            
        elif current_hour_in_vietnam%6 !=0:
            get_and_update_feature(True)
    testting = pd.read_csv("/home/ubuntu/nestquant/market_test/realtimeData/realtime_feature.csv")
    
    
    print("....")
    if current_hour_in_vietnam%6 ==0:
        print("Sucessful get preprocessed data with shape of: "+ str(testting.shape))
    
    elif current_hour_in_vietnam%6 !=0:
        print("successful updating data with shape of: "+ str(testting.shape))
    print("-----------------------------------------------------------------------------------------------------------------------------------------------")
    #training

    print("Start to train ...")
    

    path = '/home/ubuntu/nestquant/market_test/realtimeData/realtime_feature.csv'
    df_final = pd.read_csv(path)
    df_final = df_final.dropna()
    dff = df_final.copy()
    dff = dff.iloc[48014:]
    dff = dff.drop("Unnamed: 0", axis = 1)
    time_train = dff.OPEN_TIME.tolist()

    if current_hour_in_vietnam%6 ==0:

        rangee = 960
        delta= 6
        num_boost_round = 90
        model = train(dff,rangee,num_boost_round)
        file = f'/home/ubuntu/nestquant/market_test/model/trained_model.pkl'  ##get name in here
        pickle.dump(model, open(file, 'wb'))
        print("....")

        print("Sucessful trained data with the shape of: "+ str(testting.shape) + " 😋 😗")
        print("-----------------------------------------------------------------------------------------------------------------------------------------------")
    

   
    print("Time to predict and submit my result hehe 🥲 🥲")
    
    data = pd.read_csv("/home/ubuntu/nestquant/market_test/realtimeData/realtime_feature.csv")
    filename = "/home/ubuntu/nestquant/market_test/model/trained_model.pkl"
    model = pickle.load(open(filename, 'rb')) 
    #print(data)
    submit = get_predict(model, data)
    data_set = submit.to_dict('records')
    print(data_set)
    s = Submission(api_key='svx8ZNYrgMNyuithrHdnLEAkn7OzlBKp8h5rzy2e')
    timestamp = s.submit(False, data=data_set, symbol='BTC')
    print("Sucessful submit to the system: 🥳 🥳 ")
    print("Submission time: " + str(timestamp))

    print("-----------------------------------------------------------------------------------------------------------------------------------------------")




