import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '..'))

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from src.crawl import Crawler


if __name__ == "__main__":
    stocks_lst = ["AAPL","AMZN","AVGO","BRK.B","GOOG","JNJ","JPM","LLY","META","MSFT","NVDA","QQQ","TSLA","UNH","V","WMT","XOM","SPY"]
#,"AMZN","AVGO","BRK.B","GOOG","JNJ","JPM","LLY","META","MSFT","NVDA","QQQ","TSLA","UNH","V","WMT","XOM"
    fx_lst = ["AUDUSD","EURUSD","GBPUSD","USDJPY","XAUUSD"]
#,"C_EURUSD","C_GBPUSD","C_USDJPY","C_XAUUSD"
    fred_lst = ["T1YFF","SOFR","DCOILBRENTEU","CPFF","BAA10Y"]
#,"SOFR","DCOILBRENTEU","CPFF","BAA10Y"

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

    