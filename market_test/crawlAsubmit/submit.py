import os
import sys
import time
sys.path.append(os.path.dirname (os .getcwd ( ) ) )

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from src.submit import Submission



def get_submit(data):
    s = Submission(api_key='svx8ZNYrgMNyuithrHdnLEAkn7OzlBKp8h5rzy2e')
    data_set = data.to_dict('records')
    timestamp = s.submit(True, data=data_set, symbol='BTC')

    all_rec = s.get_submission_time(is_backtest=False, symbol='BTC')['BTCUSDT']






