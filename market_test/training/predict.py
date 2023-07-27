import pandas as pd
import pickle
import numpy as np
from src.submit import Submission

def get_predict(model, data, Labeltime):
    data_test = data[data["OPEN_TIME"] == (Labeltime+25*3600000)]
    df_test = data_test.drop(["LABEL_BTC", "OPEN_TIME","Unnamed: 0"], axis = 1)
    pred=model.predict(df_test)
    print(pred)
    submitt=pd.DataFrame(columns=['OPEN_TIME','PREDICTION'])
    print(data_test["OPEN_TIME"])
    submitt['OPEN_TIME']= np.array(data_test["OPEN_TIME"] + 3600000*2).flatten()
    submitt['PREDICTION']=pred.flatten()
    print(submitt)
    return submitt


if __name__ == "__main__":
    import os
    import sys
    import time
    sys.path.append(os.path.dirname (os .getcwd ( ) ) )
    s = Submission(api_key='svx8ZNYrgMNyuithrHdnLEAkn7OzlBKp8h5rzy2e')
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())


    data = pd.read_csv("/home/ubuntu/nestquant/market_test/realtimeData/realtime_feature.csv")
    filename = "/home/ubuntu/nestquant/market_test/model/trained_model.pkl"
    model = pickle.load(open(filename, 'rb')) 
    #print(data)
    submit = get_predict(model, data)
    data_set = submit.to_dict('records')
    timestamp = s.submit(True, data=data_set, symbol='BTC')
    print(data_set)
