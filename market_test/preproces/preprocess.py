import numpy as np
import pandas as pd
import lightgbm as lgb
#from preproces.utils import *
import io
from datetime import datetime


    
#vwap 
def get_vwap(data,n, name):   #n -> 9-> 20 ( usually 14)
    vwap = data.copy()
    vwap[f'VWAP_{name}_{n}'] = vwap[f'CLOSE_{name}']*vwap[f'VOLUME_{name}'].rolling(n).sum()/vwap[f'VOLUME_{name}'].rolling(n).sum()
    return vwap

#IBS
def get_ibs(data, name):
    ibs = data.copy()
    ibs[f'IBS_{name}'] = (ibs[f'CLOSE_{name}']-ibs[f'LOW_{name}'])/(ibs[f'HIGH_{name}']-ibs[f'LOW_{name}'])
    return ibs

#RSI
def get_rsi(data, n, name):
    rsi = data.copy()
    delta = rsi[f'CLOSE_{name}'] - rsi[f'CLOSE_{name}'].shift(1)
    
    # Calculate delta_up and delta_down with if-elif statements
    delta_up = delta.where(delta > 0, 0)
    delta_down = (-delta).where(delta < 0, 0)
    
    # Calculate the average up and down
    avg_up = delta_up.rolling(n).mean()
    avg_down = delta_down.rolling(n).mean()
    
    # Calculate the relative strength (RS) and RSI
    rs = avg_up / avg_down
    rsi[f'RSI_{name}_{n}'] = 100 - 100 / (1 + rs)
    
    return rsi
#RSI càng tăng thì giá càng giảm.

#MACD
def get_macd(data,  signal_window, name,short_window = 12, long_window=26): #long = 26 15-> 30, short=7->15 12, signal =7->10 9
    macd = data.copy()
    # Calculate the short-term EMA (fast line)
    short_ema = macd[f'CLOSE_{name}'].ewm(span=short_window, min_periods=1, adjust=False).mean()
    
    # Calculate the long-term EMA (slow line)
    long_ema = macd[f'CLOSE_{name}'].ewm(span=long_window, min_periods=1, adjust=False).mean()
    
    # Calculate the MACD line
    macd_line = short_ema - long_ema
    
    # Calculate the signal line (EMA of the MACD line)
    signal_line = macd_line.ewm(span=signal_window, min_periods=1, adjust=False).mean()
    
    # Calculate the MACD histogram
    macd[f'MACD_histogram_{name}_{signal_window}'] = macd_line - signal_line
    
    return macd

# def get_alpha_beta_residual(df, n,name):
#     dff = df.copy()
#     # Rolling beta
#     dff[f'Beta_{name}'] = dff[f'RETURN_{name}_1'].rolling(n).cov(dff['RETURN_SPY_1']) / dff['RETURN_SPY_1'].rolling(n).var()

#     # Rolling alpha
#     dff[f'Alpha_{name}'] = dff[f'RETURN_{name}_1'].rolling(n).mean() - dff[f'Beta_{name}'] * dff['RETURN_SPY_1'].rolling(n).mean()

#     # Residual
#     dff[f'RESIDUAL_{name}'] = dff[f'RETURN_{name}_1'] - (dff[f'Alpha_{name}'] + dff[f'Beta_{name}'] * dff['RETURN_SPY_1'])



def get_return(df, time, name):
    dff = df.copy()
    dff[f'RETURN_{name}_{time}'] = np.log(dff[f'CLOSE_{name}'] / dff[f'CLOSE_{name}'].shift(time))
    return dff

def get_vola(df, time, isReturn, name):
    dff = df.copy()
    if isReturn == False:
        dff = get_return(dff, 1)
    dff[f"std_{time}"] = dff[f'RETURN_{name}_1'].rolling(window=time).std()
    dff["std_long"] = dff[f'RETURN_{name}_1'].rolling(window=336).std()
    dff[f"VOLAT_{name}_{time}"]  = dff[f"std_{time}"]/ dff["std_long"]

    dff = dff.drop(f"std_{time}", axis = 1)
    dff = dff.drop("std_long", axis = 1)

    return dff

def preprocess_df(df, name):
    
    
    for col in df.columns:
        if (col == 'YEAR_AND_MONTH') | (col =="SYMBOL") |(col =="QUOTE_ASSET_VOLUME") |(col =="NUMBER_OF_TRADES") |(col =="TAKER_BUY_BASE_ASSET_VOLUME") |(col =="TAKER_BUY_QUOTE_ASSET_VOLUME"):
            df = df.drop(col, axis = 1)
        else:
            df[f'{col}_{name}'] = df[col]
            df = df.drop(col, axis = 1)

    close = df[f'CLOSE_{name}']
    opn = df[f'OPEN_{name}']
    high = df[f'HIGH_{name}']
    low = df[f'LOW_{name}']
    bar = (opn-close)/(high-low)
    condition1 = opn> close
    condition2 = opn < close
    a = (condition1 * bar).rolling(20).sum() / (condition2 * -bar).rolling(20).sum()
    b = (condition1 * bar).rolling(250).sum() / (condition2 * -bar).rolling(250).sum()
    reversal = a/b
    df[f"REVERSAL_{name}"] = reversal
    df[f'HLC_{name}'] = (close-low)/(high-low)
    df[f'HLOC_{name}'] = (close-opn)/(high-low)
    for i in range(1, 24):
        df = get_return(df, i, name)
        
        
        if i == 1:
            # if name== "BTC":
            #     df = get_alpha_beta_residual(df,i,name)
            df = get_ibs(df, name)
            continue
        if i != 1:
            df = get_vola(df, i, True, name)
        if (i>= 7) &(i<=10):
            df = get_macd(df, i,name) 
        elif (i>= 9) &(i<=20):
            df = get_vwap(df, i, name)
            df = get_rsi(df, i, name)
 

    for col in df.columns:
        if (col =="LABEL_BTC"):
            continue
        
        if (col ==f"LOW_{name}") | (col ==f"HIGH_{name}") | (col ==f"OPEN_{name}"):
            df = df.drop(col, axis = 1)
            continue
        for i in range(1,24):
            df[f"{col}_shift_{i}"] = df[f"{col}"].shift(i)

    return df





# def get_internal(df,name): #input n1, n2 to calculate return and volatility
#     #denote data
#     for col in df.columns:
#         if (col == 'YEAR_AND_MONTH') | (col =="SYMBOL") |(col =="QUOTE_ASSET_VOLUME") |(col =="NUMBER_OF_TRADES") |(col =="TAKER_BUY_BASE_ASSET_VOLUME") |(col =="TAKER_BUY_QUOTE_ASSET_VOLUME"):
#             df = df.drop(col, axis = 1)
#         else:
#             df[f'{col}_{name}'] = df[col]
#             df = df.drop(col, axis = 1)
            
#     dff = df.copy()

#     close = dff[f'CLOSE_{name}']
#     opn = dff[f'OPEN_{name}']
#     volume = dff[f'VOLUME_{name}']
#     high = dff[f'HIGH_{name}']
#     low = dff[f'LOW_{name}']
#     #return volatility
#     ret = np.log(close/close.shift(1))
#     #volat = ret.ewm(span = n2).std()
    
#     #diff return
#     diff_ret = ret - ret.rolling(30).mean()
#     zscore_ret = diff_ret/(ret.rolling(30).std())

#     dff[f'RET_{name}_1'] = ret

#     dff[f"std_1"] = dff[f'RET_{name}_1'].rolling(window=1).std()
#     dff["std_long"] = dff[f'RET_{name}_1'].rolling(window=168).std()
#     #dff[f"VOLAT_{name}_1"]  = dff[f"std_1"]/ dff["std_long"]

#     dff = dff.drop(f"std_1", axis = 1)
#     dff = dff.drop("std_long", axis = 1)


#     #dff[f'VolatAdjustRet_{name}'] = dff[f'RET_{name}_1']/dff[f'VOLAT_{name}_1']
#     dff[f'DiffRet_{name}'] = diff_ret
#     dff[f'ZscoreRet_{name}'] = zscore_ret


#     for i in range(2, 24):
#         ret_i = np.log(close/close.shift(i))
        
#         diff_ret_i = ret_i - ret_i.rolling(30).mean()
#         zscore_ret_i = diff_ret/(ret_i.rolling(30).std())

#         dff[f'RET_{name}_{i}'] = ret_i

#         dff[f"std_{i}"] = dff[f'RET_{name}_1'].rolling(window=i).std()
#         dff["std_long"] = dff[f'RET_{name}_1'].rolling(window=336).std()
#         dff[f"VOLAT_{name}_{i}"]  = dff[f"std_{i}"]/ dff["std_long"]
        
#         if i == 2:
#             dff[f'VolatAdjustRet_{name}_2'] = dff[f'RET_{name}_2']/dff[f'VOLAT_{name}_2']
        
        
#         dff = dff.drop(f"std_{i}", axis = 1)
#         dff = dff.drop("std_long", axis = 1)



#     #open gap
#     open_gap = (opn - close.shift(1))/close.shift(1)
    
#     #close_volume_adjusted
#     cl_vol = (close - close.shift(2))*(volume / volume.rolling(10).mean())
    
#     #bar
#     bar = (opn-close)/(high-low)
#     condition1 = opn> close
#     condition2 = opn < close
#     a = (condition1 * bar).rolling(20).sum() / (condition2 * -bar).rolling(20).sum()
#     b = (condition1 * bar).rolling(250).sum() / (condition2 * -bar).rolling(250).sum()
#     reversal = a/b
    
#     #reverse
#     reverse = close/(close.rolling(window=5).agg(lambda x : x.prod())**0.2)
    
#     #close/vwap - 1
#     typ = (high+low+close)/3
#     typ_vol = typ*volume
#     vwap = (typ_vol.rolling(14).sum())/(volume.rolling(14).sum())
#     cl_vwap = close/vwap - 1
    
    
#     #RSI
#     delta = close - close.shift(1)
    
#     # Calculate delta_up and delta_down with if-elif statements
#     delta_up = delta.where(delta > 0, 0)
#     delta_down = (-delta).where(delta < 0, 0)
    
#     # Calculate the average up and down
#     avg_up = delta_up.rolling(14).mean()
#     avg_down = delta_down.rolling(14).mean()
    
#     # Calculate the relative strength (RS) and RSI
#     rs = avg_up / avg_down
#     rsi = 100 - 100/(1 + rs)
    
#     #MACD
#     # Calculate the long-term EMA (slow line)
#     ema = close.ewm(span=9).mean()
    
#     # Calculate the MACD line
#     macd_line = ema - vwap
    
#     # Calculate the signal line (EMA of the MACD line)
#     signal_line = macd_line.ewm(span=9).mean()
    
#     # Calculate the MACD histogram
#     macd = macd_line - signal_line
    
    
#     dff[f'OpenGap_{name}'] = open_gap
#     dff[f'ClVol_{name}'] = cl_vol
#     dff[f'REVERSAl_{name}'] = reversal
#     dff[f'REVERSE_{name}'] = reverse
#     #dff[f'MOMEN_{name}'] = momen
#     dff[f'RSI_{name}'] = rsi
#     dff[f'MACD_{name}'] = macd
    
#     dff[f'HLC_{name}'] = (close-low)/(high-low)
#     dff[f'HLOC_{name}'] = (close-opn)/(high-low)
#     dff[f'VOL_{name}'] = volume/volume.rolling(20).mean()



#     ad = np.where(
#             (close == high) & (close == low) | (high == low),
#             0,
#             ((2 * close - low - high) / (high - low)) * volume
#         )
#     mf = np.sum(ad, axis=0) / np.sum(volume, axis=0)
#     dff[f'MF_{name}'] = mf

    
#     for col in dff.columns:
#         if (col =="LABEL_BTC"):
#             continue

#         if (col ==f"LOW_{name}") | (col ==f"HIGH_{name}") | (col ==f"OPEN_{name}"):
#             dff = dff.drop(col, axis = 1)
#             continue
#         for i in range(1,24):
#             dff[f"{col}_shift_{i}"] = dff[f"{col}"].shift(i)
    
        
#     return dff


# def get_diff(df, name1, name2): #name1 = 'BTC', name2 = 'SPY'
#     dff = df.copy() 
#     dff['RET2MKT']           = dff[f'RET_{name1}_1']           - dff[f'RET_{name2}_1']
#     dff['VOLAT2MKT']         = dff[f'VOLAT_{name1}_2']         - dff[f'VOLAT_{name2}_2']
#     #dff['VolatAdjustRet'] = dff[f'VolatAdjustRet_{name1}_2'] - dff[f'VolatAdjustRet_{name2}_2']
#     dff['DiffRet2MKT']       = dff[f'DiffRet_{name1}']      - dff[f'DiffRet_{name2}']
#     dff['ZscoreRet2MKT']     = dff[f'ZscoreRet_{name1}']        - dff[f'ZscoreRet_{name2}']
#     dff['OpenGap2MKT']       = dff[f'OpenGap_{name1}']      - dff[f'OpenGap_{name2}']
#     dff['ClVol2MKT']        = dff[f'ClVol_{name1}']        - dff[f'ClVol_{name2}']
#     dff['REVERSAl2MKT']      = dff[f'REVERSAl_{name1}']      - dff[f'REVERSAl_{name2}']
#     dff['REVERSE2MKT']       = dff[f'REVERSE_{name1}']       - dff[f'REVERSE_{name2}']
#     dff['ClVwap2MKT']       = dff[f'ClVwap_{name1}']       - dff[f'ClVwap_{name2}']
#     dff['RSI2MKT']           = dff[f'RSI_{name1}']           - dff[f'RSI_{name2}']
#     dff['MACD2MKT']          = dff[f'MACD_{name1}']           - dff[f'MACD_{name2}']
#     dff['HLC2MKT']           = dff[f'HLC_{name1}']           - dff[f'HLC_{name2}']
#     dff['HLOC2MKT']           = dff[f'HLOC_{name1}']           - dff[f'HLOC_{name2}']
#     dff['VOL2MKT']           = dff[f'VOL_{name1}']           - dff[f'VOL_{name2}']
#     dff['MF2MKT'] = df[f'MF_{name1}'] - df[f'MF_{name2}']
#     #n = 168
#     #dff['ROLLINGBeta'] = dff[f'RET_{name1}_1'].rolling(n).cov(dff[f'RET_{name2}_1']) / dff[f'RET_{name2}_1'].rolling(n).var()

#     # Calculate rolling alpha
#     #dff['ROLLINGAlpha'] = dff[f'RET_{name1}_1'].rolling(n).mean() - dff['ROLLINGBeta'] * dff[f'RET_{name2}_1'].rolling(n).mean()

#     # Calculate residual (difference between actual and predicted returns)
#     #dff['RESIDUAL'] = dff[f'RET_{name1}_1'] - (dff['ROLLINGAlpha'] + dff['ROLLINGBeta'] * dff[f'RET_{name2}_1'])
    
    
    
#     return dff


# Original dataframe
def get_dupp(dff):
# Convert the 'date' column to datetime type
    dff['DATE'] = pd.to_datetime(dff['DATE'])

    # Create a new dataframe to store the expanded rows
    expanded_dff = pd.DataFrame()

    # Iterate over each row in the original dataframe
    for index, row in dff.iterrows():
        date = row['DATE']
        values = row[1:]  # Exclude the 'date' column

        # Create a datetime range for the 24 hours of the day
        hour_range = pd.date_range(date, periods=24, freq='H')

        # Create a temporary dataframe with the expanded rows
        temp_dff = pd.DataFrame(hour_range, columns=['DATE'])

        # Duplicate the values for each hour
        for column, value in zip(dff.columns[1:], values):
            temp_dff[column] = value

        # Append the temporary dataframe to the expanded dataframe
        expanded_dff = pd.concat([expanded_dff,temp_dff], ignore_index=True)
    return expanded_dff




#Merge external data
def join_data(df_lst):
    global df_BTCLabel
    df_final = df_BTCLabel.copy()
    for i in df_lst:
        df_final = df_final.join(i, how = 'left').bfill().ffill()
    
    df_final = df_final.reset_index()
    
    return df_final
       





def get_dailyToHours(df):
    df = get_dupp(df)
    df["OPEN_TIME"] = df["DATE"].apply(lambda x: int(datetime.timestamp(x)) *1000)
    df = df.drop("DATE", axis= 1)
    df = df.set_index("OPEN_TIME")
    return df
