import numpy as np
import pandas as pd


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
    dff[f"VOLABILITY_{name}_{time}"]  = dff[f"std_{time}"]/ dff["std_long"]

    dff = dff.drop(f"std_{time}", axis = 1)
    dff = dff.drop("std_long", axis = 1)

    return dff


def preprocess_df(df, name, start, end):
    for col in df.columns:
        if (col == 'YEAR_AND_MONTH') | (col =="SYMBOL"):
            df = df.drop(col, axis = 1)
        else:
            df[f'{col}_{name}'] = df[col]
            df = df.drop(col, axis = 1)

    for i in range(start, end):
        df = get_return(df, i, name)
        if i == 1:
            continue
        else:
            df = get_vola(df, i, True, name)

    return df


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
        expanded_dff = expanded_dff.append(temp_dff, ignore_index=True)
    return expanded_dff


def check_cor(df, rangee, delta, time_lst, target_feature):
    corr_tuple = tuple()



    for i in range(rangee, len(time_lst)- delta, delta):
        df_tmp = df[(df.OPEN_TIME >= time_lst[i-rangee])&(df.OPEN_TIME < time_lst[i-delta])]
        correlation_values = df_tmp.corrwith(df_tmp[target_feature]).drop(target_feature)

        correlation_array = correlation_values.values
        contains_nan = np.isnan(correlation_array).any()
        if contains_nan:
            continue
        corr_tuple+= (correlation_array,)

    # Vertically stack the arrays
    stacked_array = np.vstack(corr_tuple)

    # Calculate the median array
    median_array = np.median(stacked_array, axis=0)
    df_corr = pd.DataFrame(correlation_values).reset_index()
    df_corr.drop(0, axis=1)
    df_corr["median_corr"] = median_array.flatten()
    df_corr["absolute_corr"] = df_corr["median_corr"].apply(abs)
    df_corr = df_corr.sort_values("absolute_corr", ascending=False)
    return df_corr