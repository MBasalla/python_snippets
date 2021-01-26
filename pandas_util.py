import pandas as pd


def rolling_series(series,window_size,step_size):
    window_list = []
    series.rolling(window_size).apply(lambda x: window_list.append(x) or 0)
    return window_list[::step_size]


def rolling_dataframes(df,window_size,step_size):
    columns = df.columns
    dict_of_lists = {}
    for col in columns:
        dict_of_lists[col] = rolling_series(df[col],window_size,step_size)
    list_of_dicts = dl_to_ld(dict_of_lists)
    list_of_dfs = list(map(lambda d: pd.DataFrame(d),list_of_dicts))
    return list_of_dfs


def samples_per_season(time_stamps, season_string):
    if season_string == "A" or season_string == "year":
        years_in_data = time_stamps.map(lambda t: t.year)
        return years_in_data.value_counts().median()
        #TODO: test and fix
    elif season_string == "M" or season_string == "month":
        months_in_data = time_stamps.map(lambda t: t.month)
        return months_in_data.value_counts().median()
    elif season_string == "W" or season_string == "week":
        weeks_in_data = time_stamps.map(lambda t: t.week)
        return weeks_in_data.value_counts().median()
    elif season_string == "D" or season_string == "day":
        days_in_data = time_stamps.map(lambda t: t.day)
        return days_in_data.value_counts().median()
    elif season_string == "H" or season_string == "hour":
        hours_in_data = time_stamps.map(lambda t: t.hour)
        return hours_in_data.value_counts().median()
    else:
        KeyError("season string %s is not supported" % season_string)
        

def samples_per_season(frequency, season_string):
    # TODO: figure out a way to use pandas frequncy output to get )= numbre rof samples per season
    # output can ba any of a number of time series offsets tseries.offsets.DateOffset
    # identify data frequency as one of hourly, daily, weekly monthly or quaterly:
    frequency_string = frequency.freqstr()
    if "A" in frequency_string:
        #yearly
        return None
    elif "Q" in frequency_string:
        #quaterly
        return None
    elif "M"  in frequency_string:
        #monthly
        return None
    elif "W" in frequency_string:
        # weekly
        return None
    elif "D"  in frequency_string:
         #daily
         return None
    elif "H" in frequency_string:
         # hourly
         return None


def split_into_real_and_categorical(df):
    categorical_df = df.select_dtypes(include=['object','integer','category','character'])
    real_df = df.select_dtypes(include='inexact')
    return real_df, categorical_df