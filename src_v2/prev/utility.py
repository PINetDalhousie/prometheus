import pandas as pd
from cleaning import *
from preprocessing import *
from report import *
from balancing import *
from data import *
from sklearn.decomposition import PCA



def get_past_kpis(rl_kpis, num_prev_days):
    ''' 
    '''
    # get unique entry identifier
    df_unique = rl_kpis.copy(deep=True)
    
    # pandas column of datetime type is offset by number of previous day's features
    # to get previous day datetimes
    for i in range(num_prev_days-1):
        df_unique.loc[:, f"T-{i+1}"] = df_unique["datetime"] + pd.DateOffset(days=i-1)
   
    # merge existing df that contains previous n days columns with 
    # df that has unique identifier and other features
    for i in range(1, num_prev_days):
        target_day_column_name = f"T-{i}"
        
        df_unique = df_unique.merge(rl_kpis, 
                      how = "inner", 
                      left_on = ("site_id", "mlid", target_day_column_name),
                      right_on = ("site_id", "mlid", "datetime"),
                      suffixes = ("", f"_T-{i}")
        )
    
        # remove features irrelevent to previous days' kpis
        remove_features = ['T',"datetime_T","mw_connection_no_T","neid_T","1-day-predict_T","type_T",
            "tip_T","card_type_T","adaptive_modulation_T","freq_band_T","modulation_T"]
        for feature in remove_features:
            if f"{feature}-{i}" in df_unique.columns:
                df_unique = df_unique.drop([f"{feature}-{i}"], axis=1)
       
    return df_unique

def get_past_ws_real(met_real, num_prev_days):
    # get unique entry identifier
    df_unique = met_real.copy(deep=True)
    
    # pandas column of datetime type is offset by number of previous day's features
    # to get previous day datetimes
    for i in range(num_prev_days-1):
        df_unique.loc[:, f"T-{i+1}"] = df_unique["datetime"] + pd.DateOffset(days=i-1)

    # merge existing df that contains previous n days columns with 
    # df that has unique identifier and other features
    for i in range(1, num_prev_days):
        target_day_column_name = f"T-{i}"
        
        df_unique = df_unique.merge(met_real, 
                      how = "inner", 
                      left_on = ("station_no", target_day_column_name),
                      right_on = ("station_no", "datetime"),
                      suffixes = ("", f"_T-{i}")
        )
    
        # remove features irrelevent to previous days' kpis
        remove_features = ['T',"datetime_T"]
        for feature in remove_features:
            if f"{feature}-{i}" in df_unique.columns:
                df_unique = df_unique.drop([f"{feature}-{i}"], axis=1)

    return df_unique

def get_previous_days_features(rl_kpis, num_prev_days):
    # get unique entry identifier
    df_unique = rl_kpis.copy(deep=False)
    
    # pandas column of datetime type is offset by number of previous day's features
    # to get previous day datetimes
    for i in range(num_prev_days-1):
        df_unique.loc[:, f"T-{i+1}"] = df_unique["datetime"] + pd.DateOffset(days=i-1)
    
    # merge existing df that contains previous n days columns with 
    # df that has unique identifier and other features
    for i in range(num_prev_days-1):
        target_day_column_name = f"T-{i+1}"
        
        df_unique = df_unique.merge(rl_kpis, 
                      how = "inner", 
                      left_on = ("site_id", "mlid", target_day_column_name, "closest_WS", "station_no"),
                      right_on = ("site_id", "mlid", "datetime", "closest_WS", "station_no"),
                      suffixes = ("", f"_T-{i+1}")
        )
        
        # remove redundant columns from merging with previous days features
        df_unique.drop(columns=[f"T-{i+1}",f"datetime_T-{i+1}",f"mw_connection_no_T-{i+1}",
            f"1-day-predict_T-{i+1}",f"groundheight_T-{i+1}",f"clutter_class_T-{i+1}",
            f"RL_Sites_T-{i+1}",f"min_distance_T-{i+1}", f"height_T-{i+1}", 
            f"clutter_class_forecast_T-{i+1}"], inplace=True)
        
        # remove scalibility_score feature if present
        if f"scalibility_score_T-{i+1}" in list(df_unique.columns):
            df_unique = df_unique.drop([f"scalibility_score_T-{i+1}"], axis=1)

    return df_unique



def filter_forecast_days(met_forecast, forecast_days):
    #met_forecast_filtered = met_forecast.copy(deep=False)

    # name of weather forecast features
    forecast_features = ['weather_day', 'temp_max_day', 'temp_min_day',
                        'humidity_max_day', 'humidity_min_day', 'wind_dir_day',
                        'wind_speed_day']

    # loop over forecast features and filter according to forecast_days number of days
    for day in range(forecast_days+1,6):
        met_forecast = met_forecast.drop([x+str(day) for x in forecast_features], axis=1)
    return met_forecast




def apply_pca(X_train, num_components):
    if num_components == 0:
        pca = PCA()
    else:
        pca = PCA(n_components = num_components)
    X_train = pca.fit_transform(X_train)
    return X_train, pca
    

def align_test_features(X_test, train_features):
    for feature in list(X_test.columns):
        if feature not in train_features:
            X_test = X_test.drop([feature], axis=1)
    return X_test
    

def convert_prev_dataset_to_new_format(prev_data):
    df = pd.read_excel("../data/")


def get_rlf_columns(rl_kpis, forecast_days):
    # get unique entry identifier and remove rlf column
    # for rl-kpis datetime,site_id and mlid together uniquely identifies each entry
    #df_labels = rl_kpis[["datetime", "site_id", "mlid"]]
    df_labels = rl_kpis.copy(deep=True)
    df_labels.drop(columns=["rlf"], inplace=True)

    # pandas column of datetime type is offset by 1 day to get 
    # the following 5 days datetime for each entry
    for i in range(1, forecast_days+1):
        df_labels.loc[:, f"T+{i}"] = df_labels["datetime"] + pd.DateOffset(days=i)
        #df_labels[f"T+{i+1}"] = df_labels["datetime"] + pd.DateOffset(days=i+1)
    #df_labels.head()

    # merge existing df that contains following 5 day columns with 
    # df that has unique identifier and rlf (true or false) 
    # e.g left merge t+1,site_id,mlid with datetime,site_id,mlid
    rl_kpis_view = rl_kpis[["datetime", "site_id", "mlid", "rlf"]]
    
    #df_labels.to_csv("../report/df_labels.csv")
    #rl_kpis_view.to_csv("../report/rl_kpis_view.csv")

    # iterate over each day column to get rlf columns for following 5 days
    for i in range(1, forecast_days+1):
        target_day_column_name = f"T+{i}"

        df_labels = df_labels.merge(rl_kpis_view, 
                      how = "inner", 
                      left_on = ("site_id", "mlid", target_day_column_name),
                      right_on = ("site_id", "mlid", "datetime"),
                      suffixes = ("", "_y")
        )
        # rename rlf columns according to target day name
        df_labels.rename(columns={"rlf": f"{i}-day-predict"}, inplace=True)
        # remove extra T+{i} and datetime_y columns that are produced for having overlapping column
        # name with different column values
        df_labels.drop(columns=[f"T+{i}", "datetime_y"], inplace=True)
    
    return df_labels


if __name__ == '__main__':
    
    pass 
