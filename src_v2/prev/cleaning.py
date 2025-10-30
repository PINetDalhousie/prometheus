import pandas as pd
from report import calculate_num_outliers
import numpy as np
pd.options.display.max_rows = 1000

def aggregate_along_time(df):
    ''' given a dataframe, this function uses mean and mode aggregation method
    along time dimension to get one record per day
    '''
    # TO DO:
    # create multi-indexed dataframe based on weather station id
    # create multi-indexed dataframe based on each day per weather station
    # perform mean and mode imputation to get one record per day
    # remove all multi-index and return the dataframe

    # calculate mean and mode for each (station_no,datetime) pairs
    df_continuous = df.groupby(['station_no','datetime']).mean()
    df_continuous = df_continuous.reset_index()

    df_categorical = df.select_dtypes(exclude=['number'])

    # if there is categorical features only then aggregate using mode
    if len(df_categorical.columns.tolist()) > 2:
        df_categorical = df_categorical.groupby(['station_no','datetime']).agg(pd.Series.mode)

        # groupby method return multi-level dataframe. Reset index to get standard dataframe
        df_categorical = df_categorical.reset_index()

        # applying mode with groupby creates weather_day feature values of ndarray_numpy
        for column in df_categorical.columns.tolist()[2:]:
            df_categorical[column] = df_categorical[column].transform(lambda x: x[0] if (type(x).__module__ == np.__name__) else x)

        # merge continuous and categorical dataframes on station_no and datetime
        df = df_continuous.merge(df_categorical,
                                how = "inner",
                                left_on = ("station_no","datetime"),
                                right_on = ("station_no","datetime"),
                            )
    else :
        df = df_continuous

    return df

def impute_missing_values(df):
    # instead of dropping, we can use mean and median data imputation
    # get list of columns with missing values
    columns_with_missing_values = df.columns[df.isnull().any()].tolist()
    
    continuous_features = df.select_dtypes('number').columns.to_list()
    categorical_features = df.select_dtypes(include=['object','bool']).columns.to_list()
    
    for miss_column in columns_with_missing_values:
        if miss_column in continuous_features:
            df[miss_column] = df[miss_column].fillna(df[miss_column].median())
        elif miss_column in categorical_features:
            df[miss_column] = df[miss_column].fillna(df[miss_column].mode()[0])

    return df

def time_series_interpolation(df):
    ''' given a dataframe, this function returns linearly interpolated dataframe
    '''
    # TO DO:
    # identify which features are time series features
    # create multi-indexed dataframe based on weather station id
    # for each weather station linearly interpolate the missing values
    # remove multi-index and return the dataframe

    # identify unique station identifier. For radio sites it will be site_id and
    # for weater stations it will be station_no
    if 'site_id' in list(df.columns):
        rl_or_met_site_id = 'site_id'
    elif 'station_no' in list(df.columns):
        rl_or_met_site_id = 'station_no'

    # sort by rl or met site id, so that all the same sites are next to each other 
    df = df.sort_values(by=[rl_or_met_site_id])
    df = df.set_index(rl_or_met_site_id)
    
    # iterate over each station and sort by datetime
    for station_no in df.index.get_level_values(rl_or_met_site_id).unique():
        df.loc[station_no] = df.loc[station_no].sort_values(by=['datetime'])
        # for each continuous feature, perform linear time series interpolation
        for column in df.loc[station_no].select_dtypes(include='number'):
            df.loc[station_no][column].interpolate(method='linear',inplace=True,limit_direction='both')
        
    # reset index to get standard dataframe
    df = df.reset_index()

    return df


def clean_rl_kpis_table(rl_kpis, method):
    if method == "dropna":
        # remove rows with missing values as % of missing instances is very low
        rl_kpis = rl_kpis.dropna()
    elif method == "MMimpute":
        # instead of dropping, we can use mean and median data imputation
        rl_kpis = impute_missing_values(rl_kpis)
    elif method == "time_series_interpolation":
        # for time series features use linear interpolation for continuous
        # and mode imputation for categorical missing value features 
        rl_kpis = time_series_interpolation(rl_kpis)
    return rl_kpis

def clean_sites_table(sites, method):
    if method == "dropna":
        # remove rows with missing values as % of missing instances is very low
        sites = sites.dropna()
    elif method == "MMimpute":
        # instead of dropping, we can use mean and median data imputation
        sites = impute_missing_values(sites) 
    return sites


def clean_distances_table(distances):
    # remove all rows with weather station index
    filter = distances.index.str.contains("^WS")
    distances = distances[~filter]

    # remove all columns with radio site name
    distances = distances[distances.columns.drop(list(distances.filter(regex="^RL")))]
    
    return distances

def clean_weather_forecast_table(met_forecast, method):
    # data imputation
    if method == "dropna":
        # remove rows with missing values
        met_forecast = met_forecast.dropna()
    elif method == "MMimpute":
        # instead of dropping, we can use mean and median data imputation
        met_forecast = impute_missing_values(met_forecast)
    elif method == "time_series_interpolation":
        # for time series features use linear interpolation for continuous
        # and mode imputation for categorical missing value features 
        met_forecast = time_series_interpolation(met_forecast)
    return met_forecast

def clean_met_real_table(met_real, method):
    # data imputation
    if method == "dropna":
        # remove rows with missing values
        met_real = met_real.dropna()
    elif method == "MMimpute":
        # instead of dropping, we can use mean and median data imputation
        met_real = impute_missing_values(met_real)
    elif method == "time_series_interpolation":
        # for time series features use linear interpolation for continuous
        # and mode imputation for categorical missing value features 
        met_real = time_series_interpolation(met_real)
        # time series interpolation doest not interpolate all nan values 
        # so we remove the nan rows
        met_real = met_real.dropna()
    return met_real

def clean_final_merged_table(rl_kpis, features_to_drop, prev_days, model, keep_features,
        keep_feature_method):
    # remove rows with NaN
    rl_kpis = rl_kpis.dropna()

    # drop unnecessary columns that were previously used as identifiers
    if model=='LSTM_AE':
        drop_identifiers = ['mlid','mw_connection_no','site_id','RL_Sites', 'closest_WS',
            'station_no']
    else :
        drop_identifiers = ['mlid','mw_connection_no','site_id','RL_Sites', 'closest_WS',
            'station_no']


    for column in drop_identifiers:
        if column in rl_kpis.columns:
            rl_kpis = rl_kpis.drop([column], axis=1)

    # remove predefined features to perform experiments
    for i in range(1, prev_days):
        for feature in features_to_drop:
            if f'{feature}' in list(rl_kpis.columns):
                rl_kpis = rl_kpis.drop([f'{feature}'], axis=1)
            if f'{feature}_T-{i}' in list(rl_kpis.columns):
                rl_kpis = rl_kpis.drop([f'{feature}_T-{i}'], axis=1)


    # keep relevant and most important features only to increase features incrementally
    ########### KEEP FEATURES NEED TO HAVE DIFFERENT METHODS LIKE KEEP ALL OR KEEP ONLY CONTINUOS

    if keep_features:
        if keep_feature_method == "kpi_forecast":
            feature_names = ["severaly_error_second","error_second","unavail_second","avail_time","bbe","rxlevmax"]
            keep_columns = ["height","clutter_class_forecast",'temp_max_day1','temp_min_day1','humidity_max_day1','humidity_min_day1',
                'wind_dir_day1','wind_speed_day1',"min_distance",'weather_day1',"groundheight","capacity","type","tip",
                "adaptive_modulation","card_type","freq_band","modulation","clutter_class",'1-day-predict']
            for feature in feature_names:
                keep_columns.append(f"{feature}")
                for i in range(1, prev_days):
                    keep_columns.append(f"{feature}_T-{i}")
            rl_kpis = rl_kpis[keep_columns]
        elif keep_feature_method == "kpi_real_continuous":
            # only keep number features because we want to work with kpi and real weather features that
            # are continuous
            rlf_column = rl_kpis['1-day-predict']
            rl_kpis = rl_kpis.select_dtypes(include=['number','datetime'])
            rl_kpis['1-day-predict'] = rlf_column

    return rl_kpis
    
def clean_kpi_label_table(rl_kpis):
    # some kpi dates don't have corresponding next day data point
    # all rows with missing 1 day rlf values were dropped
        # remove rlf column as it's not gonna be of use anymore
    rl_kpis = rl_kpis.drop(['rlf'],axis=1)
    
    return rl_kpis

def remove_outliers(df):
    '''remove outliers of continuous features
    '''
    cont_features = df.select_dtypes('number')
    for feature in cont_features:
        upper_limit = df[feature].mean() + 3 * df[feature].std()
        lower_limit = df[feature].mean() -3 * df[feature].std()
        df = df[(df[feature] < upper_limit) & (df[feature] > lower_limit)]
    return df


def remove_outliers(df, mode, num_times):

    if mode == 'non_failure':
        '''remove outliers of non failure continuoys features only
        '''
        cont_features = df.select_dtypes('number')
        for i in range(num_times):
            for feature in cont_features:
                upper_limit = df[df['1-day-predict']==False][feature].mean() + 3 * df[df['1-day-predict']==False][feature].std()
                lower_limit = df[df['1-day-predict']==False][feature].mean() - 3 * df[df['1-day-predict']==False][feature].std()
                rows_to_drop = df[df['1-day-predict']==False][~((df[df['1-day-predict']==False][feature] < upper_limit) & (df[df['1-day-predict']==False][feature] > lower_limit))]
                df = df.drop(rows_to_drop.index.values.tolist())
            #print(f"shape after outlier removal step {i} : {df.shape}")
            #calculate_num_outliers(df[df['1-day-predict']==False], table_name=f'after_{i}')
    elif mode == 'all':
        '''remove outliers of all continuoys features 
        '''
        cont_features = df.select_dtypes('number')
        for i in range(num_times):
            for feature in cont_features:
                upper_limit = df[feature].mean() + 3 * df[feature].std()
                lower_limit = df[feature].mean() - 3 * df[feature].std()
                rows_to_drop = df[~((df[feature] < upper_limit) & (df[feature] > lower_limit))]
                df = df.drop(rows_to_drop.index.values.tolist())
        
    return df

def remove_highly_missing_features(df, dataset):
    # remove features with significant missing values
    # in future you can automate this step to remove features with certain percentage of missing values
    # you can also explicitly mention here in comments, about the number of missing values
    if dataset == 'prev':
        feature_list = ['direction','polarization','neid','link_length','scalibility_score']
        for feature in feature_list:
            if feature in df.columns.tolist():
                df = df.drop([feature], axis=1)
        print(f'features with high missing values were dropped for {dataset} dataset')
    
    # feature_list = ['direction','polarization','neid','link_length','scalibility_score']
    # for feature in feature_list:
    #     if feature in df.columns.tolist():
    #         df = df.drop([feature], axis=1)
    # print(f'features with high missing values were dropped for {dataset} dataset')
    
    return df




if __name__ == '__main__':
    pass
