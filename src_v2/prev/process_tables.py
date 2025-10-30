from data import *
from cleaning import *
from report import *
from utility import *
from remove_columns import *
from column_typecheck import *
import sys

def process_all_tables(data_zip_path, data_split, generate_histograms, num_prev_days, 
    forecast_days, handle_scalibility, dataset):
    '''Process all tables'''
    rl_kpis = process_kpi_table(data_zip_path, data_split, generate_histograms, num_prev_days, 
        forecast_days, handle_scalibility, dataset)
    rl_sites = process_rl_sites_table(data_zip_path, data_split, generate_histograms)
    distances = process_distance_table(data_zip_path)
    met_forecast = process_forecast_table(data_zip_path, data_split, forecast_days, 
        generate_histograms, dataset)
    met_stations = process_met_stations_table(data_zip_path, data_split, generate_histograms)
    met_real = process_met_real_table(data_zip_path, data_split, generate_histograms, num_prev_days)
    print('all tables have been read and processed \nprinting out relevant stats of tables: ')
    return rl_kpis, rl_sites, distances, met_forecast, met_stations, met_real

def process_kpi_table(data_zip_path, data_type, generate_histograms, num_prev_days, 
    forecast_days, handle_scalibility, dataset):
    ###### rl-kpis table ######
    # read data for rl-kpis
    rl_kpis = read_table_from_zip(data_zip_path, "rl-kpis")
    #print(f"raw rl_kpis.shape: {rl_kpis.shape}")
    
    # remove unnecessary feature columns
    rl_kpis = kpi_remove_features(rl_kpis, data_type)
    
    # perform typecasting to save computation time and fix incorrect types
    rl_kpis = kpi_feature_typecasting(rl_kpis)
    
    ###### IMPUTATION ######
    # only train feature reports are generated as the test feature
    # reports are generated separately after all test data is combined
    if data_type == 'train':

        if generate_histograms:
            ### generate feature reports ###
            get_kpi_feature_reports(rl_kpis, rlf_separate = True, data_type=data_type)
            ### generate feature distributions ###
            continuous_hist(rl_kpis, table_name='rl_kpis')
            categorical_bar(rl_kpis, table_name='rl_kpis')

            ### generate data distribution for non failure points
            continuous_hist(rl_kpis[rl_kpis['rlf']==False], table_name='non_failure_rl_kpis')
            categorical_bar(rl_kpis[rl_kpis['rlf']==False], table_name='non_failure_rl_kpis')

            ### generate data distribution for failure points
            continuous_hist(rl_kpis[rl_kpis['rlf']==True], table_name='failure_rl_kpis')
            categorical_bar(rl_kpis[rl_kpis['rlf']==True], table_name='failure_rl_kpis')

        # remove features with significant missing values
        # for the new dataset there are missing valeu
        rl_kpis = remove_highly_missing_features(rl_kpis,dataset)
        # impute training data points with missing feature values
        rl_kpis = clean_rl_kpis_table(rl_kpis, method="time_series_interpolation")
        rl_kpis = clean_rl_kpis_table(rl_kpis, method="MMimpute")

    elif data_type == 'test':
        get_input_feature_reports(rl_kpis, data_type = f'{data_type}_rl_kpis')
        # drop test data points with missing feature values 
        rl_kpis = clean_rl_kpis_table(rl_kpis, method="dropna")
        # drop scalability score as train data does not have this feature
        #rl_kpis = rl_kpis.drop(['scalibility_score'], axis=1)
    

    ###### GET RLF COLUMNS ######
    #print(f"cleaned/imputed rl_kpis.shape: {rl_kpis.shape}")
    # get rl-kpis with 1-day-rlf columns and remove the previous rlf column
    rl_kpis = get_rlf_columns(rl_kpis, forecast_days)


    ###### GET PREVIOUS DAYS' KPIS ######
    rl_kpis = get_past_kpis(rl_kpis, num_prev_days) 

    return rl_kpis


def process_rl_sites_table(data_zip_path, data_type, generate_histograms):
    ###### radio link sites table ######
    # read data for rl-sites
    rl_sites = read_table_from_zip(data_zip_path, "rl-sites")

    # perform typecasting to save computation time and fix incorrect types
    rl_sites = rl_sites_feature_typecasting(rl_sites)
    
    if data_type == 'train':
        if generate_histograms:
            # get continuoys and categorical feature reports
            get_site_feature_reports(rl_sites, site_type = 'rl_sites')
            ### generate feature distributions ###
            continuous_hist(rl_sites, table_name='rl_sites')
            categorical_bar(rl_sites, table_name='rl_sites')
        # impute training data points with missing feature values
        rl_sites = clean_sites_table(rl_sites, method="MMimpute")
    elif data_type == 'test':
        # get continuoys and categorical feature reports
        get_input_feature_reports(rl_sites, data_type = f'{data_type}_rl_sites')
        # drop test data points with missing feature values 
        rl_sites = clean_sites_table(rl_sites, method="dropna")
    
    return rl_sites


def process_forecast_table(data_zip_path, data_type, forecast_days, 
    generate_histograms, dataset):
    ###### weather-forecast table ######
    # read weather station forecast data
    met_forecast = read_table_from_zip(data_zip_path, "met-forecast")
    #print(f"raw met-forecast.shape: {met_forecast.shape}")

    # remove the report_time column
    met_forecast = forecast_remove_features(met_forecast)

    # perform typecasting to save computation time and fix incorrect types
    met_forecast = met_forecast_typecasting(met_forecast)

    # aggregate feature values along time to have one record per day
    met_forecast = aggregate_along_time(met_forecast)

    if data_type == 'train':
        if generate_histograms:
            # get continuous and categorical feature reports for weather forecast data
            get_forecast_feature_reports(met_forecast)
            ### generate feature distributions ###
            continuous_hist(met_forecast, table_name='met_forecast')
            categorical_bar(met_forecast, table_name='met_forecast')    
        # clean weather forecast data
        met_forecast = clean_weather_forecast_table(met_forecast, method='time_series_interpolation')

        # remove stations with high percentage of missing values
        if dataset == "new":
            met_forecast = met_forecast[met_forecast['station_no']!='WS_19111']
            met_forecast = met_forecast[met_forecast['station_no']!='WS_17047']
            
    elif data_type == 'test':
        get_input_feature_reports(met_forecast, data_type = f'{data_type}_met_forecast')
        met_forecast = clean_weather_forecast_table(met_forecast,method='dropna')
    
    ###### FILTER ONLY REQUIRED FORECAST DAYS DATA ###### 
    # filter based on number of forecast days
    met_forecast = filter_forecast_days(met_forecast, forecast_days)    
    #print(f"cleaned and imputed met-forecast.shape: {met_forecast.shape}")

    return met_forecast


def process_met_real_table(data_zip_path, data_type, generate_histograms, num_prev_days):
    ###### real weather station data ######
    # read data for met-real
    met_real = read_table_from_zip(data_zip_path, "met-real")

    # remove unnecessary features
    met_real = real_remove_features(met_real)

    # perform typecasting to save computation time and fix incorrect types
    met_real = met_real_typecasting(met_real)

    # aggregate feature values along time to have one record per day
    met_real = aggregate_along_time(met_real)
    # pd.options.display.max_columns = 1000
    # pd.options.display.max_rows = 1000

    if data_type == 'train':
        if generate_histograms:
            # get continuoys and categorical feature reports
            get_input_feature_reports(met_real, 'met-real')
            ### generate feature distributions ###
            continuous_hist(met_real, table_name='met-real')
            categorical_bar(met_real, table_name='met-real')

        # impute training data points with missing feature values
        met_real = clean_met_real_table(met_real, method="time_series_interpolation")
        
    elif data_type == 'test':
        get_input_feature_reports(met_real, data_type = f'{data_type}_met_real')
        # drop test data points with missing feature values 
        met_real = clean_met_real_table(met_real, method="dropna")
    
    # pd.options.display.max_columns = 1000

    ###### GET PREVIOUS DAYS' KPIS ######
    met_real = get_past_ws_real(met_real, num_prev_days)
    
    # print(met_real.info())
    # print(asd)
    return met_real


def process_met_stations_table(data_zip_path, data_type, generate_histograms):
    ###### met stations table ######
    # read data for met stations 
    met_stations = read_table_from_zip(data_zip_path, "met-stations")

    # perform typecasting to save computation time and to keep
    # consistent dtypes after merging
    met_stations = met_stations_typecasting(met_stations)
    
    if data_type == 'train':
        if generate_histograms:
            # get continuoys and categorical feature reports
            get_site_feature_reports(met_stations, site_type = 'met_stations')
            ### generate feature distributions ###
            continuous_hist(met_stations, table_name='met_stations')
            categorical_bar(met_stations, table_name='met_stations') 
        # impute training data points with missing feature values
        met_stations = clean_sites_table(met_stations, method="MMimpute")
    elif data_type == 'test':
        get_input_feature_reports(met_stations, data_type = f'{data_type}_met_stations')
        # drop test data points with missing feature values 
        met_stations = clean_sites_table(met_stations, method="dropna")
    return met_stations


def process_distance_table(data_zip_path):
    ###### distance table ######
    # read data for distances
    distances = read_table_from_zip(data_zip_path, "distances")

    # perform typecasting to save computation time and to keep
    # consistent dtypes after merging
    distances = distances_typecasting(distances)

    #print(f"raw distances.shape: {distances.shape}")
    # clean distances table
    distances = clean_distances_table(distances)
    #print(f"cleaned distances.shape: {distances.shape}")      
    return distances
    

if __name__ == '__main__':
    pass
