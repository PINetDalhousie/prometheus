from aggregate_ws import aggregate_met_real_features, seperate_rl_met_features
from data import *
from report import *
from cleaning import *
from utility import *
import sys
from process_tables import *
from merge import *
from ws_association import get_closest_ws

def generate_min_available_rl2ws_dist():
    data_zip_path = "../data/train/RegionA.zip"
    distances = process_distance_table(data_zip_path)
    
    met_forecast = process_forecast_table(data_zip_path, "train")
    forecast_stations = met_forecast['station_no'].unique()
    
    available_ws_distances = []
    available_ws_list = []  
    
    # iterate over rows
    for i in range(len(distances)):
        for j in range(20):
            if j == 0:
                current_rlsite_min_distance = distances.iloc[i,j]
                current_rlsite_min_ws = distances.columns[j]
                continue
            
            if distances.iloc[i,j] < current_rlsite_min_distance:
                if distances.columns[j] in forecast_stations:
                    current_rlsite_min_distance = distances.iloc[i,j]
                    current_rlsite_min_ws = distances.columns[j]
        
        available_ws_distances.append(current_rlsite_min_distance)
        available_ws_list.append(current_rlsite_min_ws)
   

    min_distance_column = distances.min(axis=1)
    distances['closest_WS'] = distances.idxmin(axis=1)
    distances['min_distance'] = min_distance_column 
    distances['RL_Sites'] = distances.index
    distances['avail_close_ws'] = available_ws_list 
    distances['avail_close_distance'] = available_ws_distances 

    distances = distances[["RL_Sites","closest_WS","min_distance","avail_close_ws", "avail_close_distance"]]

    distances.to_csv('../report/modified_distances.csv')
   
    

def remove_previous_ws_features(rl_kpis, forecast_days):
    for i in range(1, forecast_days+1):
        for column in (rl_kpis.columns):
            if len(column)>5:
                if column[-5] == str(i):
                    rl_kpis = rl_kpis.drop([column], axis=1)
    return rl_kpis


def get_test_identifiers(test_df):
    ''' get feature columns for inference step
    '''
    identifiers = ['datetime','RL_Sites','mlid']
    identifiers_df = test_df[identifiers]
    return identifiers_df


def get_test_results(data_zip_path, optimal_dist, prev_days, forecast_days, model, features_to_drop, 
    keep_features, dataset):
  
    rl_kpis, rl_sites, distances, met_forecast, met_stations, met_real = process_all_tables(data_zip_path,
        data_split='test', generate_histograms=False, num_prev_days=prev_days,
        forecast_days=forecast_days, handle_scalibility=False, dataset=dataset)

    # print out some relevant stats from the tables that has been read
    print(f"shape of imputed radio link kpi table with rlf columns: {rl_kpis.shape}")
    print(f"stats on radio link failure events: {rl_kpis['1-day-predict'].value_counts()}")
    print(f"number of radio link sites from rl sites table: {rl_sites['site_id'].unique().shape}")
    print(f"number of radio link sites from kpi table: {rl_kpis['site_id'].unique().shape}")
    print(f"number of radio link sites from distance table: {distances.shape[0]}")
    print(f"number of ws sites from distance table: {distances.shape[1]}")
    print(f"number of ws sites from forecast table: {met_forecast['station_no'].unique().shape}")
    print(f"number of sites from ws real table: {met_real['station_no'].unique().shape}")
    print(f"number of ws sites from ws sites table: {met_stations['station_no'].unique().shape}")
    print(f"shape of imputed forecast table: {met_forecast.shape}")

    distances = get_closest_ws(distances, optimal_dist, met_forecast, met_real, method="k_nearest",
        num_neighbors=3, use_ws_list="real")
    print(f"closest ws distnace table shape {distances.shape}")
    rl_kpis = merge_tables(rl_kpis, rl_sites, distances, met_forecast, met_stations, met_real, 
                            include_forecast=False, include_real=True)
    print(f"test after merging shape {rl_kpis.shape}")
    print(f"stats on radio link failure events: {rl_kpis['1-day-predict'].value_counts()}")
    rl_kpis, df_met = seperate_rl_met_features(rl_kpis, prev_days)

    rl_kpis = aggregate_met_real_features(rl_kpis, df_met)
    print(f"test after aggregating met real features shape {rl_kpis.shape}")
    print(f"stats on radio link failure events: {rl_kpis['1-day-predict'].value_counts()}")
    identifiers = get_test_identifiers(rl_kpis)

    rl_kpis = clean_final_merged_table(rl_kpis, features_to_drop, prev_days=prev_days, model=model,
        keep_features=keep_features, keep_feature_method="kpi_real_continuous")
    print(f"test after cleaning shape {rl_kpis.shape}")
    print(f"stats on radio link failure events: {rl_kpis['1-day-predict'].value_counts()}")
    Y_test = rl_kpis["1-day-predict"].astype('int').to_numpy()
    X_test = rl_kpis.drop(['1-day-predict'], axis=1)
    
    return X_test, Y_test, identifiers

if __name__ == '__main__':
    pass
