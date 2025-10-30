import pandas as pd
import numpy as np

def seperate_rl_met_features(df, prev_days):
    ''' df contains both rl features and met real features. for data processing 
    purposes (aggregation of met features) down the line we first seperate these
    two kinds of features. 
    '''
    # met real feature list
    met_real_features = ["min_distance", "temp", "temp_max", "temp_min",
                         "wind_dir", "wind_speed", "humidity",
                         "precipitation", "precipitation_coeff", "pressure", "pressure_sea_level",
                         "height"]

    temporal_features = ["temp", "temp_max", "temp_min", "wind_dir", "wind_speed", "humidity", "precipitation", "precipitation_coeff",
                         "pressure", "pressure_sea_level"]

    # add feature columns from past met real data
    for i in range(1, prev_days):
        for feature in temporal_features:
            met_real_features.append(feature+f"_T-{i}")

    # sepearate kpi features and weather features that need to be aggregated
    df_met = df[met_real_features + ["site_id", "mlid", "datetime"]]
    df = df.drop(met_real_features, axis=1)
    df = df.drop_duplicates(subset=["site_id", "mlid", "datetime"])

    return df, df_met


def aggregate_met_real_features(df, df_met):
    ''' the df dataframe contains contains merged 
    '''
    # merge with distances table
    # df_met = df_met.merge(distances,
    #               how="inner",
    #               left_on=("station_no"),
    #               right_on=("closest_WS"),
    #               suffixes=("", "_distance")
    #               )
    # # remove redundant columns
    # df_met = df_met.drop(['station_no','closest_WS'], axis=1)
    #print(df.info())
    #print(df_met.info())
    
    # AGGREGATE all of them using one function agg([list])  
    df_met = df_met.groupby(["site_id", "mlid", "datetime"]).agg(['mean','min','max','std'])
    df_met.columns = ['_'.join(col) for col in df_met.columns]
    df_met = df_met.reset_index()
    #print(df_met.info())
    # finally merge table with derived features with the original table
    df = df.merge(df_met,
                  how="inner",
                  left_on=("site_id", "mlid", "datetime"),
                  right_on=("site_id", "mlid", "datetime"),
                  suffixes=("", "_met")
                  )
    #pd.set_option('display.max_rows', None)
    #print(f"df info {df.info()}")
    #print(f"df columns {df.columns.tolist()}")
    #print(asd)

    return df


def aggreagate_for_GNN(df, prev_days, k_neighbors):
    ''' given a df with met real weather data where each radio site is associated with k nearest
    weather stations, this function groups rows based on radio site, minilink and date time so that
    each group is uniquely identified with mini link id and the datetime. Each group is then considered
    to get the list of associated weather stations feature values. These values are padded if necessary.
    '''
    # TO DO
    # define feature columns for weather data and unique identifiers
    # append column names of previous day values if needed
    # group dataframe by rlsite,minilink and datetime
    # iterate over each group to get values of all associated ws
    # convert list of values to final dataframe


    # met real feature list
    met_real_features = ["min_distance", "temp", "temp_max", "temp_min",
                         "wind_dir", "wind_speed", "wind_dir_max", "wind_speed_max", "humidity",
                         "precipitation", "precipitation_coeff", "pressure", "pressure_sea_level",
                         "height"]

    temporal_features = ["temp", "temp_max", "temp_min", "wind_dir", "wind_speed", "wind_dir_max",
                         "wind_speed_max", "humidity", "precipitation", "precipitation_coeff",
                         "pressure", "pressure_sea_level"]

    unique_identifiers = ["site_id", "mlid", "datetime"]

    # add feature columns from past met real data
    for i in range(1, prev_days):
        for feature in temporal_features:
            met_real_features.append(feature+f"_T-{i}")

    df_agg = df[met_real_features + unique_identifiers]

    # group dataframe
    grouped = df_agg.groupby(unique_identifiers)

    aggregated_for_gnn = []
    for name, group in grouped:
        # get the list of unique identifiers
        group_identifiers = group[unique_identifiers].head(1).values.tolist()
        # get list of associated ws feature values
        ws_df = group.drop(unique_identifiers, axis=1)
        ws_feature_values = ws_df.to_numpy().flatten().tolist()

        # append to final list
        aggregated_for_gnn.append(group_identifiers+ws_feature_values)

    del grouped
    del df_agg
    del df

    # convert list of values to dataframe
    # define column names of new dataframe
    aggregated_feature_columns = unique_identifiers
    for neighbor in (1,k_neighbors+1):
        for feature in met_real_features:
            aggregated_feature_columns.append(feature+f"_ws{neighbor}")
    df = pd.DataFrame(aggregated_for_gnn, columns = aggregated_feature_columns)

    return df

if __name__ == '__main__':
    pass
