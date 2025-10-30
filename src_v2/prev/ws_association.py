import pandas as pd
import numpy as np

def get_closest_ws(distances, optimal_distance, met_forecast, met_real, method, num_neighbors, use_ws_list):
    ''' associate each radio site with the appropriate weather stations based on 
    the method provided. There are actually two steps in associating weather stations.
    The first step involves finding out which weather stations should be considered for
    each radio site. Second, given the set of weather stations for each radio site, what
    method should be used to aggregate the information from neighboring weather stations.
    In this function we implement the first step. 
    There are three ways we can implement 

    Args:
        method: can be either of "closest","optimal","k_nearest"
            "closest", associates rl with closest ws
            "optimal", associates rl with ws that fall within an optimal distance
            "k_nearest", associates rl with k nearest ws
        use_ws_list: in order to consider ws for radio sites, only the ws with available
            real or forecast daily data should be used.
            "real", makes use of unique ws list from met_real table to check while associating rl to ws
            "forecast", makes use of unique ws list from met_forecast table to check while associating rl to ws
    '''
    if method == "closest":
        # for each rl-site get ws id with smallest distance
        min_distance_column = distances.min(axis=1)
        distances['closest_WS'] = distances.idxmin(axis=1)
        distances['min_distance'] = min_distance_column

        # get the necessary columns only
        distances['RL_Sites'] = distances.index
        distances = distances[["RL_Sites","closest_WS","min_distance"]]
        
        # ADD a step where the closest ws is only considered if it's
        # present in the ws table that are going to be used. So, if 
        # the forecast table is gonna be used then we need to make sure that
        # all the associated weather stations are present in the ws forecast
        # table. If there are some that is missing then we should consider 
        # the next closest weather station.

    elif method == "optimal":
        # check which weather stations are within an optimal radius
        # and consider them as associated with the radio site at the 
        # center of the circle
        rl_sites = []
        ws_sites = []
        rl2ws_distances = []
        
        # iterate over rows
        for i in range(len(distances)):
            for j in range(len(distances.columns)):
                # check if distance is within optimal distance
                if distances.iloc[i,j] <= optimal_distance:
                    rl_sites.append(distances.index[i])
                    ws_sites.append(distances.columns[j])
                    rl2ws_distances.append(distances.iloc[i,j])
        
        distances = pd.DataFrame(list(zip(rl_sites, ws_sites, rl2ws_distances)),
                                columns = ['RL_Sites','closest_WS','min_distance'])
    elif method == "k_nearest":
        if use_ws_list=="real":
            met_stations = list(met_real['station_no'].unique())
        elif use_ws_list=="forecast":
            met_stations = list(met_forecast['station_no'].unique())

        rl_sites = []
        ws_sites = []
        rl2ws_distances = []
        
        for i in range(distances.shape[0]):
            num_ws_associated = 0
            sorted_row = pd.DataFrame(distances.iloc[i,:]).sort_values(by=distances.index[i])
            for j in range(distances.shape[1]):
                if num_ws_associated < num_neighbors:
                    if sorted_row.index[j] in met_stations:
                        rl_sites.append(distances.index[i])
                        ws_sites.append(sorted_row.index[j])
                        rl2ws_distances.append(sorted_row.iloc[j,0])
                        num_ws_associated += 1
            
        distances = pd.DataFrame(list(zip(rl_sites, ws_sites, rl2ws_distances)),
                                columns = ['RL_Sites','closest_WS','min_distance'])
                
    return distances
    




if __name__ == '__main__':
    
    pass 