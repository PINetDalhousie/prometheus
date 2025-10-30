import pandas as pd
import numpy as np
from table import DistancesTable,MetrealTable,RlkpisTable
from deployment import Deployment

class AssignWS:

    def __init__(self) -> None:
        '''
        Associate each radio site with the appropriate weather stations based on 
        the method provided. There are actually two steps in associating weather stations.
        The first step involves finding out which weather stations should be considered for
        each radio site. Second, given the set of weather stations for each radio site, what
        method should be used to aggregate the information from neighboring weather stations.
        In this method we implement the first step.
        There are three ways we can implement 

        To do:
            There might be weather stations and radio sites present in distances table that are
                not present in the met-real and rl-kpis tables. The rl sites and met stations
                should be validated before calculating optimal distance or k nearest method.
        

        Args:
            method: can be either of "optimal" or "k_nearest"
                "optimal", associates rl with ws that fall within an optimal distance
                "k_nearest", associates rl with k nearest ws
            use_ws_list: in order to consider ws for radio sites, only the ws with available
                real or forecast daily data should be used.
                "real", makes use of unique ws list from met_real table to check while associating rl to ws
                "forecast", makes use of unique ws list from met_forecast table to check while associating rl to ws
        

        Attributes:
            table_headers: a list of string values, that denotes the dataframe header
                for returned DistancesTable object
        '''
        self.table_headers = ['RL_Sites','assigned_WS','ws_distance']
    


    def k_nearest_method(self,deployment:Deployment,num_neighbors:int) -> Deployment:
        # set num_neighbors in rl kpis table attributes
        deployment.tables["rl-kpis"].num_neighbors = num_neighbors
        # get distances table from Deployment
        data_df = deployment.tables["distances"].get_df()
        # Get list of weather stations
        met_stations = list(deployment.tables["met-real"].get_df()['station_no'].unique())

        rl_sites = []
        ws_sites = []
        rl2ws_distances = []
        
        for i in range(data_df.shape[0]):
            num_ws_associated = 0
            sorted_row = pd.DataFrame(data_df.iloc[i,:]).sort_values(by=data_df.index[i])
            for j in range(data_df.shape[1]):
                if num_ws_associated < num_neighbors:
                    # Add weather station if it exists in the met-real table
                    if sorted_row.index[j] in met_stations:
                        rl_sites.append(data_df.index[i])
                        ws_sites.append(sorted_row.index[j])
                        rl2ws_distances.append(sorted_row.iloc[j,0])
                        num_ws_associated += 1
            
        data_df = pd.DataFrame(list(zip(rl_sites, ws_sites, rl2ws_distances)),
                                columns = self.table_headers)
        
        # set dataframe as current distances dataframe
        deployment.tables["distances"].set_df(data_df)

        # give feedback
        print(f"Assigned weather stations to radio sites based on k nearest method")

        return deployment
        



    def optimal_distance(self,deployment:Deployment,optimal_distance:float) -> Deployment:
        '''
        Args:
            optimal_distance: a float value to indicate the distance from a radio link
                within which a weather station is considered to be assigned to the radio
                link.
        '''
        # get distances table from Deployment
        data_df = deployment.tables["distances"].get_df()

        # check which weather stations are within an optimal radius
        # and consider them as associated with the radio site at the 
        # center of the circle
        rl_sites = []
        ws_sites = []
        rl2ws_distances = []
        
        # iterate over rows
        for i in range(len(data_df)):
            # check if current radio site exists in the rl-kpis table
            for j in range(len(data_df.columns)):
                # check if distance is within optimal distance
                if data_df.iloc[i,j] <= optimal_distance:
                    rl_sites.append(data_df.index[i])
                    ws_sites.append(data_df.columns[j])
                    rl2ws_distances.append(data_df.iloc[i,j])
        
        data_df = pd.DataFrame(list(zip(rl_sites, ws_sites, rl2ws_distances)),
                                columns = self.table_headers)
        
        # set dataframe as current distances dataframe
        deployment.tables["distances"].set_df(data_df)

        # give feedback
        print(f"Assigned weather stations to radio sites based on optimal distance method")
    
        return deployment




if __name__ == '__main__':
    pass 