

import pandas as pd
from deployment import Deployment



class MergedTable:

    def __init__(self,deployment:Deployment) -> None:
        '''
        Attributes:
            data_df: a pandas dataframe containing the current dataframe for MergedTable.
            unused_features: a list of strings that are column names which are considered
                to be unused and unnecessary for downstream operations. These are removed
                by methods from RemoveValues.

        '''
        self.data_df = None
        self.unused_features = ["mw_connection_no","measured_date", "measured_hour","station_no","RL_Sites"]
        self.forecast_days = deployment.tables["rl-kpis"].forecast_days
        self.num_prev_days = deployment.tables["rl-kpis"].num_prev_days
        self.unique_identifiers = deployment.tables["rl-kpis"].unique_identifiers
        self.rl_kpis_features = list(deployment.tables["rl-kpis"].default_feature_types.keys())
        self.rl_sites_features = list(deployment.tables["rl-sites"].default_feature_types.keys())
        self.met_real_features = list(deployment.tables["met-real"].default_feature_types.keys())
        self.met_stations_features = list(deployment.tables["met-stations"].default_feature_types.keys())
        self.distance_features = ['ws_distance']
        self.num_neighbors = deployment.tables["rl-kpis"].num_neighbors


    def set_unused_features(self,unused_features:list) -> None:
        '''
        Sets the unused feature list
        '''
        self.unused_features = unused_features


    def set_df(self,data_df:pd.DataFrame) -> None:
        '''
        Sets the current dataframe for merged table
        '''
        self.data_df = data_df


    def get_df(self) -> pd.DataFrame:
        '''
        Returns the current dataframe for the table
        '''
        return self.data_df
    

    def get_previous_day_features(self,num_prev_days:int=4) -> None:
        ''' 
        Adds feature columns from previous days

        For each row, this method retrieves the feature values from previous days if 
        available and adds the feature columns.
        '''
        # get unique entry identifier
        data_df_copy = self.data_df.copy(deep=True)
        
        # pandas column of datetime type is offset by number of previous day's features
        # to get previous day datetimes
        for i in range(num_prev_days-1):
            data_df_copy.loc[:, f"T-{i+1}"] = data_df_copy["datetime"] + pd.DateOffset(days=i-1)
    
        # merge existing df that contains previous n days columns with 
        # df that has unique identifier and other features
        for i in range(1, num_prev_days):
            target_day_column_name = f"T-{i}"
            
            data_df_copy = data_df_copy.merge(self.data_df, 
                        how = "inner", 
                        left_on = (self.unique_identifiers[0],self.unique_identifiers[1],
                                   target_day_column_name,"assigned_WS"),
                        right_on = tuple(self.unique_identifiers+["assigned_WS"]),
                        suffixes = ("", f"_T-{i}")
            )
        
            # remove static features that don't change with time (e.g. categorical features)
            remove_columns = ["T","datetime_T","1-day-predict_T"]
            for column in remove_columns:
                data_df_copy = data_df_copy.drop([f"{column}-{i}"], axis=1)
        
        # set label column as the last column
        label = data_df_copy.pop("1-day-predict")
        data_df_copy = pd.concat([data_df_copy, label], axis=1)
        # set assigned weather station as last column
        assigned_WS = data_df_copy.pop("assigned_WS")
        data_df_copy = pd.concat([data_df_copy, assigned_WS], axis=1)


        # set transformed dataframe as current dataframe for this table
        self.data_df = data_df_copy

        # Give feedback to the user
        print(f"Added previous day features to each datapoint for merged table")


    def __post_process_get_assigned_ws_features(self,data_df:pd.DataFrame) -> pd.DataFrame:
        '''
        There are redundant columns in dataframe now. The assigned WS columns are not necessary 
        and are removed. The 1-day-predict labels for different WS are removed. Only one is kept
        and is added at the end of the dataframe as the last column the float64 values are casted
        into float32 values.
        '''
        # seperate label column
        label = data_df.pop("1-day-predict")
        # remove redundant label columns and assigned WS name columns using regex
        cols_to_remove = data_df.filter(regex='^(assigned_WS|1-day-predict)').columns
        data_df = data_df.drop(cols_to_remove, axis=1)

        # add label column at the end
        data_df = pd.concat([data_df, label], axis=1)

        # cast float64 values into float32 values
        float64_cols = data_df.select_dtypes(include='float64').columns        
        data_df[float64_cols] = data_df[float64_cols].astype('float32')
        
        return data_df

    
    def get_assigned_ws_features(self) -> None:
        ''' 
        Adds assigned weather station features 
        
        To do:
            Group by site_id,mlid,datetime
            Iterate over each group and add all weather station data from that group into a list
            Append individual lists to create one final dataframe
            This will be O(n^4)
        '''
        # Get feature list excluding unique identifiers
        transformed_dataframe_features = self.data_df.columns.to_list()
        features_without_identifiers = self.data_df.columns.to_list()[len(self.unique_identifiers):]
        for assigned_ws in range(2,self.num_neighbors+1):
            assigned_ws_features = [f"{feature}_WS{assigned_ws}" for feature in features_without_identifiers]
            transformed_dataframe_features += assigned_ws_features

        # Group by site_id and aggregate the values into a list
        data_df = self.data_df.set_index(self.unique_identifiers)

        site_counter = 0
        transformed_dataset = []
        # iterate over each unique site_id
        for site_id in data_df.index.get_level_values(0).unique():
            # iterate over each unique mlid for a given site_id
            for mlid in data_df.loc[site_id].index.get_level_values(0).unique():
                # iterate over each date for a given (site_id and mlid) pair
                for date in data_df.loc[(site_id,mlid)].index.get_level_values(0).unique():
                    temp_row = []
                    counter = 0
                    for row in data_df.loc[(site_id,mlid,date)].reset_index().values.tolist():
                        if counter == 0:
                            temp_row += row[:len(self.unique_identifiers)]
                        temp_row += row[len(self.unique_identifiers):]
                        counter += 1
                    transformed_dataset.append(temp_row)

            site_counter += 1
            print(f"Processed {site_counter} sites out of {len(data_df.index.get_level_values(0).unique())}")
        
        # set the list transformed_dataset as dataframe
        data_df = pd.DataFrame(transformed_dataset, columns=transformed_dataframe_features)
        # post process transformed dataframe
        self.data_df = self.__post_process_get_assigned_ws_features(data_df)


    def get_assigned_ws_features_v1(self) -> None:
        ''' 
        Adds assigned weather station features 
        
        To do:
            Group by site_id,mlid,datetime
            Iterate over each group and add all weather station data from that group into a list
            Append individual lists to create one final dataframe
            This will be O(n)
        '''
        # Get feature list excluding unique identifiers
        transformed_dataframe_features = self.data_df.columns.to_list()
        features_without_identifiers = self.data_df.columns.to_list()[len(self.unique_identifiers):]
        for assigned_ws in range(2,self.num_neighbors+1):
            assigned_ws_features = [f"{feature}_WS{assigned_ws}" for feature in features_without_identifiers]
            transformed_dataframe_features += assigned_ws_features

        # Group by site_id and aggregate the values into a list
        data_df = self.data_df.groupby(self.unique_identifiers)[features_without_identifiers].apply(lambda x: x.values.tolist()).reset_index()
        
        # Rename the columns of the DataFrame
        data_df.columns = self.unique_identifiers + ['features']
        
        # Split the list of features into separate columns
        new_features_df = data_df['features'].apply(lambda x: pd.Series([item for sublist in x for item in sublist]))
        new_features_df.columns = transformed_dataframe_features[len(self.unique_identifiers):]
        
        # Drop the original 'features' column
        data_df = data_df.drop(columns=['features'])
        data_df = pd.concat([data_df, new_features_df], axis=1)
        
        # post process transformed dataframe
        self.data_df = self.__post_process_get_assigned_ws_features(data_df)


    def __group_func(self,group):
        """
        A function that takes a DataFrame group as input, processes the group, and returns a list of transformed rows.

        Args:
            group: A DataFrame group containing rows with the same combination of site_id, mlid, and date.

        Returns:
            A list of transformed rows.
        """
        temp_row = []
        counter = 0
        for row in group.values.tolist():
            if counter == 0:
                temp_row += row[:len(self.unique_identifiers)]
            temp_row += row[len(self.unique_identifiers):]
            counter += 1
        return temp_row

    def get_assigned_ws_features_vc(self) -> None:
        ''' 
        Adds assigned weather station features 
        
        To do:
            Group by site_id,mlid,datetime
            Iterate over each group and add all weather station data from that group into a list
            Append individual lists to create one final dataframe
            This will be O(n)
        '''
        # Get feature list excluding unique identifiers
        transformed_dataframe_features = self.data_df.columns.to_list()
        features_without_identifiers = self.data_df.columns.to_list()[len(self.unique_identifiers):]
        for assigned_ws in range(2,self.num_neighbors+1):
            assigned_ws_features = [f"{feature}_WS{assigned_ws}" for feature in features_without_identifiers]
            transformed_dataframe_features += assigned_ws_features

        # group the DataFrame by site_id, mlid, and date, and apply the group_func to each group
        self.data_df = self.data_df.groupby(self.unique_identifiers).apply(self.__group_func)

        # concatenate the resulting Series into a new DataFrame
        self.data_df = pd.DataFrame(
            self.data_df.tolist(),
            columns=transformed_dataframe_features)
        
        # post process transformed dataframe
        self.data_df = self.__post_process_get_assigned_ws_features(self.data_df)
    


if __name__ == '__main__':
    pass