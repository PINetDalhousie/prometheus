

from merge_table import MergedTable
from typing import Tuple
import pandas as pd
from table import MetrealTable,MetstationsTable,RlkpisTable

class AggregateWS:

    def __init__(self) -> None:
        pass

    
    def __get_met_features_to_seperate(self,features_without_identifiers:list,merged_table:MergedTable) -> list:
        # Get list of met features to seperate
        met_features_to_seperate = []
        for feature in features_without_identifiers:
            if feature in merged_table.met_real_features:
                met_features_to_seperate.append(feature)
                for day in range(1,merged_table.num_prev_days):
                    met_features_to_seperate.append(f"{feature}_T-{day}")
                continue
            
            # add met stations features
            if feature in merged_table.met_stations_features:
                met_features_to_seperate.append(feature)
        
        # add distance feature and return value
        return met_features_to_seperate + merged_table.distance_features



    def __seperate_met_columns(self,merged_table:MergedTable) -> Tuple[pd.DataFrame,pd.DataFrame]:
        '''
        Seperates weather station columns of a merged table.
        Returns:
            rl_df: a pandas DataFrame containing radio link unique identifier columns along
                with radio link features.
            met_df: a pandas DataFrame containing radio link unique identifier columns along
                with weather station features.
        '''
        # Get dataframe
        data_df = merged_table.get_df()

        # Get list of merged_table features without identifiers
        features_without_identifiers = [x for x in data_df.columns.to_list() if x not in merged_table.unique_identifiers] 

        # get list of rl and met features to use for separating merged dataframe into rl and met dataframes
        met_features_to_seperate = self.__get_met_features_to_seperate(features_without_identifiers,merged_table)
        rl_features_to_seperate = [feature for feature in features_without_identifiers if feature not in met_features_to_seperate]
        
        # Seperate dataframes based on rl or met type
        met_df = data_df[merged_table.unique_identifiers + met_features_to_seperate]
        rl_df = data_df[merged_table.unique_identifiers + rl_features_to_seperate]
        
        # Drop duplicates from the radio link dataframe
        rl_df = rl_df.drop_duplicates(subset=merged_table.unique_identifiers)

        # Set merged_table dataframe to None
        merged_table.set_df(None)

        return rl_df,met_df



    def aggregate(self,merged_table:MergedTable,
                  derived_features:list=['mean','min','max','std']) -> MergedTable:
        '''
        Merges dataframe and returns a MergedTable containing the mereged dataframe.
        '''
        # Get seperate dataframe
        rl_df,met_df = self.__seperate_met_columns(merged_table)

        # Aggregate weather features based on derived_features  
        met_df = met_df.groupby(merged_table.unique_identifiers).agg(derived_features)
        met_df.columns = ['_'.join(col) for col in met_df.columns]
        met_df = met_df.reset_index()
        
        # Merge weather station dataframe with the radio link dataframe
        data_df = rl_df.merge(met_df,
                              how="inner",
                              left_on=tuple(merged_table.unique_identifiers),
                              right_on=tuple(merged_table.unique_identifiers),
                              suffixes=("_rl", "_met")
                              )
        
        # Set merged_table dataframe
        merged_table.set_df(data_df)

        # Give feedback to user
        print(f"Aggregated weather station features to get derived features {derived_features}")

        return merged_table


    def sort_columns(self,merged_table:MergedTable):
        '''
        Sorts columns so that the first three columns are unique identifiers and categorical features are ordered
        after numerical features

        Args:
            merged_table (MergedTable): A MergedTable object containing the data to be sorted.

        Returns:
            MergedTable: A MergedTable object with sorted columns.
        '''
        
        # Get the dataframe from the MergedTable object
        data_df = merged_table.get_df()

        # Get a list of numerical columns
        data_df_numerical = data_df.select_dtypes(include='number').columns.to_list()

        # Remove unique identifiers from numerical columns
        data_df_numerical = [i for i in data_df_numerical if i not in merged_table.unique_identifiers]

        # Get a list of categorical columns
        data_df_categorical = data_df.select_dtypes(include='object').columns.to_list()

        # Remove unique identifiers from categorical columns
        data_df_categorical = [i for i in data_df_categorical if i not in merged_table.unique_identifiers]
        
        # Sort the dataframe columns
        data_df = data_df[merged_table.unique_identifiers+data_df_numerical+data_df_categorical]

        # Move the "1-day-predict" column to the end of the dataframe
        label = data_df.pop("1-day-predict")
        data_df = pd.concat([data_df, label], axis=1)

        # Update the MergedTable object with the sorted dataframe
        merged_table.set_df(data_df)

        return merged_table



if __name__ == '__main__':
    pass
