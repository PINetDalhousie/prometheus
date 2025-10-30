
from table import Table
import pandas as pd
import numpy as np

class AggregateToDaily:

    def __init__(self) -> None:
        
        # This attribute keeps tracks of tables to perfrom daily aggregation on
        self.tables_to_aggregate = ["met-real","met-forecast"]

        pass

    def aggregate_continuos_features(self,table:Table) -> pd.DataFrame:
        '''
        Given a dataframe this function returns the aggregated dataframe for 
        continuos features. The aggregation is performed considering the tables
        unique idenfiers.
        '''
        # list of continuous features
        continuous_features = table.get_df().select_dtypes(include=['number']).columns.tolist()
        # Retrieve only the continuous features along with unique identifiers
        data_df_continuous = table.get_df()[table.unique_identifiers + continuous_features]

        # calculate mean based on unique identifiers.
        # After groupby operation we get a multi index dataframe
        data_df_continuous = data_df_continuous.groupby(table.unique_identifiers).mean()
        # Remove the multi index from the dataframe and return dataframe with one index
        data_df_continuous = data_df_continuous.reset_index()
        return data_df_continuous


    def aggregate_categorical_features(self,table:Table) -> pd.DataFrame:

        # Retrieve only the categorical features
        data_df_categorical = table.get_df().select_dtypes(exclude=['number'])

        # Group datapoints by table unique identifiers. This way data is aggregated to daily
        # points for categorical features. We calculate mode of each categorical feature over a day.
        data_df_categorical = data_df_categorical.groupby(table.unique_identifiers).agg(pd.Series.mode)

        # groupby method return multi-level dataframe. Reset index to get standard dataframe
        data_df_categorical = data_df_categorical.reset_index()

        # applying mode with groupby creates weather_day feature values of ndarray_numpy
        for column in data_df_categorical.columns.tolist():
            data_df_categorical[column] = data_df_categorical[column].transform(lambda x: x[0] if (type(x).__module__ == np.__name__) else x)

        return data_df_categorical
    


    def merge_aggregated_dataframes(self,data_df_continuous:pd.DataFrame,data_df_categorical:pd.DataFrame,
                                    table_unique_identifiers:list) -> pd.DataFrame:
        # merge continuous and categorical dataframes on station_no and datetime
        data_df = data_df_continuous.merge(data_df_categorical,
                                           how = "inner",
                                           left_on = tuple(table_unique_identifiers),
                                           right_on = tuple(table_unique_identifiers),
                                        )
        
        return data_df


    def aggregate(self,table:Table) -> Table:
        ''' Given a Table, this fmethod aggregates continuous and categorical features
        using mean and mode along the time dimension.
        '''
        if table.name in self.tables_to_aggregate:
            # aggregate continuos features
            data_df_continuous = self.aggregate_continuos_features(table)

            # aggregate categorical features
            data_df_categorical = self.aggregate_categorical_features(table)

            # merge output from aggregated dataframes
            data_df = self.merge_aggregated_dataframes(data_df_continuous,data_df_categorical,table.unique_identifiers)

            # set table dataframe
            table.set_df(data_df)

            print(f"Table was aggregated along time using unique identifiers. Aggregation method was Mean and Median")
            return table
        else:
            print(f"Table already has one data point per day. No aggregation was performed")
            return table


if __name__ == '__main__':
    pass