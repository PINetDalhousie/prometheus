'''
Tables in the deployment may have high percentage of missing values or low percentage
of missing values. The features with high number of missing values are removed using
methods from RemoveValues class. While the features with low percentage of missing 
values are imputed using methods from this class. So this class is responsible for
imputing missing values.
'''


from table import Table
import pandas as pd


class ImputeValues:

    # available imputation methods
    MeanImputation = "mean"
    MedianImputation = "median"
    TimeSeriesImputation = "time_series"
    ModeImputation = "mode"


    # available table names
    Rl_Kpis = "rl-kpis"
    Met_Real = "met-real"
    Met_Forecast = "met-forecast"

    def __init__(self) -> None:
        # validation the passed imputation_method argument
        pass

    def verify_identifier_columns(self):
        '''
        Because of faulty data collection process. There might be some missing values
        in the columns that uniquely identify records of a table; which is not desirable.
        This method verifies if there are any missing values in the unique identifier
        columns of the tables
        '''
        pass


    def median_imputation(self,table:Table) -> Table:
        '''
        Perform median imputation on the continuous features of table.
        '''
        # get table dataframe
        data_df = table.get_df()
        # get only the continuous columns from dataframe
        data_df_view = data_df.select_dtypes(include=['number'])

        # Get the list of continuous features from the table dataframe
        continuous_features = data_df_view.columns.to_list()
        # Get the list of features with missing values 
        columns_with_missing_values = data_df_view.columns[data_df_view.isnull().any()].tolist()

        # check if there are any columns in the columns_with_missing_value list
        if len(columns_with_missing_values) > 0:
            # Itereate over the columns with missing values and perform median imputation
            # only on the ones that are continuos features
            for missing_value_column in columns_with_missing_values:
                if missing_value_column in continuous_features:
                    data_df[missing_value_column] = data_df[missing_value_column].fillna(data_df[missing_value_column].median())
                    print(f"Performed {ImputeValues.MedianImputation} imputation on column '{missing_value_column}'.")
        elif len(columns_with_missing_values) == 0:
            print(f"There were no columns to impute. Returning the same table")

        # set table dataframe 
        table.set_df(data_df)
        return table
    

    def __iterate_over_rlsite_mlid_pairs(self,data_df:pd.DataFrame,columns_to_impute:list) -> pd.DataFrame:
        '''
        rl-kpis table entities are minilinks. Each mini link is uniquely identified by
        (site_id, mlid) pairs. So, the dataframe is iterated over in two loops.
        '''
        # iterate over each unique site_id
        for site_id in data_df.index.get_level_values(0).unique():
            # iterate over each unique mlid for a given site_id
            for mlid in data_df.loc[site_id].index.get_level_values(0).unique():
                # sort dataframe based on values of datetime for each site_id mlid pairs
                data_df.loc[(site_id,mlid)] = data_df.loc[(site_id,mlid)].sort_values(by=['datetime'])
                # for each continuous feature, perform linear time series interpolation
                for column in columns_to_impute:
                   data_df.loc[(site_id,mlid)][column].interpolate(method='linear',inplace=True,limit_direction='both')
                
        return data_df


    def __iterate_over_met_stations(self,data_df:pd.DataFrame,columns_to_impute:list) -> pd.DataFrame:
        '''
        met-real and met-forecast table entities are met stations. Each met station is
        uniquely identified by station_no. So, the dataframe is terated over unique
        station_no.
        '''
        # iterate over each station and sort by datetime
        for station_no in data_df.index.get_level_values(0).unique():
            data_df.loc[station_no] = data_df.loc[station_no].sort_values(by=['datetime'])
            # for each continuous feature, perform linear time series interpolation
            for column in columns_to_impute:
                data_df.loc[station_no][column].interpolate(method='linear',inplace=True,limit_direction='both')
        
        return data_df


    def time_series_imputation(self,table:Table) -> Table:
        '''
        This method perform line time series imputation on the continuos features from Table
        dataframe. It uses the unique identifiers from the table to group relevant rows into
        groups using groupby method and then the interpolation is performed using the datetime column.
        
        '''
        # Time series imputation can only be performed if a datetime column is present
        # First, validate the presence of datetime column
        if "datetime" not in table.get_df().columns.to_list():
            print(f"There is no datetime column in the table dataframe. So time sereis imputation " \
                  "cannot be performed")
            return table

        # Tables have unique identifiers associated with them. This list of column names (unique
        # identifier) uniquely identifies each records in the table dataframe. As, we want to perform
        # time series interpolation along the time dimension, we first retrieve all records along time
        # for each entities (e.g. site_id and mlid pair for radio sites; station no for met-stations).
        # So, the dataframe is first sorted using unique identifiers excluding datetime column
        unique_identifiers_without_datetime = table.unique_identifiers[0:-1]
        data_df = table.get_df().sort_values(by=unique_identifiers_without_datetime)
        data_df = data_df.set_index(unique_identifiers_without_datetime)


        # Get the list of continuous features from the table dataframe
        continuous_features = data_df.columns.to_list()
        # Get the list of features with missing values 
        columns_with_missing_values = data_df.columns[data_df.isnull().any()].tolist()
        # get the continuous column names that will be imputed
        columns_to_impute = list(set(continuous_features).intersection(columns_with_missing_values))

        if len(columns_to_impute) > 0:
            # The rl-kpis table have two features as identifiers while the met-real and met-forecast have
            # one identifier. So, these two cases need two and one loops to iterate over respectively. So,
            # they are handled by different functions.
            if table.name == ImputeValues.Rl_Kpis:
                data_df = self.__iterate_over_rlsite_mlid_pairs(data_df,columns_to_impute)
            else:
                data_df = self.__iterate_over_met_stations(data_df,columns_to_impute)
        elif len(columns_to_impute) == 0:
            print(f"There's no continuous columns with missing values that need imputation")
            return table
        
        # reset index to remove multilevel index
        data_df = data_df.reset_index()
        
        # set table dataframe
        table.set_df(data_df)

        # give feedback 
        print(f"Time series imputation have been performed on continuous features of table {table.name}")

        return table


    def mode_imputation(self,table:Table) -> Table:
        '''
        Perform mode imputation on the continuous features of table.
        '''
        # get table dataframe
        data_df = table.get_df()
        # get only the categorical columns from dataframe
        data_df_view = data_df.select_dtypes(include=['object'])

        # Get the list of continuous features from the table dataframe
        categorical_features = data_df_view.columns.to_list()
        # Get the list of features with missing values 
        columns_with_missing_values = data_df_view.columns[data_df_view.isnull().any()].tolist()

        # check if there are any columns in the columns_with_missing_value list
        if len(columns_with_missing_values) > 0:
            # Itereate over the columns with missing values and perform mode imputation
            # only on the ones that are continuos features
            for missing_value_column in columns_with_missing_values:
                if missing_value_column in categorical_features:
                    data_df[missing_value_column] = data_df[missing_value_column].fillna(data_df[missing_value_column].mode()[0])
                    print(f"Performed {ImputeValues.ModeImputation} imputation on column '{missing_value_column}'.")
        elif len(columns_with_missing_values) == 0:
            print(f"There were no columns to impute. Returning the same table")

        # set table dataframe 
        table.set_df(data_df)
        return table


    def impute_missing_values(self,table:Table) -> Table:
        '''
        This method takes a Table as input and imputes the missing values
        '''
        pass



if __name__ == '__main__':
    pass