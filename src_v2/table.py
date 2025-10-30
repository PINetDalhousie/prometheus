

from zipfile import ZipFile
import pandas as pd
import os
from typing import Tuple

class Table:
    '''
    Represents a table from ITU dataset

    This class abstracts away the individual table (kpi,rl_sites,distances etc). It
    keeps track of a pandas dataframe as a reoresentation of the table.

    Attributes:
        df: a pandas dataframe to keep track 
        name: string that identifies unique table name
    
    Class Attributes:
        default_itu_table_names: a list of strings that contain the default ITU 
            table names. 
    '''
    # Class attributes for table. The default table names are used to validate if 
    # the passed table name is correct or not.
    default_itu_table_names = ['rl-kpis','rl-sites','met-forecast','met-real',
                               'met-stations','distances']
    

    def __init__(self,dataset_zip:str,table_name:str) -> None:
        """
        read dataframe from zip file which contains multiple 
        tab seperated text files
        
        Args: 
            dataset_zip - string
                path to regionA zip file
            table_name - array-like of shape (n_samples,)
                Corresponding label for each sample in X.
        Returns:
            table_from_zip - pandas dataframe
                one of the tables from zip file
        """
        # Validate that path is correct
        assert os.path.exists(dataset_zip), f"{dataset_zip} is not a valid path" 
        # Validate if the table name follows the convention of ITU table names
        assert table_name in Table.default_itu_table_names, f"{table_name} is not a valid table name"

        # retrieve table from zip file
        print(f"Reading {table_name} table from {dataset_zip}")
        with ZipFile(dataset_zip) as zip_file:
            with zip_file.open(table_name+".tsv") as file:
                df = pd.read_csv(file, sep="\t", index_col=0, low_memory=False)

        # set current dataframe in self and current table name as name attribute
        self.df = df
        self.name = table_name
       
        # Based on the table, the default features and their types are set. These 
        # features are likely to be present in the table and their default data types
        # are also known. These are later on used by the feature typecasting class 
        # properly cast features. If you want to change the default types for the
        # features, you should change and set the following attribute.
        self.default_feature_types = None

        # Different tables have different columns as unique identifiers. e.g. for rl-kpis
        # (site_id,mlid,datetime) uniquely identifies each link while for met-real and 
        # met-forecast (station_no,datetime) uniquely identifies each data point. Some
        # operations require the function to know about the unique identifiers for the 
        # table. So we keep track of these using the following attribute.
        self.unique_identifiers = None


    def get_df(self) -> pd.DataFrame:
        '''
        Returns the current dataframe for the table
        '''
        return self.df
    
    def set_df(self,data_df:pd.DataFrame) -> None:
        '''
        Sets the passed dataframe as current df for the table
        '''
        self.df = data_df
    

    def get_name(self) -> str:
        '''
        Returns the table name
        '''
        return self.name
    

    def __str__(self) -> str:
        '''
        Returns the table representation which is just the table name as of now.
        '''
        print(f"table name : {self.name}")
        print(self.get_df().info())
        return ""
    

    
    
class RlkpisTable(Table):
    '''
    This class is responsible for representing rl-kpis table and includes
    methods that are specific to the table.
    '''

    def __init__(self,dataset_zip:str,table_name:str) -> None:
        # initialize parent class attributes
        super().__init__(dataset_zip, table_name)
        self.default_feature_types = self.__get_rl_kpis_default_feature_types()
        self.unique_identifiers = ["site_id","mlid","datetime"]
        self.forecast_days = 0
        self.num_prev_days = 0
        self.num_neighbors = 0


    def __get_rl_kpis_default_feature_types(self) -> dict:
        '''
        This function returns the default feature types for rl-kpis table.
        '''
        return {
                "type":"object",
                "datetime":"datetime64[ns]",
                "tip":"object",
                "mlid":"object",
                "site_id":"object",
                "card_type":"object",
                "adaptive_modulation":"object",
                "freq_band":"object",
                "modulation":"object",
                "polarization":"object",
                "direction":"object",
                "mw_connection_no":"float32",
                "neid":"float32",
                "severaly_error_second":"float32",
                "error_second":"float32",
                "unavail_second":"float32",
                "avail_time":"float32",
                "link_length":"float32",
                "bbe":"float32",
                "rxlevmax":"float32",
                "capacity":"float32",
                "scalibility_score":"float32",
                "rlf":"int"
            }

    def calculate_label(self,forecast_days:int=1) -> None:
        '''
        Calculates label columns(rlf) for future days, using datetime and rlf columns.

        The current dataframe in this table object has rlf column for a given datetime.
        But we want to predict next day rlf for each radio link. Each link is uniquely
        identified by site_id, mlid and datetime. So, this method checks if there is rlf
        values for the next days for a given datetime, and if it is available, adds the
        label_rlf column as target value. 

        Args:
            farecast_days: An integer indicating the number of future days to calculate
                rlf labels for. (e.g. a value of 2 would add next two days rlf columns)
        '''
        # set object attribute value
        self.forecast_days = forecast_days

        # get unique entry identifier and remove rlf column
        # for rl-kpis datetime,site_id and mlid together uniquely identifies each entry
        data_df = self.df.copy(deep=True)
        data_df.drop(columns=["rlf"], inplace=True)

        # pandas column of datetime type is offset by 1 day to get 
        # the next days datetime for each entry
        for i in range(1, forecast_days+1):
            data_df.loc[:, f"T+{i}"] = data_df["datetime"] + pd.DateOffset(days=i)

        # merge existing df that contains following day columns with 
        # df that has unique identifier and rlf (true or false) 
        # e.g left merge t+1,site_id,mlid with datetime,site_id,mlid
        rl_kpis_view = self.df[self.unique_identifiers + ["rlf"]]
        # iterate over each day column to get rlf columns for next days
        for i in range(1, forecast_days+1):
            target_day_column_name = f"T+{i}"

            data_df = data_df.merge(rl_kpis_view, 
                        how = "inner", 
                        left_on = (self.unique_identifiers[0], self.unique_identifiers[1],
                                   target_day_column_name),
                        right_on = tuple(self.unique_identifiers),
                        suffixes = ("", "_y")
            )
            # rename rlf columns according to target day name
            data_df.rename(columns={"rlf": f"{i}-day-predict"}, inplace=True)
            # remove extra T+{i} and datetime_y columns that are produced for having overlapping column
            # name with different column values
            data_df.drop(columns=[f"T+{i}", "datetime_y"], inplace=True)
        
        # Set the current dataframe as the new dataframe with label column
        self.df = data_df

        # Give feedback to user
        print(f"Added label radio link failure columns to the table dataframe for {self.name} table")


    def get_ious_day_features(self,num_prev_days:int=4) -> None:
        ''' 
        Adds feature columns from previous days

        For each row, this method retrieves the feature values from previous days if 
        available and adds the feature columns.
        '''
        # set object attribute
        self.num_prev_days = num_prev_days

        # get unique entry identifier
        data_df = self.df.copy(deep=True)
        
        # pandas column of datetime type is offset by number of previous day's features
        # to get previous day datetimes
        for i in range(num_prev_days-1):
            data_df.loc[:, f"T-{i+1}"] = data_df["datetime"] + pd.DateOffset(days=i-1)
    
        # merge existing df that contains previous n days columns with 
        # df that has unique identifier and other features
        for i in range(1, num_prev_days):
            target_day_column_name = f"T-{i}"
            
            data_df = data_df.merge(self.df, 
                        how = "inner", 
                        left_on = (self.unique_identifiers[0],self.unique_identifiers[1],
                                   target_day_column_name),
                        right_on = tuple(self.unique_identifiers),
                        suffixes = ("", f"_T-{i}")
            )
        
            # remove static features that don't change with time (e.g. categorical features)
            remove_columns = self.df.select_dtypes(exclude=['number']).columns.to_list()
            remove_columns = [column+"_T" for column in remove_columns if column not in self.unique_identifiers[0:-1]]
            remove_columns.append("T")
            for column in remove_columns:
                data_df = data_df.drop([f"{column}-{i}"], axis=1)
        
        # set transformed dataframe as current dataframe for this table
        self.df = data_df

        # Give feedback to the user
        print(f"Added previous day features to each datapoint for {self.name} table")



class RlsitesTable(Table):
    '''
    This class is responsible for representing rl-sites table and includes
    methods that are specific to the table.
    '''

    def __init__(self,dataset_zip:str,table_name:str) -> None:
        # initialize parent class attributes
        super().__init__(dataset_zip, table_name)
        self.default_feature_types = self.__get_rl_sites_default_feature_types()
        self.unique_identifiers = ["site_id",]

    
    def __get_rl_sites_default_feature_types(self) -> dict:
        '''
        Given this function returns the default feature types for rl-sites table.
        '''
        return {
                "site_id":"object",
                "clutter_class":"object",
                "groundheight":"float32"
            }


class MetrealTable(Table):
    '''
    This class is responsible for representing met-real table and includes
    methods that are specific to the table.
    '''

    def __init__(self,dataset_zip:str,table_name:str) -> None:
        # initialize parent class attributes
        super().__init__(dataset_zip, table_name)
        self.default_feature_types = self.__get_met_real_default_feature_types()
        self.unique_identifiers = ["station_no", "datetime"]

    def __get_met_real_default_feature_types(self) -> dict:
        '''
        Given this function returns the default feature types for met-real table.
        '''
        return {
                "station_no":"object",
                "datetime":"datetime64[ns]",
                "measured_date":"object",
                "measured_hour":"object",
                "temp":"float32",
                "temp_max":"float32",
                "temp_min":"float32",
                "wind_dir":"float32",
                "wind_dir_max":"float32",
                "wind_speed":"float32",
                "wind_speed_max":"float32",
                "humidity":"float32",
                "precipitation":"float32",
                "precipitation_coeff":"float32",
                "pressure":"float32",
                "pressure_sea_level":"float32",
            }
    
    def get_previous_day_features(self,num_prev_days:int=4) -> None:
        ''' 
        Adds feature columns from previous days

        For each row, this method retrieves the feature values from previous days if 
        available and adds the feature columns.
        '''

        # get unique entry identifier
        data_df = self.df.copy(deep=True)
        
        # pandas column of datetime type is offset by number of previous day's features
        # to get previous day datetimes
        for i in range(num_prev_days-1):
            data_df.loc[:, f"T-{i+1}"] = data_df["datetime"] + pd.DateOffset(days=i-1)
    
        # merge existing df that contains previous n days columns with 
        # df that has unique identifier and other features
        for i in range(1, num_prev_days):
            target_day_column_name = f"T-{i}"
            
            data_df = data_df.merge(self.df, 
                        how = "inner", 
                        left_on = (self.unique_identifiers[0],target_day_column_name),
                        right_on = tuple(self.unique_identifiers),
                        suffixes = ("", f"_T-{i}")
            )
        
            # remove static features that don't change with time (e.g. categorical features)
            remove_columns = self.df.select_dtypes(exclude=['number']).columns.to_list()
            remove_columns = [column+"_T" for column in remove_columns if column not in self.unique_identifiers[0:-1]]
            remove_columns.append("T")
            for column in remove_columns:
                data_df = data_df.drop([f"{column}-{i}"], axis=1)
        
        # set transformed dataframe as current dataframe for this table
        self.df = data_df

        # Give feedback to the user
        print(f"Added previous day features to each datapoint for {self.name} table")



class MetforecastTable(Table):
    '''
    This class is responsible for representing met-forecast table and includes
    methods that are specific to the table.
    '''

    def __init__(self,dataset_zip:str,table_name:str) -> None:
        # initialize parent class attributes
        super().__init__(dataset_zip, table_name)
        self.default_feature_types = self.__get_met_forecast_default_feature_types()
        self.unique_identifiers = ["station_no", "datetime"]
    
    def __get_met_forecast_default_feature_types(self) -> dict:
        '''
        Given this function returns the default feature types for met-forecast table.
        '''
        feature_types_dict = {
                "station_no":"object",
                "datetime":"datetime64[ns]",
                "report_time":"object"
            }
        for i in range(1,6):
                feature_types_dict.update({
                    f"weather_day{i}":"object",
                    f"temp_max_day{i}":"float32",
                    f"temp_min_day{i}":"float32",
                    f"humidity_max_day{i}":"float32",
                    f"humidity_min_day{i}":"float32",
                    f"wind_dir_day{i}":"float32",
                    f"wind_speed_day{i}":"float32",
                })
        return feature_types_dict

class MetstationsTable(Table):
    '''
    This class is responsible for representing met-stsations table and includes
    methods that are specific to the table.
    '''

    def __init__(self,dataset_zip:str,table_name:str) -> None:
        # initialize parent class attributes
        super().__init__(dataset_zip, table_name)
        self.default_feature_types = self.__get_met_stations_default_feature_types()

    def __get_met_stations_default_feature_types(self) -> dict:
        '''
        Given this function returns the default feature types for met-stations table.
        '''
        return {
                "station_no":"object",
                "clutter_class":"object",
                "height":"float32"
            }


class DistancesTable(Table):
    '''
    This class is responsible for representing distances table and includes
    methods that are specific to the table.
    '''

    def __init__(self,dataset_zip:str,table_name:str) -> None:
        # initialize parent class attributes
        super().__init__(dataset_zip, table_name)
        self.default_feature_types = self.__get_distances_default_feature_types()

    def __get_distances_default_feature_types(self) -> str:
        '''
        Given this function returns the default feature types for met-stations table.
        '''
        return "float32"
    
    def remove_ws_rows(self):
        '''
        Removes all rows with weather station values
        '''
        # remove all rows with weather station index
        filter = self.df.index.str.contains("^WS")
        data_df = self.df[~filter]

        # remove all columns with radio site name
        data_df = data_df[data_df.columns.drop(list(data_df.filter(regex="^RL")))]
        
        # set dataframe as current table dataframe
        self.df = data_df



if __name__ == '__main__':
    pass