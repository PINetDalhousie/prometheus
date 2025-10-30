

from table import Table
import pandas as pd
from merge_table import MergedTable

class RemoveValues:

    def __init__(self) -> None:
        pass
    
    
    def remove_identifiers(self,merged_table:MergedTable) -> MergedTable:
        '''
        Removes unique identifiers from merged_table
        '''
        # get dataframe
        data_df = merged_table.get_df()
        # Drop unique identifiers
        data_df = data_df.drop(merged_table.unique_identifiers, axis=1)
        # set dataframe
        merged_table.set_df(data_df)
        # give feedback to user
        print(f"Removed unique identifiers from merged table.")
        return merged_table
    


    def remove_categorical_features(self,merged_table:MergedTable) -> MergedTable:
        '''
        Removes categorical features from MergedTable

        This method allows to remove categorical features so that we only consider
        continuous features in downstream tasks
        '''
        # get dataframe
        data_df = merged_table.get_df()

        # get categorical feature list without unique identifiers
        object_columns = data_df.select_dtypes(include=['object'])
        remove_categorical_columns = [column for column in object_columns if column not in merged_table.unique_identifiers]
        # For merged table assigned_WS will not be removed
        remove_categorical_columns.remove("assigned_WS")

        # remove categorical features
        data_df = data_df.drop(remove_categorical_columns, axis=1)

        # set dataframe
        merged_table.set_df(data_df)

        # give feedback to user
        print(f"Removed categorical features from merged table")
        return merged_table



    def remove_unused_features(self,merged_table:MergedTable) -> MergedTable:
        '''
        Removes unsed features from MergedTable.

        There are some features which are identified to be not useful for downstream tasks.
        These features are found in MergedTable attributes. This method removes those 
        columns and occurences of those columns in past days.

        To do:
            Add capability to remove these same values for future days
        '''
        # get dataframe
        data_df = merged_table.get_df()

        # Iterate over unused_features list in merged_table and drop those columns along
        # with their occurences in previous days.
        for feature in merged_table.unused_features:
            if feature in data_df.columns.tolist():
                data_df = data_df.drop([feature], axis=1)
                for i in range(1, merged_table.num_prev_days):
                    data_df = data_df.drop([f'{feature}_T-{i}'],axis=1,errors='ignore')
        
        # set dataframe
        merged_table.set_df(data_df)

        # Give feedback to user    
        print(f"Dropped unused features from merged table")
        return merged_table



    def remove_highly_missing_features(self,table:Table,threshold:float=0.2) -> Table:
        '''
        Given a table, remove columns with percentage of missing values that
        are higher than the threshold value.
        '''
        # get table dataframe
        data_df = table.get_df()

        # keep track of columns that are removed
        remove_columns = [] 
        missing_percentage_of_removed_columns = []

        # Calculate missing percentages for all columns
        percent_missing_in_df = data_df.isnull().sum() / len(data_df)
        # Iterate over column names and percentage of missing values per column
        for column, percent_missing_in_column in zip(data_df.columns,percent_missing_in_df):
            # If percentage of missing value is over threshold argument, then remove the column
            if percent_missing_in_column > threshold:
                data_df = data_df.drop([column], axis=1)
                # update list of removed columns
                remove_columns.append(column)
                missing_percentage_of_removed_columns.append(percent_missing_in_column)
    
        # set table dataframe
        table.set_df(data_df)

        # give feedback to users
        if len(remove_columns) > 0:
            print(f"Columns {remove_columns} have been removed from table {table.name}, " \
                f"because of having the following percentage of missing values " \
                f"{missing_percentage_of_removed_columns}.")
        elif len(remove_columns) == 0:
            print(f"No columns were removed as missing value percentage for columns were " \
                  f"below the threshold")
        
        return table



    def remove_time_from_datetime(self,table:Table) -> Table:
        '''
        Datetime features can have both date and time for some tables. But we only need
        date for this problem. So, in order to perform aggregation methods properly, the 
        time component should be removed from datetime. This function takes a table and
        removes the time component from datetime
        '''
        # get table dataframe
        data_df = table.get_df()

        if 'datetime' in data_df.columns.to_list():
            # remove time from datetime
            data_df['datetime'] = pd.to_datetime(data_df['datetime'].dt.date)
            print(f"Removed time component from datetiime feature")
        else:
            print(f"datetime column not present in table. Returning unchanged table")
        
        # set table dataframe
        table.set_df(data_df)


        return table
    


if __name__ == '__main__':
    pass