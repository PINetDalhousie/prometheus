'''
This class is reponsible for typecasting all the different tables. So, the state of the
typecastfeature object comprises of the different default feature types that are predefined.
The object will try to cast the features of a given table based on these default values.
'''

from table import Table

class TypecastFeatures():

    def __init__(self) -> None:
        pass

    def __print_current_feature_types(self,table:Table) -> None:
        '''
        print the current inferred features for the table
        '''
        print(table.get_df().info())
        

    def print_default_feature_types():
        '''
        print the default features of the table
        '''
        pass

    
    
    def cast_features(self,table:Table) -> Table:
        '''
        Based on the default feature types of the table, this method casts all feature types
        to default feature types. This increases efficiency down the pipeline. If there are 
        errors (ValueError) while casting, it means there are unexpected values in features
        that don't match the defalut feature types (e.g. a value of 1,012.3 can't be converted
        into float because of the comma). 

        Args:
            table: An object of class Table
            
        '''
        # Give current feature types to users first
        print(f"Current features and types for table {table.name}")
        self.__print_current_feature_types(table)

        # get dataframe from table
        data_df = table.get_df()
        
        # Only distances table is different than the rest of the tables. Distances table has
        # pairwise distances for all radio site and weather station pairs. We cannot use a 
        # dictionary of feature name and feature type, to cast the table. So, when distances
        # table is encountered the casting is handled differently comprared to other tables.
        if table.name != "distances":
            # only consider the subset of columns in the dictionary of default feature types, that
            # are present in the column list for current dataframe
            features_to_cast = {}
            for key in data_df.columns.to_list():
                if key in table.default_feature_types.keys():
                    features_to_cast[key] = table.default_feature_types[key]
                else:
                    print(f"WARNING: Column {key} is not present in default feature types for the table")
        else:
            features_to_cast = table.default_feature_types
        
        # cast the features that are present 
        data_df = data_df.astype(features_to_cast)
        table.set_df(data_df=data_df)

        # Give feedback to the user
        print(f"Casting current feature types of table {table.name} to default types for the corresponding features")
        print(f"Feature types after transformation")
        print(table.get_df().info())

        return table
    

if __name__ == '__main__':
    pass