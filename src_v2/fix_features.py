'''
Because of hardware or software failure, 
'''

from table import Table

class FixFeatures:

    def __init__(self) -> None:
        pass

    def fix_rl_kpis_features(self,table:Table) -> Table:
        '''
        In the rural (previous) dataset there is one inconsistant value. It's removed
        in order to properly type cast features.
        '''
        # get dataframe from table
        data_df = table.get_df()

        # fix scalibility_score wrong typecasting by removing inconsistent data point
        if 'scalibility_score' in list(data_df.columns):
            if data_df.loc[686810]['scalibility_score'] == '2025-01-01 00:00:00':
                data_df = data_df.drop(686810)

        # mw_connection_no values have commas in them and so we remove these commas
        data_df["mw_connection_no"] = data_df["mw_connection_no"].replace(regex=r"[a-zA-Z,;]", value="")

        # set dataframe from table
        table.set_df(data_df)
        
        return table

    def fix_met_real_features(self,table:Table) -> Table:
        '''
        met real can have features with unexpected values (e.g. 1,012.3 value considered 
        as string and cannot be converted to float because of the comma).
        '''
        # get table dataframe
        data_df = table.get_df()

        # wind_dir_max values have commas in them and so we remove these commas
        data_df["wind_dir_max"] = data_df["wind_dir_max"].replace(regex=r"[a-zA-Z,;]", value="")

        # pressure values have commas in them and so we remove these commas
        data_df["pressure"] = data_df["pressure"].replace(regex=r"[a-zA-Z,;]", value="")
        
        # pressure values have commas in them and so we remove these commas
        data_df["pressure_sea_level"] = data_df["pressure_sea_level"].replace(regex=r"[a-zA-Z,;]", value="")

        # set the table dataframe
        table.set_df(data_df)

        return table
    
    def fix_table(self,table:Table) -> Table:
        '''
        The table features are fixed based on the table (e.g rl-kpis, met-real etc). So, this public
        functon calls relevant functions based on the table type.
        '''
        if table.name == "met-real":
            return self.fix_met_real_features(table)
        elif table.name == "rl-kpis":
            return self.fix_rl_kpis_features(table)

        return table
    
if __name__ == '__main__':
    pass