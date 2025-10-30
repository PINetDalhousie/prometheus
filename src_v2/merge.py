


from deployment import Deployment
from table import RlkpisTable,RlsitesTable,MetforecastTable,MetrealTable,DistancesTable,Table
from merge_table import MergedTable

class Merge:

    def __init__(self) -> None:
        pass



    def _merge_with_rlkpis(self,deployment:Deployment,merged_table:MergedTable) -> MergedTable:
        '''
        Merge dataframe from MergedTable with rlkpis
        '''
        if merged_table.get_df() is not None:
            # retrieve merged and rl-kpis dataframe
            merged_df = merged_table.get_df()
            rl_kpis = deployment.tables["rl-kpis"].get_df()
            # Merge the two dataframes
            merged_df = merged_df.merge(rl_kpis,
                                how="inner",
                                left_on=("site_id"),
                                right_on=("site_id"),
                                suffixes=("_merged", "_rl_kpis")
                                )
            merged_table.set_df(merged_df)
        else:
            merged_table.set_df(deployment.tables["rl-kpis"].get_df())

        # Remove the unnecessary tables from deployment
        deployment.tables.pop("rl-kpis")

        return merged_table


    def _merge_with_rlsites(self,deployment:Deployment,merged_table:MergedTable) -> MergedTable:
        '''
        Merge with rl-sites
        '''
        if merged_table.get_df() is not None:
            # retrieve merged and rl-kpis dataframe
            merged_df = merged_table.get_df()
            rl_sites = deployment.tables["rl-sites"].get_df()
            # Merge the two dataframes
            merged_df = merged_df.merge(rl_sites,
                                how="inner",
                                left_on=("site_id"),
                                right_on=("site_id"),
                                suffixes=("_merged", "_rl_sites")
                                )
            merged_table.set_df(merged_df)
        else:
            merged_table.set_df(deployment.tables["rl-sites"].get_df())

        # Remove the unnecessary tables from deployment
        deployment.tables.pop("rl-sites")

        return merged_table


    def _merge_with_distances(self,deployment:Deployment,merged_table:MergedTable) -> MergedTable:
        '''
        Merge with distances
        '''
        if merged_table.get_df() is not None:
            # retrieve merged and rl-kpis dataframe
            merged_df = merged_table.get_df()
            distances = deployment.tables["distances"].get_df()
            # Merge the two dataframes
            merged_df = merged_df.merge(distances,
                                how="inner",
                                left_on=("site_id"),
                                right_on=("RL_Sites"),
                                suffixes=("_merged", "_distances")
                                )
            merged_table.set_df(merged_df)
        else:
            merged_table.set_df(deployment.tables["rl-sites"].get_df())
        # Remove the unnecessary tables from deployment
        deployment.tables.pop("distances")

        return merged_table


    def _merge_with_metreal(self,deployment:Deployment,merged_table:MergedTable) -> MergedTable:
        '''
        Merge met-real with met-stations
        '''
        if merged_table.get_df() is not None:
            # retrieve merged and rl-kpis dataframe
            merged_df = merged_table.get_df()
            met_real = deployment.tables["met-real"].get_df()
            # Merge the two dataframes
            merged_df = merged_df.merge(met_real,
                                how="inner",
                                left_on=("assigned_WS", "datetime"),
                                right_on=("station_no", "datetime"),
                                suffixes=("_merged", "_met_real")
                                )
            merged_table.set_df(merged_df)
        else:
            merged_table.set_df(deployment.tables["met-real"].get_df())
        # Remove the unnecessary tables from deployment
        deployment.tables.pop("met-real")
        return merged_table
    
    

    def _merge_with_metstations(self,deployment:Deployment,merged_table:MergedTable) -> MergedTable:
        '''
        Merge met-real with met-stations
        '''
        if merged_table.get_df() is not None:
            # retrieve merged and rl-kpis dataframe
            merged_df = merged_table.get_df()
            met_real = deployment.tables["met-stations"].get_df()
            # Merge the two dataframes
            merged_df = merged_df.merge(met_real,
                                how="inner",
                                left_on=("station_no"),
                                right_on=("station_no"),
                                suffixes=("_merged", "_met_stations")
                                )
            merged_table.set_df(merged_df)
        else:
            merged_table.set_df(deployment.tables["met-stations"].get_df())
        # Remove the unnecessary tables from deployment
        deployment.tables.pop("met-stations")
        return merged_table


    def merge(self,deployment:Deployment,merged_table:MergedTable) -> MergedTable:
        '''
        Merge tables within a Deployment.

        This method merges the different tables and removes the tables from
        deployment so that MergedTable contains the final merged dataframe 
        after the merging process
        '''
        merged_table = self._merge_with_rlkpis(deployment,merged_table)
        merged_table = self._merge_with_rlsites(deployment,merged_table)
        merged_table = self._merge_with_distances(deployment,merged_table)
        merged_table = self._merge_with_metreal(deployment,merged_table)
        merged_table = self._merge_with_metstations(deployment,merged_table)

        return merged_table

        

        


if __name__ == '__main__':
    pass
