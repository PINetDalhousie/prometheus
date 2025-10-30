'''
The dataset comprises of different tables. We assume all these tables to be zipped into
a single .zip file. This class entity tries to keep track of all the constituent tables 
for a given deployment/dataset. So, the instance attributes comprise of tables. Deployment
is basically a collection of tables. Where each table is a Table object.
'''
from table import Table,RlkpisTable,RlsitesTable,MetrealTable,MetforecastTable,MetstationsTable,DistancesTable
import os


class Deployment:

    def __init__(self,dataset_zip:str,deployment_name:str) -> None:
        '''
        initialize the deployment
        '''
        # validate inputs
        assert os.path.exists(dataset_zip), f"{dataset_zip} is not a valid path" 

        # path to zip file
        self.dataset_zip = dataset_zip
        print(f"Deployment zip path : {self.dataset_zip}")
        # dictionary to keep track of the tables
        self.tables = {}
        self.name = deployment_name
        


    def add_table(self,table_name:str) -> None:
        '''
        Adds a table to the deployment object
        '''
        if table_name == "rl-kpis":
            print(self.dataset_zip)
            table = RlkpisTable(self.dataset_zip,table_name)
        if table_name == "rl-sites":
            table = RlsitesTable(self.dataset_zip,table_name)
        if table_name == "met-real":
            table = MetrealTable(self.dataset_zip,table_name)
        if table_name == "met-forecast":
            table = MetforecastTable(self.dataset_zip,table_name)
        if table_name == "met-stations":
            table = MetstationsTable(self.dataset_zip,table_name)
        if table_name == "distances":
            table = DistancesTable(self.dataset_zip,table_name)

        # get table as dataframe using Table method
        #table = Table(self.dataset_zip,table_name)
        # set the retrieved table 
        setattr(self,table_name,table)
        # add the table in the object dictionary attribute 
        self.tables.update({table_name:table})
        # give feedback to the user
        print(f"Added table {table_name} to deployment {self.name}")


    def __str__(self) -> str:
        '''
        print out the dataset zip path and name of the tables
        '''
        output_str = f"Deployment name : {self.name} \n"
        output_str += "Deployment path : {self.dataset_zip} \n"
        output_str += f"Tables in the deployment : \n"
        for table in self.tables.values():
            output_str += f"table name : {table.get_name()} \n"
        
        return output_str
        
if __name__ == '__main__':
    pass

    

