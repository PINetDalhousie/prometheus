'''
The main purpose of this class is to generate Exploratory Data Analysis reports in 
order to handle 
'''
import pandas as pd
from pathlib import Path
from table import Table
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random

class EDA:

    def __init__(self,output_path:str) -> None:
        # assert if the path is valid
        assert Path(output_path).exists(), f"Passed string is not a valid path"

        # set output path for EDA artifacts 
        self.output_path = output_path


    def __get_continuous_feature_report(self,data_df:pd.DataFrame) -> pd.DataFrame:
        '''
        Build feature report for continuous features as a pandas dataframe
        '''
        # get continuos features
        features = data_df.select_dtypes('number').columns.to_list()

        con_head = ['Count', 'Miss %', 'Card.', 'Min', '1st Qrt.',
                'Mean', 'Median', '3rd Qrt', 'Max', 'Std. Dev.']
        con_out_df = pd.DataFrame(index=features, columns=con_head)
        columns_df = data_df[features]
        # Calculate number of data points
        con_out_df[con_head[0]] = len(columns_df)
        # Calculate percentage of missing values
        con_out_df[con_head[1]] = columns_df.isna().sum() / len(columns_df) * 100
        # Calculate cardinality of a feature
        con_out_df[con_head[2]] = columns_df.nunique()
        # Calculate the minimum value of a feature
        con_out_df[con_head[3]] = columns_df.min()
        # Calculate the 1st quartile value of a feature
        con_out_df[con_head[4]] = columns_df.quantile(0.25)
        # Calculate the mean of a continuous feature
        con_out_df[con_head[5]] = columns_df.mean()
        # Calculate the median of a feature
        con_out_df[con_head[6]] = columns_df.median()
        # Calculate the 3rd quartile of a feature
        con_out_df[con_head[7]] = columns_df.quantile(0.75)
        # Calculate the maximum of a feature
        con_out_df[con_head[8]] = columns_df.max()
        # Calculate the standard deviation of a feature
        con_out_df[con_head[9]] = columns_df.std()
        
        return con_out_df

    
    def __get_categorical_feature_report(self,data_df:pd.DataFrame) -> pd.DataFrame:
        '''
        Build feature report for categorical features as a pandas dataframe
        '''
        # Get categorical features from dataframe
        features = data_df.select_dtypes(include=['object','bool']).columns.to_list()
        # Set output dataframe headers
        catHead = ['Count', 'Miss %', 'Card.', 'Mode', 'Mode Freq',
                'Mode %', '2nd Mode', '2nd Mode Freq', '2nd Mode %']
        # Get dataframe of categorical features only
        columns_df = data_df[features]
        # Prepare a dictionary for storing data
        stats_dict = {k: ['']*len(features) for k in catHead}
        # Calculate cardinality of feature
        stats_dict['Card.'] = columns_df.nunique()
        missing = columns_df.isna().sum() / len(columns_df) * 100

        # Iterate over each categorical features and calculate 
        for col in columns_df:
            values = columns_df[col].value_counts()
            index = features.index(col)
            # Calculate number of data points for features
            stats_dict['Count'][index] = len(columns_df)
            # Calculate number of missing values of feature
            stats_dict['Miss %'][index] = missing[col]
            # Calculate 1st and 2nd Mode of features
            mode = values.index[0]
            mode2 = values.index[1] if len(values.index) > 1 else mode
            stats_dict['Mode'][index] = mode
            stats_dict['2nd Mode'][index] = mode2
            # Calculate mode frequency of 1st and 2nd Modes
            modeCount = values.loc[mode]
            modeCount2 = values.loc[mode2]
            stats_dict['Mode Freq'][index] = modeCount
            stats_dict['2nd Mode Freq'][index] = modeCount2
            # Calculate percentage of data points having 1st and 2nd Mode values
            miss = stats_dict['Miss %'][index]
            modePer = (modeCount/(len(columns_df)*((100-miss)/100)))*100
            stats_dict['Mode %'][index] = round(modePer, 2)
            modePer2 = (modeCount2/(len(columns_df)*((100-miss)/100)))*100
            stats_dict['2nd Mode %'][index] = round(modePer2, 2)
        # write categorical feature report to disk
        output_df = pd.DataFrame.from_dict(stats_dict)

        return output_df


    def write_feature_reports(self,table:Table,report_name:str="") -> None:
        '''
        Write continuos and categorical feature reports for a Table

        Args:
            table: Table object to write the feature reports for
            report_name: additional name to add to file names
        '''
        # make directory if needed
        os.system(f" mkdir -p {self.output_path}feature_reports/")

        # Calculate continuous and categorical feature reports
        continuous_feature_report_df = self.__get_continuous_feature_report(table.get_df())
        categorical_feature_report_df = self.__get_categorical_feature_report(table.get_df())

        # Write the feature reports to disk
        continuous_feature_report_df.to_csv(self.output_path+"feature_reports/"+f"{table.name}_{report_name}_contFeatureReport_.csv")
        categorical_feature_report_df.to_csv(self.output_path+"feature_reports/"+f"{table.name}_{report_name}_catFeatureReport.csv")
        
        # give feedback to the user
        print(f"Wrote feature reports of table {table.name} at location {self.output_path}." \
            "In order to check if there are differences between inferred and defualt" \
            "feature types use the method match_inferred_and_default_feature_types")


    def box_plot(self,table:Table) -> None:
        '''save box plot of the continuoys features in table
        '''
        # make directory if needed
        os.system(f" mkdir -p {self.output_path}box_plots/")

        if table.name == 'distances':
            print(f"Cannot ge t box plots for '{table.name}' table")
            return
        else:
            # get continuous features from table
            cont_features = table.get_df().select_dtypes('number')
            # Iterate of continuous features and save corresponding box plots to disk
            for feature in cont_features:
                figure = sns.boxplot(x=table.get_df()[feature])
                figure.get_figure().savefig(f'{self.output_path}box_plots/{feature}.png')
                plt.close()

        # give user feedback
        print(f"Saved box plots of continuous features for table {table.name}")

    def visualize_failures(self,dataframe_splits,feature_name):
        '''
        Visualize failure events with respect to a feature column. 

        We want to see for example how many failures as one feature value (e.g. precipitation) changes over
        time for a specific weather station. This function plots the failure events with respect to a feature. 
        '''
        # make directory if needed
        os.system(f" mkdir -p {self.output_path}failure_plots/")
        
        # get first data split
        data_split = dataframe_splits[0]

        # get dataframe of a unique link feature values across time using (site_id,mlid,datetime) as index
        unique_link_feature_values = data_split[['site_id','mlid','datetime',feature_name,'1-day-predict']].drop_duplicates(subset=['site_id','mlid','datetime'])

        print(unique_link_feature_values.info(verbose=True))


        # get list of unique (site_id,mlid) pairs
        unique_pairs = unique_link_feature_values[['site_id', 'mlid']].drop_duplicates()
        unique_pairs = list(unique_pairs.to_records(index=False))

        print(unique_pairs[:10])

        # get 10 random (site_id, mlid) pairs
        random_pairs = random.sample(unique_pairs, 10)

        # loop over the random pairs and plot the feature values across time for each pair
        for random_pair in random_pairs:
            # get dataframe for the random pair
            unique_link = unique_link_feature_values[(unique_link_feature_values['site_id'] == random_pair[0]) & (unique_link_feature_values['mlid'] == random_pair[1])]

            # sort dataframe by datetime
            unique_link = unique_link.sort_values(by=['datetime'])

            # save as csv 
            unique_link.to_csv(f'{self.output_path}failure_plots/{feature_name}_over_time_{random_pair[0]}_{random_pair[1]}.csv',index=False)

            # plot feature values across time
            plt.plot(unique_link['datetime'], unique_link[feature_name])
            plt.xlabel('datetime')
            plt.ylabel(feature_name)
            plt.savefig(f'{self.output_path}failure_plots/{feature_name}_over_time_{random_pair[0]}_{random_pair[1]}.png')
            plt.close()


        # get dataframe for one pair at random
        # get a random (site_id, mlid) pair
        # random_pair = random.choice(unique_pairs)

        # # get dataframe for the random pair
        # unique_link_feature_values = unique_link_feature_values[(unique_link_feature_values['site_id'] == random_pair[0]) & (unique_link_feature_values['mlid'] == random_pair[1])]

        # #unique_link_feature_values = unique_link_feature_values[(unique_link_feature_values['site_id'] == unique_pairs[0][0]) & (unique_link_feature_values['mlid'] == unique_pairs[0][1])]

        # print(unique_link_feature_values.info(verbose=True))

        # # sort dataframe by datetime
        # unique_link_feature_values = unique_link_feature_values.sort_values(by=['datetime'])

        # # save as csv 
        # unique_link_feature_values.to_csv(f'{self.output_path}failure_plots/{feature_name}_over_time.csv',index=False)

        # # plot failure events with respect to feature_name
        # # only put a red marker if there is a failure event
        # # plot feature values across time
        # plt.plot(unique_link_feature_values['datetime'], unique_link_feature_values[feature_name])
        # plt.xlabel('datetime')
        # plt.ylabel(feature_name)
        # plt.savefig(f'{self.output_path}failure_plots/{feature_name}_over_time.png')
        # plt.close()
        

    def match_inferred_and_default_feature_types(self,table:Table):
        '''
        Each table is initialized with default feature types. This means certain kind
        of tables (e.g. rl-kpis) expect features to be of certain types. This method
        checks if the inferred and default feature types match or not
        '''
        


        pass


    def get_higly_missing_features():
        '''
        For each table there are features with high number of missing values. Given a
        table this function returns the list of features with high number of missing
        values in terms of percentage.
        '''
        pass



    def get_unnecessary_features(self,table:Table):
        '''
        Describes the unnecessary features for the current table
        '''

        pass




if __name__ == '__main__':
    pass