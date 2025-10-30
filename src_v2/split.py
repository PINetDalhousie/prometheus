
from merge_table import MergedTable
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

class Split:

    def __init__(self) -> None:
        self.train_val = None

    def random_split(self,merged_table:MergedTable,val_ratio:float,test_ratio:float) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
        '''
        Performs random split on the merged_table dataframe and returns
        train, validation and test set

        First, the input dataframe is split into (train+val) and test set based on test_ratio. This 
        (train+val) set has both the train set and the validation set. The (train+val) set can separated
        into train and validation set by using val_ratio. But input val_ratio is based on the complete
        dataset (train+val+test). So, val_ratio is used the calculate the relative validation ratio to
        use for seperating train and validation set from (train+val) set. 
        '''

        # Calculate val_ratio relative to train+validation set
        train_ratio = 1 - (val_ratio+test_ratio)
        val_ratio = 1 - (train_ratio/(train_ratio+val_ratio))

        # get dataframe
        data_df = merged_table.get_df()

        # Split into (train+val) and test
        train_val, test = train_test_split(data_df, test_size=test_ratio, random_state = 42)

        # Split the (train+val) set into train and validation
        train, validation = train_test_split(train_val, test_size=val_ratio, random_state = 42)

        return train,validation,test


    def __train_test_time_split(self,data_df:pd.DataFrame,test_size:float) -> Tuple[pd.DataFrame,pd.DataFrame]:
        '''
        Splits input dataframe data_df into train and test set in the temporal dimension.

        Get the maximum and minimum dates to calculate the datetime range for dataframe data_df. Then based on the
        range, calculate time index to use for splitting into train and test set.
        '''
        # Get minimum and maximum date of dataframe data_df
        min_date = data_df.datetime.min()
        max_date = data_df.datetime.max()

        # Calculate the range of datetime and time index for splitting the dataset
        time_range = max_date - min_date
        train_cutoff = min_date + (1-test_size)*time_range

        # Split the dataset based on train_cutoff
        train_df = data_df[data_df.datetime <= train_cutoff]
        test_df = data_df[data_df.datetime > train_cutoff]
        
        return train_df, test_df



    def time_split(self,merged_table:MergedTable,val_ratio:float,test_ratio:float) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
        # Calculate val_ratio relative to train+validation set
        train_ratio = 1-(val_ratio+test_ratio)
        val_ratio = 1 - (train_ratio/(train_ratio+val_ratio))

        data_df = merged_table.get_df()

        # Split into (train+val) and test
        train_val, test = self.__train_test_time_split(data_df, test_size=test_ratio)

        # Set train_val which is used for time_series_cross_validation
        self.train_val = train_val

        # Split the (train+val) set into train and validation
        train, validation = self.__train_test_time_split(train_val, test_size=val_ratio)

        # Give feedback to user
        print("Datetime range of Train Split: ", train.datetime.min(), train.datetime.max())
        print("Datetime range of Validation Split:", validation.datetime.min(), validation.datetime.max())
        print("Datetime range of Test Split:", test.datetime.min(), test.datetime.max())

        return train,validation,test
    

    def time_slice(self,data_df:pd.DataFrame,ratio_tuple:tuple) -> pd.DataFrame:
        """
        Slice a Pandas DataFrame based on a ratio tuple.

        Arguments:
            - data_df: A Pandas DataFrame to be sliced.
            - ratio_tuple: A tuple of two float values representing the lower and upper bounds of the slice.

        Returns:
            - A new Pandas DataFrame sliced according to the ratio tuple.

        Example Usage:
        data = pd.read_csv('data.csv')
        ratio = (0.2, 0.8)
        sliced_data = time_slice(data, ratio)
        """
        # Calculate the lower and upper bounds for the slice based on the ratio tuple
        low = (data_df["datetime"] > data_df["datetime"].min() + (data_df["datetime"].max() - data_df["datetime"].min()) * ratio_tuple[0])
        high = (data_df["datetime"] <= data_df["datetime"].min() + (data_df["datetime"].max() - data_df["datetime"].min()) * ratio_tuple[1])
        # Return a new DataFrame sliced according to the ratio tuple
        return data_df.loc[low & high]

    # def write_time_series_splits(train,validation,test,merged_table,current_time,args):
    #     train[i] = train[i].drop(merged_table.unique_identifiers, axis=1)
    #     validation[i] = validation[i].drop(merged_table.unique_identifiers, axis=1)
    #     test[i] = test[i].drop(merged_table.unique_identifiers, axis=1)

    #     # save 
    #     train[i].to_csv(f'../data/{current_time}/train_{args.approach}_{args.deployment}_time{i}_{args.name_ext}.csv',index=False)
    #     validation[i].to_csv(f'../data/{current_time}/validation_{args.approach}_{args.deployment}_time{i}_{args.name_ext}.csv',index=False)
    #     test[i].to_csv(f'../data/{current_time}/test_{args.approach}_{args.deployment}_time{i}_{args.name_ext}.csv',index=False)
    
    def time_series_cross_validation(self,merged_table:MergedTable,val_ratio:float,test_ratio:float,splits=5) -> Tuple[list,list,list]:
        '''
        Creates time series cross validation splits

        Dataframe is sliced based on time range. So, test_ratio of 0.1 represents the data in the last 10% time range.
        Train, validation and test ratio is defined by two ratio values. e.g. val_ratio of 0.2 will be represented as
        data in the 70%-90% time range. So, if train,val and test ratios are 0.7,0.2,0.1 then they are converted to
        tuples of (0.0,0.7),(0.7,0.9) and (0.9,1.0) for the first split. Iteratively to generate each split each of
        these tuples are offset by test_ratio. So, split 2 will have tuples of (0.0,0.6),(0.6,0.8) and (0.8,0.9).
        These tuple ratios are used to retrieve data points in those time ranges.
        '''
        # Define ratios
        train_ratio = 1 - test_ratio - val_ratio

        # Get current dataframe from merged_table
        data_df = merged_table.get_df()

        # Define ratio tuples
        ratios = {
            'train': (0.0, train_ratio),
            'validation': (train_ratio, train_ratio + val_ratio),
            'test': (train_ratio + val_ratio, 1.0)
        }

        # Initialize split lists
        train_splits = []
        validation_splits = []
        test_splits = []

        # Loop through splits
        for i in range(splits):
            # Update ratio tuples for current split
            if i != 0:
                ratios["train"] = (ratios["train"][0], ratios["train"][1]-test_ratio)
                ratios["validation"] = (ratios["validation"][0]-test_ratio, ratios["validation"][1]-test_ratio)
                ratios["test"] = (ratios["test"][0]-test_ratio, ratios["test"][1]-test_ratio)
            
            # Retrieve slice based on ratio tuples
            train = self.time_slice(data_df,ratios["train"])
            validation = self.time_slice(data_df,ratios["validation"])
            test = self.time_slice(data_df,ratios["test"])

            # Give feedback to user
            print("Datetime range of Train Split: ", train.datetime.min(), train.datetime.max())
            print(f"Number of samples in train data : {len(train)}")
            print("Datetime range of Validation Split:", validation.datetime.min(), validation.datetime.max())
            print(f"Number of samples in Validation data : {len(validation)}")
            print("Datetime range of Test Split:", test.datetime.min(), test.datetime.max())
            print(f"Number of samples in test data : {len(test)}")
            


            # Append split to a list
            train_splits.append(train)
            validation_splits.append(validation)
            test_splits.append(test)

        return train_splits,validation_splits,test_splits


if __name__ == '__main__':
    pass 
