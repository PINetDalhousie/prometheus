import tensorflow as tf
from data_preprocess_K6_XAI import DataPreprocess
import pandas as pd

class ToTfData():
    
    def __init__(
            self,
            batch_size:int,
            approach:str,
            autoencoder:str,
            positional_encoding:str,
            ) -> None:
        '''
        Attributes:
            batch_size: an integer value indicating the batch size to use
        '''
        self.batch_size = batch_size
        self.approach = approach
        self.autoencoder = autoencoder
        self.positional_encoding = positional_encoding

    def train_to_tfdata(self,kpis:pd.DataFrame,labels:pd.DataFrame) -> tf.data.Dataset:
        """
        Transforms pandas dataframe input and labels into iterable train tf.data.Dataset 
        based on the approach used
        """
        # Create tf.data.Dataset object from dataset
        train_ds = tf.data.Dataset.from_tensor_slices((kpis, labels))
        # Reshape based on approach used
        if self.approach == "new":
            train_ds = train_ds.map(DataPreprocess.reshape_for_weather_stations)
        elif self.approach == "prev" :
            train_ds = train_ds.map(DataPreprocess.reshape_for_time_steps)
        
        if self.positional_encoding:
            train_ds = train_ds.map(DataPreprocess.positional_encoding)
            
           #print("I am here after reshape of train data")

        if self.autoencoder:
            train_ds = train_ds.map(DataPreprocess.reshape_for_autoencoder)
        
        # batch data and return
        train_ds = train_ds.shuffle(10000)
        train_ds = train_ds.batch(self.batch_size,drop_remainder=True)
        # sample
        #train_ds = train_ds.take(1)

        return train_ds
    

    def val_to_tfdata(self,kpis:pd.DataFrame,labels:pd.DataFrame) -> tf.data.Dataset:
        """
        Transforms pandas dataframe input and labels into iterable validation tf.data.Dataset 
        based on the approach used
        """
        val_ds = tf.data.Dataset.from_tensor_slices((kpis, labels))
        print("val_ds", val_ds.__sizeof__)
        if self.approach == "new":
            val_ds = val_ds.map(DataPreprocess.reshape_for_weather_stations)
        else:
            val_ds = val_ds.map(DataPreprocess.reshape_for_time_steps)

        if self.positional_encoding:
            val_ds = val_ds.map(DataPreprocess.positional_encoding)
            
        if self.autoencoder:
            val_ds = val_ds.map(DataPreprocess.reshape_for_autoencoder)

        val_ds = val_ds.shuffle(10000)
        val_ds = val_ds.batch(self.batch_size,drop_remainder=True) 
        #val_ds = val_ds.take(1)
        
        return val_ds


    def test_to_tfdata(self,kpis:pd.DataFrame,labels:pd.DataFrame) -> tf.data.Dataset:
        """
        Transforms pandas dataframe input and labels into iterable test tf.data.Dataset 
        based on the approach used
        """
        # Create tf.data.Dataset object from dataset
        test_ds = tf.data.Dataset.from_tensor_slices((kpis, labels))
        # Reshape based on approach used
        if self.approach == "new":
            test_ds = test_ds.map(DataPreprocess.reshape_for_weather_stations)
        elif self.approach == "prev" :
            test_ds = test_ds.map(DataPreprocess.reshape_for_time_steps)

        if self.positional_encoding:
            test_ds = test_ds.map(DataPreprocess.positional_encoding)

        if self.autoencoder:
            test_ds = test_ds.map(DataPreprocess.test_reshape_for_autoencoder)
        
        test_ds = test_ds.batch(self.batch_size,drop_remainder=False)
        return test_ds



if __name__ == '__main__':
    pass