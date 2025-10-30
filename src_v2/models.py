import time
from time import perf_counter
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import random
from layers import SeperateCatFeatures, ReshapeBatch, ReshapeAndSliceStaticFeatures, ReshapeLSTMOutput, MaxReduction, ExpandDims, ReshapeBatchInverse, ReshapeBatchWS, PositionalEncoding, EncoderLayer, SumReduction, TransformerEncoder
from sklearn import svm
from data_preprocess_K import DataPreprocess
from tensorflow import keras
#from keras import register_keras_serializable
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
#class SVM(tf.keras.Model):
#Create a svm Classifier
    #clf = svm.SVC(kernel='sigmoid') 
    #return clf
class SVM0(tf.keras.Model):
    def __init__(self,**kwargs):
        super(SVM0, self).__init__()

        # Set up attributes
        ratio = kwargs["ratio"]
        self.num_stations = kwargs["num_stations"]
        self.prev_days_data = kwargs["prev_days_data"]
        self.num_features = kwargs["num_features"] + 1
        self.batch_size = kwargs["batch_size"]
        self.cat_features_number = kwargs["cat_features_number"]

        # set bias for last dense layer
       # bias_values = [np.log(1-ratio),np.log(ratio)]

        # Set up layers
        self.reshape_batch = ReshapeBatch(self.prev_days_data,self.num_features)

        self.seperate_cat_features = SeperateCatFeatures(self.cat_features_number)
        self.reshape_and_slice_cat = ReshapeAndSliceStaticFeatures(
            self.batch_size,
            self.cat_features_number
        )

        '''
        self.positional_encoding_1=PositionalEncoding(
            position=self.prev_days_data,
            d_model=self.num_features - self.cat_features_number
        )
        '''
       # Dense layers
        self.svm_1 = svm.SVC(random_state = 42, degree=3,class_weight=ratio, kernel='sigmoid')
        self.model_2 = svm.SVC(kernel='sigmoid')
        #self.dense_2 = svm.SVC(kernel='sigmoid') 
        #self.dense_3 = tf.keras.layers.Dense(self.num_features-self.cat_features_number,activation="relu")

        
        #self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        
        #self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        # Set up reshape LSTM output layer
        self.reshape_lstm_output = ReshapeLSTMOutput(
            self.batch_size,
            lstm_output=int(self.num_features-self.cat_features_number)
            )
        self.reshape_out = ReshapeLSTMOutput(
            self.batch_size,
            lstm_output=int(self.num_features)
            )
        self.maximum = MaxReduction(axis=1)
        # Define Static Branch
        self.dense_static_1 = tf.keras.layers.Dense(64)
        self.dense_static_2 = tf.keras.layers.Dense(17)

        # Concatenate output from two branches
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        # Add Dense layer after concatenation 
        self.dense_concat = tf.keras.layers.Dense(16)

        # Set up dense and softmax layers
        self.model_3 = svm.SVC(kernel='sigmoid')
        self.softmax = tf.keras.layers.Softmax()


    def call(self, inputs, training):
        if training == True:
            inputs = inputs[:,0:random.randint(1,self.num_stations),:,:]

        inputs = self.reshape_batch(inputs)
        # Seperate categorical, continuous weather and continuous link features
        temporal_features,static_features = self.seperate_cat_features(inputs)

       # temporal_features = self.positional_encoding_1(temporal_features)

        # Encode continuous weather and continuous link features
        model_1 = self.svm_1()
       #model= self.model_1(inputs)
    
       # model_1.fit(inputs);
        #temporal_features = self.global_pool(temporal_features)
        
       # temporal_features = self.reshape_lstm_output(temporal_features)
       # temporal_features = self.maximum(temporal_features)
       # print(temporal_features.shape)
       # static_features = self.model_2(static_features)
      #  static_features = self.dense_static_1(static_features)
       # static_features = self.dense_static_2(static_features)
        
        #combined = self.concat([temporal_features,static_features])
       # combined = self.dense_concat(combined)
        #print("Before reshape", combined.shape)
        #combined=self.reshape_out(combined)
        #print("completed reshape", combined.shape)
        #combined = self.dmodel_3(combined)
        #combined = self.softmax(combined)
        return model_1

class VarLSTM(tf.keras.Model):
    """
    A class that implements a variable input number LSTM model.

    Args:
        ratio (float): The ratio of the last dense layer bias values.
        num_stations (int): The number of stations.
        prev_days_data (int): The number of previous days' data.
        num_features (int): The number of features.
        batch_size (int): The batch size.

    Attributes:
        reshape_batch (ReshapeBatch): A ReshapeBatch object.
        lstm_1 (tf.keras.layers.LSTM): An LSTM layer with 128 units and return sequences set to True.
        lstm_2 (tf.keras.layers.LSTM): An LSTM layer with 64 units and return sequences set to True.
        lstm_3 (tf.keras.layers.LSTM): An LSTM layer with 32 units and return sequences set to True.
        lstm_4 (tf.keras.layers.LSTM): An LSTM layer with 16 units.
        reshape_lstm_output (ReshapeLSTMOutput): A ReshapeLSTMOutput object.
        maximum (MaxReduction): A MaxReduction object.
        dense (tf.keras.layers.Dense): A Dense layer with 2 units and bias initializer set to Constant(bias_values).
        softmax (tf.keras.layers.Softmax): A Softmax layer.

    Methods:
        call(inputs, training):
            Executes the model.

            Args:
                inputs (tensor): The input tensor.
                training (bool): Whether the model is being trained.

            Returns:
                tensor: The output tensor.
    """

    def __init__(self,**kwargs):
        super(VarLSTM, self).__init__()

        # Set up attributes
        ratio = kwargs["ratio"]
        self.num_stations = kwargs["num_stations"]
        self.prev_days_data = kwargs["prev_days_data"]
        self.num_features = kwargs["num_features"]
        self.batch_size = kwargs["batch_size"]

        # set bias for last dense layer
        bias_values = [np.log(1-ratio),np.log(ratio)]

        # Set up layers
        self.reshape_batch = ReshapeBatch(self.prev_days_data,self.num_features)

        # Set up LSTM layers
        self.lstm_1 = tf.keras.layers.LSTM(128,return_sequences=True)
        self.lstm_2 = tf.keras.layers.LSTM(64,return_sequences=True)
        self.lstm_3 = tf.keras.layers.LSTM(32,return_sequences=True)
        self.lstm_4 = tf.keras.layers.LSTM(16)

        # Set up reshape LSTM output layer
        self.reshape_lstm_output = ReshapeLSTMOutput(self.batch_size,lstm_output=self.num_features)

        # Set up maximum reduction layer
        self.maximum = MaxReduction(axis=1)

        # Set up dense and softmax layers
        self.dense = tf.keras.layers.Dense(2,bias_initializer=tf.keras.initializers.Constant(bias_values))
        self.softmax = tf.keras.layers.Softmax()


    def call(self, inputs, training):
        if training == True:
            inputs = inputs[:,0:random.randint(1,self.num_stations),:,:]

        x = self.reshape_batch(inputs)
        x = self.lstm_1(x)
        x = self.lstm_2(x)
        x = self.lstm_3(x)
        x = self.lstm_4(x)
        x = self.reshape_lstm_output(x)
       
        x = self.maximum(x)
        x = self.dense(x)
        x = self.softmax(x)
        return x



class VanillaLSTM(tf.keras.Model):

    def __init__(self,**kwargs):
        super(VanillaLSTM, self).__init__()
        
        ratio = kwargs["ratio"]
        # set bias for last dense layer
        bias_values = [np.log(1-ratio),np.log(ratio)]
        self.lstm_1 = tf.keras.layers.LSTM(128,return_sequences=True)
        self.lstm_2 = tf.keras.layers.LSTM(64,return_sequences=True)
        self.lstm_3 = tf.keras.layers.LSTM(32,return_sequences=True)
        self.lstm_4 = tf.keras.layers.LSTM(16)
        self.dense = tf.keras.layers.Dense(2,bias_initializer=tf.keras.initializers.Constant(bias_values))
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs):
        x = self.lstm_1(inputs)
        x = self.lstm_2(x)
        x = self.lstm_3(x)
        x = self.lstm_4(x)
        x = self.dense(x)
        x = self.softmax(x)
        return x
    



class LSTMPlus(tf.keras.Model):

    def __init__(self,**kwargs):
        super(LSTMPlus, self).__init__()
        
        ratio = kwargs["ratio"]
        cat_features_number = kwargs["cat_features_number"]
        self.seperate_cat_features = SeperateCatFeatures(cat_features_number)
        
        # Define Temporal Branch
        self.lstm_1 = tf.keras.layers.LSTM(128,return_sequences=True)
        self.lstm_2 = tf.keras.layers.LSTM(64,return_sequences=True)
        self.lstm_3 = tf.keras.layers.LSTM(32,return_sequences=True)
        self.lstm_4 = tf.keras.layers.LSTM(16)

        # Define Static Branch
        self.dense_static_1 = tf.keras.layers.Dense(32)
        self.dense_static_2 = tf.keras.layers.Dense(16)

        # Concatenate output from two branches
        self.concat = tf.keras.layers.Concatenate(axis=-1)

        # Add Dense layer after concatenation 
        self.dense_concat = tf.keras.layers.Dense(16)

        # set bias for last dense layer
        bias_values = [np.log(1-ratio),np.log(ratio)]
        self.dense = tf.keras.layers.Dense(2,bias_initializer=tf.keras.initializers.Constant(bias_values))

        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs):
        temporal_features,static_features = self.seperate_cat_features(inputs)
        temporal_features = self.lstm_1(temporal_features)
        temporal_features = self.lstm_2(temporal_features)
        temporal_features = self.lstm_3(temporal_features)
        temporal_features = self.lstm_4(temporal_features)

        static_features = self.dense_static_1(static_features)
        static_features = self.dense_static_2(static_features)

        combined = self.concat([temporal_features,static_features])
        combined = self.dense_concat(combined)
        combined = self.dense(combined)
        combined = self.softmax(combined)
        return combined
class LSTMPlus2(tf.keras.Model):

    def __init__(self,**kwargs):
        super(LSTMPlus2, self).__init__()
        
        ratio = kwargs["ratio"]
        cat_features_number = kwargs["cat_features_number"]
        self.seperate_cat_features = SeperateCatFeatures(cat_features_number)
        
        # Define Temporal Branch
       # self.lstm_1 = tf.keras.layers.LSTM(128,return_sequences=True)
        #self.lstm_2 = tf.keras.layers.LSTM(64,return_sequences=True)
        self.lstm_3 = tf.keras.layers.LSTM(32,return_sequences=True)
        self.lstm_4 = tf.keras.layers.LSTM(16)

        # Define Static Branch
        self.dense_static_1 = tf.keras.layers.Dense(32)
        self.dense_static_2 = tf.keras.layers.Dense(16)

        # Concatenate output from two branches
        self.concat = tf.keras.layers.Concatenate(axis=-1)

        # Add Dense layer after concatenation 
        self.dense_concat = tf.keras.layers.Dense(16)

        # set bias for last dense layer
        bias_values = [np.log(1-ratio),np.log(ratio)]
        self.dense = tf.keras.layers.Dense(2,bias_initializer=tf.keras.initializers.Constant(bias_values))

        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs):
        temporal_features,static_features = self.seperate_cat_features(inputs)
        temporal_features = temporal_features[:, :, :8]
        #temporal_features = self.lstm_1(temporal_features)
        #temporal_features = self.lstm_2(temporal_features)
        temporal_features = self.lstm_3(temporal_features)
        temporal_features = self.lstm_4(temporal_features)

        static_features = self.dense_static_1(static_features)
        static_features = self.dense_static_2(static_features)

        #combined = self.concat([temporal_features,static_features])
        combined = self.dense_concat(temporal_features)
        combined = self.dense(combined)
        combined = self.softmax(combined)
        return combined    

class LSTMPlus_KC(tf.keras.Model):

    def __init__(self,**kwargs):
        super(LSTMPlus_KC, self).__init__()
        
        ratio = kwargs["ratio"]
        cat_features_number = kwargs["cat_features_number"]
        self.seperate_cat_features = SeperateCatFeatures(cat_features_number)
        
        # Define Temporal Branch
        self.lstm_1 = tf.keras.layers.LSTM(128,return_sequences=True)
        self.lstm_2 = tf.keras.layers.LSTM(64,return_sequences=True)
        self.lstm_3 = tf.keras.layers.LSTM(32,return_sequences=True)
        self.lstm_4 = tf.keras.layers.LSTM(16)

        # Define Static Branch
        self.dense_static_1 = tf.keras.layers.Dense(32)
        self.dense_static_2 = tf.keras.layers.Dense(16)

        # Concatenate output from two branches
        self.concat = tf.keras.layers.Concatenate(axis=-1)

        # Add Dense layer after concatenation 
        self.dense_concat = tf.keras.layers.Dense(16)

        # set bias for last dense layer
        bias_values = [np.log(1-ratio),np.log(ratio)]
        self.dense = tf.keras.layers.Dense(2,bias_initializer=tf.keras.initializers.Constant(bias_values))

        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs):
        inputs= tf.convert_to_tensor(inputs)
         #   inputs=tf.reshape(inputs,[1024,3,4,84])
       # inputs=tf.reshape(inputs,[1024,4,107])
        inputs=tf.reshape(inputs,[1024,4,96]) #for fold 2-96, 1-97
        print("input feature shape:", inputs.shape)
        temporal_features,static_features = self.seperate_cat_features(inputs)
        print("feature shape:",temporal_features.shape, static_features.shape)
        #temporal_features = temporal_features[:, :, :8]
        temporal_features = self.lstm_1(temporal_features)
        temporal_features = self.lstm_2(temporal_features)
        temporal_features = self.lstm_3(temporal_features)
        temporal_features = self.lstm_4(temporal_features)

        static_features = self.dense_static_1(static_features)
        static_features = self.dense_static_2(static_features)

        combined = self.concat([temporal_features,static_features])
        combined = self.dense_concat(temporal_features)
        combined = self.dense(combined)
        combined = self.softmax(combined)
        return combined  
class LSTMPlusExp(tf.keras.Model):

    def __init__(self,**kwargs):
        super(LSTMPlusExp, self).__init__()

        ratio = kwargs["ratio"]
        cat_features_number = kwargs["cat_features_number"]
        self.seperate_cat_features = SeperateCatFeatures(cat_features_number)
        
        # Define Temporal Branch
        self.lstm_1 = tf.keras.layers.LSTM(128,return_sequences=True)
        self.lstm_2 = tf.keras.layers.LSTM(64,return_sequences=True)
        self.lstm_3 = tf.keras.layers.LSTM(32,return_sequences=True)
        self.lstm_4 = tf.keras.layers.LSTM(16)

        # Define Static Branch
        self.dense_static_1 = tf.keras.layers.Dense(32)
        self.dense_static_2 = tf.keras.layers.Dense(16)

        # Concatenate output from two branches
        self.concat = tf.keras.layers.Concatenate(axis=-1)

        # Add Dense layer after concatenation 
        self.dense_concat = tf.keras.layers.Dense(16)

        # set bias for last dense layer
        bias_values = [np.log(1-ratio),np.log(ratio)]
        self.dense = tf.keras.layers.Dense(2,bias_initializer=tf.keras.initializers.Constant(bias_values))

        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs):
        temporal_features,static_features = self.seperate_cat_features(inputs)
        index = [2, 4, 6, 7, 8, 10, 12, 13, 16, 17, 20, 24, 31, 32]
        temporal_features = tf.gather(temporal_features, indices=index, axis=2)
        #temporal_features = temporal_features[:, :, :8]
        temporal_features = self.lstm_1(temporal_features)
        temporal_features = self.lstm_2(temporal_features)
        temporal_features = self.lstm_3(temporal_features)
        temporal_features = self.lstm_4(temporal_features)

        static_features = self.dense_static_1(static_features)
        static_features = self.dense_static_2(static_features)

        #combined = self.concat([temporal_features,static_features])
        combined = self.dense_concat(temporal_features)
        combined = self.dense(combined)
        combined = self.softmax(combined)
        return combined    
class ULSTMPlus(tf.keras.Model):

    def __init__(self,**kwargs):
        super(ULSTMPlus, self).__init__()
        
        ratio = kwargs["ratio"]
        cat_features_number = kwargs["cat_features_number"]
        self.seperate_cat_features = SeperateCatFeatures(cat_features_number)
        
        # Define Temporal Branch
        #self.lstm_1 = tf.keras.layers.LSTM(128,return_sequences=True)
        self.lstm_2 = tf.keras.layers.LSTM(64,return_sequences=True)
        #self.lstm_3 = tf.keras.layers.LSTM(32,return_sequences=True)
        self.lstm_4 = tf.keras.layers.LSTM(16,return_sequences=True)
        #self.lstm_5 = tf.keras.layers.LSTM(8,return_sequences=True)
        self.lstm_6 = tf.keras.layers.LSTM(4)

        # Define Static Branch
        self.dense_static_1 = tf.keras.layers.Dense(32)
        self.dense_static_2 = tf.keras.layers.Dense(16)

        # Concatenate output from two branches
        self.concat = tf.keras.layers.Concatenate(axis=-1)

        # Add Dense layer after concatenation 
        self.dense_concat = tf.keras.layers.Dense(16)

        # set bias for last dense layer
        bias_values = [np.log(1-ratio),np.log(ratio)]
        self.dense = tf.keras.layers.Dense(2,bias_initializer=tf.keras.initializers.Constant(bias_values))

        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs):
        temporal_features,static_features = self.seperate_cat_features(inputs)
        #temporal_features = self.lstm_1(temporal_features)
        temporal_features = self.lstm_2(temporal_features)
        #temporal_features = self.lstm_3(temporal_features)
        temporal_features = self.lstm_4(temporal_features)
        #temporal_features = self.lstm_5(temporal_features)
        temporal_features = self.lstm_6(temporal_features)

        static_features = self.dense_static_1(static_features)
        static_features = self.dense_static_2(static_features)

        combined = self.concat([temporal_features,static_features])
        combined = self.dense_concat(combined)
        combined = self.dense(combined)
        combined = self.softmax(combined)
        return combined
    

class VarLSTMPlus(tf.keras.Model):
    """
    
    """

    def __init__(self,**kwargs):
        super(VarLSTMPlus, self).__init__()

        # Set up attributes
        ratio = kwargs["ratio"]
        self.num_stations = kwargs["num_stations"]
        self.prev_days_data = kwargs["prev_days_data"]
        self.num_features = kwargs["num_features"]
        self.batch_size = kwargs["batch_size"]
        self.cat_features_number = kwargs["cat_features_number"]

        # set bias for last dense layer
        bias_values = [np.log(1-ratio),np.log(ratio)]

        # Set up layers
        self.reshape_batch = ReshapeBatch(self.prev_days_data,self.num_features)

        self.seperate_cat_features = SeperateCatFeatures(self.cat_features_number)
        self.reshape_and_slice_cat = ReshapeAndSliceStaticFeatures(
            self.batch_size,
            self.cat_features_number
        )

        # Set up LSTM layers
        self.lstm_1 = tf.keras.layers.LSTM(128,return_sequences=True)
        self.lstm_2 = tf.keras.layers.LSTM(64,return_sequences=True)
        self.lstm_3 = tf.keras.layers.LSTM(32,return_sequences=True)
        self.lstm_4 = tf.keras.layers.LSTM(16)

        # Set up reshape LSTM output layer
        self.reshape_lstm_output = ReshapeLSTMOutput(
            self.batch_size,
            lstm_output=int(self.num_features-self.cat_features_number)
            )

        # Set up maximum reduction layer
        self.maximum = MaxReduction(axis=1)

        # Define Static Branch
        self.dense_static_1 = tf.keras.layers.Dense(32)
        self.dense_static_2 = tf.keras.layers.Dense(16)

        # Concatenate output from two branches
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        # Add Dense layer after concatenation 
        self.dense_concat = tf.keras.layers.Dense(16)

        # Set up dense and softmax layers
        self.dense = tf.keras.layers.Dense(
            2,
            bias_initializer=tf.keras.initializers.Constant(bias_values)
            )
        self.softmax = tf.keras.layers.Softmax()


    def call(self, inputs, training):
        if training == True:
            inputs = inputs[:,0:random.randint(1,self.num_stations),:,:]

        inputs = self.reshape_batch(inputs)
        temporal_features,static_features = self.seperate_cat_features(inputs)
        
        temporal_features = self.lstm_1(temporal_features)
        temporal_features = self.lstm_2(temporal_features)
        temporal_features = self.lstm_3(temporal_features)
        temporal_features = self.lstm_4(temporal_features)
        temporal_features = self.reshape_lstm_output(temporal_features)
        temporal_features = self.maximum(temporal_features)
        
        static_features = self.reshape_and_slice_cat(static_features)
        static_features = self.dense_static_1(static_features)
        static_features = self.dense_static_2(static_features)
        
        combined = self.concat([temporal_features,static_features])
        combined = self.dense_concat(combined)
        combined = self.dense(combined)
        combined = self.softmax(combined)
        return combined



class LSTMAutoencoderV0(tf.keras.Model):
    def __init__(self,**kwargs):
        super(LSTMAutoencoder, self).__init__()

        self.feature_number = kwargs["feature_number"]
        self.timesteps = kwargs["prev_days_data"]
        cat_features_number = kwargs["cat_features_number"]
        #cont_features_number = self.feature_number - cat_features_number

        self.seperate_cat_features = SeperateCatFeatures(cat_features_number)

        self.encoder1 = tf.keras.layers.LSTM(128, return_sequences=True)
        self.encoder2 = tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True)
        self.encoder3 = tf.keras.layers.LSTM(32, activation='tanh', return_sequences=True)
        self.encoder4 = tf.keras.layers.LSTM(16, activation='tanh', return_sequences=False,return_state=True)

        self.repeat = tf.keras.layers.RepeatVector(self.timesteps)

        # Define Static Branch
        self.dense_static_1 = tf.keras.layers.Dense(32)
        self.dense_static_2 = tf.keras.layers.Dense(16)

        # Concatenate output from two branches
        self.concat = tf.keras.layers.Concatenate(axis=-1)

        self.decoder1 = tf.keras.layers.LSTM(16, activation='tanh', return_sequences=True)
        self.decoder2 = tf.keras.layers.LSTM(32, activation='tanh', return_sequences=True)
        self.decoder3 = tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True)
        self.decoder4 = tf.keras.layers.LSTM(self.feature_number, activation='tanh', return_sequences=True)

        # Define Static Decoder Branch
        self.dense_static_d1 = tf.keras.layers.Dense(16)
        self.dense_static_d2 = tf.keras.layers.Dense(32)
        
        self.dense_output = tf.keras.layers.Dense(self.feature_number)
        self.time_distributed = tf.keras.layers.TimeDistributed(self.dense_output)
        
    def call(self, x):
        x, y = self.seperate_cat_features(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x,_,_ = self.encoder4(x)

        y = self.dense_static_1(y)
        y = self.dense_static_2(y)

        x = self.concat([x,y])
        x = self.repeat(x)

        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.decoder4(x)
        x = self.time_distributed(x)
        return x
    
class LSTMAutoencoder(tf.keras.Model):
    def __init__(self,**kwargs):
        super(LSTMAutoencoder, self).__init__()

        self.feature_number = kwargs["feature_number"]
        self.timesteps = kwargs["prev_days_data"]
        cat_features_number = kwargs["cat_features_number"]
        
        self.encoder1 = tf.keras.layers.LSTM(32, activation='tanh',return_sequences=True)
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.encoder3 = tf.keras.layers.LSTM(24, activation='tanh', return_sequences=False,return_state=True)
        
        self.repeat = tf.keras.layers.RepeatVector(self.timesteps)
        
        self.decoder0 = tf.keras.layers.LSTM(24, activation='tanh', return_sequences=True)
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.decoder2 = tf.keras.layers.LSTM(32, activation='tanh', return_sequences=True)
        self.batch_norm_3 = tf.keras.layers.BatchNormalization()
        self.decoder3 = tf.keras.layers.LSTM(self.feature_number, activation=None, return_sequences=True)
                
    def call(self, x):
        #print(x.shape)
        x = self.encoder1(x)
        x = self.batch_norm_1(x)
        x,h,c = self.encoder3(x)
        x = self.repeat(x)
        x = self.decoder0(x,initial_state=[h,c])
        x = self.batch_norm_2(x)
        x = self.decoder2(x)
        x = self.batch_norm_3(x)
        x = self.decoder3(x)
        return x
    
class LSTMAutoencoder_latest(tf.keras.Model):
    def __init__(self,**kwargs):
        super(LSTMAutoencoder_latest, self).__init__()

        self.feature_number = kwargs["feature_number"]
        self.timesteps = kwargs["prev_days_data"]
        cat_features_number = kwargs["cat_features_number"]
        
        #self.encoder1 = tf.keras.layers.LSTM(512,return_sequences=True)
        self.encoder2 = tf.keras.layers.LSTM(512,return_sequences=True)
        self.encoder3 = tf.keras.layers.LSTM(256,return_sequences=True)
        self.encoder4 = tf.keras.layers.LSTM(128,return_sequences=False,return_state=True)

        self.repeat = tf.keras.layers.RepeatVector(self.timesteps)
        
        self.decoder1 = tf.keras.layers.LSTM(128,return_sequences=True)
        self.decoder2 = tf.keras.layers.LSTM(256,return_sequences=True)
        self.decoder3 = tf.keras.layers.LSTM(512,return_sequences=True)
        #self.decoder4 = tf.keras.layers.LSTM(512,return_sequences=True)
        
        self.dense_output = tf.keras.layers.Dense(self.feature_number-cat_features_number)
        self.time_distributed = tf.keras.layers.TimeDistributed(self.dense_output)
                
    def call(self, x):
        #x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x,h,c = self.encoder4(x)
        x = self.repeat(x)
        x = self.decoder1(x,initial_state=[h,c])
        x = self.decoder2(x)
        x = self.decoder3(x)
        #x = self.decoder4(x)
        x = self.time_distributed(x)
        return x
    
class LSTMAutoencoder_Prev(tf.keras.Model):
    def __init__(self,**kwargs):
        super(LSTMAutoencoder_Prev, self).__init__()

        self.feature_number = kwargs["feature_number"]
        self.timesteps = kwargs["prev_days_data"]
        cat_features_number = kwargs["cat_features_number"]
        
        self.encoder1 = tf.keras.layers.LSTM(256,activation='tanh',return_sequences=True)
        self.encoder2 = tf.keras.layers.LSTM(128,activation='tanh',return_sequences=True)
        self.encoder3 = tf.keras.layers.LSTM(64,activation='tanh',return_sequences=True)
        self.encoder4 = tf.keras.layers.LSTM(32,activation='tanh',return_sequences=False)

        self.repeat = tf.keras.layers.RepeatVector(self.timesteps)
        
        #self.decoder1 = tf.keras.layers.LSTM(256,return_sequences=True)
        
        self.dense_output = tf.keras.layers.Dense(self.feature_number)
        self.time_distributed = tf.keras.layers.TimeDistributed(self.dense_output)
                
    def call(self, x):
        #print(x.shape)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)
        #print(x.shape)
        x = self.repeat(x)
        #print(x.shape)
        #x = self.decoder1(x)
        
        x = self.time_distributed(x)
        #print(x.shape)
        #print(asd)
        return x


class LSTMAutoencoderPlus(tf.keras.Model):
    def __init__(self,**kwargs):
        super(LSTMAutoencoderPlus, self).__init__()

        self.feature_number = kwargs["feature_number"]
        self.timesteps = kwargs["prev_days_data"]
        self.num_cat_features = kwargs["cat_features_number"]
        self.num_temporal_features = kwargs["feature_number"] - kwargs["cat_features_number"]

        # Separate categorical features
        self.seperate_cat_features = SeperateCatFeatures(self.num_cat_features)
        
        # Define Temporal Encoder Branch
        self.lstm_encoder_1 = tf.keras.layers.LSTM(128, activation='relu',return_sequences=True)
        #self.add_e1 = tf.keras.layers.Add()
        #self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.lstm_encoder_2 = tf.keras.layers.LSTM(128, activation='relu',return_sequences=True)
        #self.add_e2 = tf.keras.layers.Add()
        #self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.lstm_encoder_3 = tf.keras.layers.LSTM(128, activation='relu',return_sequences=False)
        #self.add_e3 = tf.keras.layers.Add()
        #self.batch_norm_3 = tf.keras.layers.BatchNormalization()
        #self.lstm_encoder_4 = tf.keras.layers.LSTM(32, activation='relu', return_sequences=False,return_state=False)
        #self.batch_norm_4 = tf.keras.layers.BatchNormalization()
        #self.add_e4 = tf.keras.layers.Add()

        # Define Static Encoder Branch
        self.dense_encoder_1 = tf.keras.layers.Dense(32, activation='relu')
        #self.batch_norm_5 = tf.keras.layers.BatchNormalization()
        self.dense_encoder_2 = tf.keras.layers.Dense(24, activation='relu')
        #self.batch_norm_6 = tf.keras.layers.BatchNormalization()
        #self.dense_encoder_3 = tf.keras.layers.Dense(16, activation='relu')
        #self.batch_norm_7 = tf.keras.layers.BatchNormalization()
        #self.dense_encoder_4 = tf.keras.layers.Dense(16, activation='relu')
        #self.batch_norm_8 = tf.keras.layers.BatchNormalization()

        # Concatenate output from two branches
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        
        # Dense layer after concatenation
        #self.dense_concat = tf.keras.layers.Dense(16, activation='relu')

        # Repeat vector
        self.repeat = tf.keras.layers.RepeatVector(self.timesteps)
        
        # Define Temporal Decoder Branch
        #self.decoder0 = tf.keras.layers.LSTM(self.num_temporal_features, activation=None, return_sequences=True)
        self.decoder1 = tf.keras.layers.LSTM(128, activation='relu', return_sequences=True)
        #self.add_d1 = tf.keras.layers.Add()
        self.decoder2 = tf.keras.layers.LSTM(128, activation='relu', return_sequences=True)
        #self.add_d2 = tf.keras.layers.Add()
        #self.decoder3 = tf.keras.layers.LSTM(128, activation='relu', return_sequences=True)
        #self.add_d3 = tf.keras.layers.Add()
        self.decoder4 = tf.keras.layers.LSTM(self.num_temporal_features, activation=None, return_sequences=True)

        # Define Static Decoder Branch
        #self.dense_static_d1 = tf.keras.layers.Dense(16, activation='relu')
        #self.dense_static_d2 = tf.keras.layers.Dense(24, activation='relu')
        self.dense_static_d3 = tf.keras.layers.Dense(32, activation='relu')
        self.dense_static_d4 = tf.keras.layers.Dense(self.num_cat_features,activation='sigmoid')
        self.expand_dims = ExpandDims(timesteps=self.timesteps)

        # Concatenate output from two branches
        self.concat1 = tf.keras.layers.Concatenate(axis=-1)
        
        
    def call(self, x):
        temporal, static = self.seperate_cat_features(x)
        temporal = self.lstm_encoder_1(temporal)
        #temporal = self.batch_norm_1(temporal)
        #temporal = self.add_e1([temporal,temporal_e1])
        
        temporal = self.lstm_encoder_2(temporal)
        #temporal = self.batch_norm_2(temporal)
        #temporal = self.add_e2([temporal,temporal_e2])

        temporal = self.lstm_encoder_3(temporal)
        #temporal = self.batch_norm_3(temporal)
        #temporal = self.add_e3([temporal,temporal_e3])

        #temporal = self.lstm_encoder_4(temporal)
        #temporal = self.batch_norm_4(temporal)
        #temporal = self.add_e4([temporal,temporal_e4])

        static = self.dense_encoder_1(static)
        #static = self.batch_norm_5(static)
        static = self.dense_encoder_2(static)
        #static = self.batch_norm_6(static)
        #static = self.dense_encoder_3(static)
        #static = self.batch_norm_7(static)
        #static = self.dense_encoder_4(static)
        #static = self.batch_norm_8(static)
        concatenated = self.concat([temporal,static])
        #concatenated = self.dense_concat(concatenated)
        temporal = self.repeat(concatenated)
        #temporal = self.decoder0(temporal)
        #print(temporal.shape)
        temporal = self.decoder1(temporal)
        temporal = self.decoder2(temporal)
        #temporal = self.decoder3(temporal)
        temporal = self.decoder4(temporal)
        #static = self.dense_static_d1(concatenated)
        #static = self.dense_static_d2(concatenated)
        static = self.dense_static_d3(concatenated)
        static = self.dense_static_d4(static)
        static = self.expand_dims(static)
        concatenated = self.concat1([temporal,static])
        return concatenated


class VarLSTMAutoencoderV0(tf.keras.Model):
    def __init__(self,**kwargs):
        super(VarLSTMAutoencoderV0, self).__init__()

        # Set up attributes
        self.latent_dim = kwargs["latent_dim"]
        ratio = kwargs["ratio"]
        self.num_stations = kwargs["num_stations"]
        self.prev_days_data = kwargs["prev_days_data"]
        self.num_features = kwargs["num_features"]
        self.batch_size = kwargs["batch_size"]

        # Set up layers
        self.reshape_batch = ReshapeBatch(self.prev_days_data,self.num_features)

        self.encoder = tf.keras.layers.LSTM(self.latent_dim, return_sequences=True, return_state=True)
        self.decoder = tf.keras.layers.LSTM(self.latent_dim, return_sequences=True)
        
    def call(self, inputs, training):
        if training == True:
            inputs = inputs[:,0:random.randint(1,self.num_stations),:,:]

        x = self.reshape_batch(inputs)
        _, state_h, state_c = self.encoder(x)
        reconstructed = self.decoder(inputs, initial_state=[state_h, state_c])
        return reconstructed


class VarLSTMAutoencoderV1(tf.keras.Model):
    def __init__(self,**kwargs):
        super(VarLSTMAutoencoderV1, self).__init__()

        self.feature_number = kwargs["feature_number"]
        self.timesteps = kwargs["prev_days_data"]
        cat_features_number = kwargs["cat_features_number"]
        self.num_stations = kwargs["num_stations"]
        self.batch_size = kwargs["batch_size"]

        self.reshape_batch = ReshapeBatch(self.timesteps,self.feature_number)
        
        self.encoder1 = tf.keras.layers.LSTM(32, activation='tanh',return_sequences=True)
        self.encoder3 = tf.keras.layers.LSTM(24, activation='tanh', return_sequences=False,return_state=True)
        
        self.repeat = tf.keras.layers.RepeatVector(self.timesteps)
        
        self.decoder0 = tf.keras.layers.LSTM(24, activation='tanh', return_sequences=True)
        self.decoder2 = tf.keras.layers.LSTM(32, activation='tanh', return_sequences=True)
        self.decoder3 = tf.keras.layers.LSTM(self.feature_number, activation=None, return_sequences=True)

        self.reshape_batch_inverse = ReshapeBatchInverse(self.timesteps,self.feature_number, self.batch_size)
        
    def call(self, x, training):
        if training == True:
            x = x[:,0:random.randint(1,self.num_stations),:,:]
        x = self.reshape_batch(x)
        x = self.encoder1(x)
        x,h,c = self.encoder3(x)
        x = self.repeat(x)
        x = self.decoder0(x,initial_state=[h,c])
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.reshape_batch_inverse(x)
        return x

class VarLSTMAutoencoder(tf.keras.Model):
    def __init__(self,**kwargs):
        super(VarLSTMAutoencoder, self).__init__()

        self.feature_number = kwargs["feature_number"]
        self.timesteps = kwargs["prev_days_data"]
        self.cat_features_number = kwargs["cat_features_number"]
        self.num_stations = kwargs["num_stations"]
        self.batch_size = kwargs["batch_size"]

        self.reshape_batch = ReshapeBatch(self.timesteps,self.feature_number)
        
        self.encoder1 = tf.keras.layers.LSTM(32, activation='tanh',return_sequences=True)
        self.encoder3 = tf.keras.layers.LSTM(24, activation='tanh', return_sequences=False,return_state=True)

        # Set up reshape LSTM output layer
        self.reshape_lstm_output = ReshapeLSTMOutput(
            self.batch_size,
            lstm_output=24
            )
        # Set up maximum reduction layer
        self.maximum = MaxReduction(axis=1)
        # Repeat for number of stations
        self.repeat_ws = tf.keras.layers.RepeatVector(self.num_stations)
        # Reshape 
        self.reshape_batch_ws = ReshapeBatchWS(24)
        
        self.repeat = tf.keras.layers.RepeatVector(self.timesteps)
        
        self.decoder0 = tf.keras.layers.LSTM(24, activation='tanh', return_sequences=True)
        self.decoder2 = tf.keras.layers.LSTM(32, activation='tanh', return_sequences=True)
        self.decoder3 = tf.keras.layers.LSTM(self.feature_number, activation=None, return_sequences=True)

        self.reshape_batch_inverse = ReshapeBatchInverse(self.timesteps,self.feature_number, self.batch_size)
        
    def call(self, x, training):
        if training == True:
            num_stations = random.randint(1,self.num_stations)
            x = x[:,0:num_stations,:,:]
        x = self.reshape_batch(x)
        x = self.encoder1(x)
        x,h,c = self.encoder3(x)
        x = self.reshape_lstm_output(x)
        x = self.maximum(x)
        if training == True:
            x = tf.repeat(tf.expand_dims(x, axis=1), repeats=num_stations, axis=1)
        else:
            x = self.repeat_ws(x)
        x = self.reshape_batch_ws(x)
        x = self.repeat(x)
        x = self.decoder0(x,initial_state=[h,c])
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.reshape_batch_inverse(x)
        return x
    

class VarLSTMAutoencoder_v2(tf.keras.Model):
    def __init__(self,**kwargs):
        super(VarLSTMAutoencoder_v2, self).__init__()

        self.feature_number = kwargs["feature_number"]
        self.timesteps = kwargs["prev_days_data"]
        self.cat_features_number = kwargs["cat_features_number"]
        self.num_stations = kwargs["num_stations"]
        self.batch_size = kwargs["batch_size"]

        self.reshape_batch = ReshapeBatch(self.timesteps,self.feature_number)
        
        #self.encoder1 = tf.keras.layers.LSTM(512,return_sequences=True)
        self.encoder2 = tf.keras.layers.LSTM(512,return_sequences=True)
        self.encoder3 = tf.keras.layers.LSTM(256,return_sequences=True)
        self.encoder4 = tf.keras.layers.LSTM(128,return_sequences=False,return_state=True)

        self.repeat = tf.keras.layers.RepeatVector(self.timesteps)
        
        self.decoder1 = tf.keras.layers.LSTM(128,return_sequences=True)
        self.decoder2 = tf.keras.layers.LSTM(256,return_sequences=True)
        self.decoder3 = tf.keras.layers.LSTM(512,return_sequences=True)
        #self.decoder4 = tf.keras.layers.LSTM(512,return_sequences=True)
        
        self.dense_output = tf.keras.layers.Dense(self.feature_number)
        self.time_distributed = tf.keras.layers.TimeDistributed(self.dense_output)

        self.reshape_batch_inverse = ReshapeBatchInverse(self.timesteps,self.feature_number, self.batch_size)
        
    def call(self, x, training):
        if training == True:
            x = x[:,0:random.randint(1,self.num_stations),:,:]
        x = self.reshape_batch(x)

        #x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x,h,c = self.encoder4(x)
        x = self.repeat(x)
        x = self.decoder1(x,initial_state=[h,c])
        x = self.decoder2(x)
        x = self.decoder3(x)
        #x = self.decoder4(x)
        x = self.time_distributed(x)
        x = self.reshape_batch_inverse(x)
        return x
    


class TransformerV0(tf.keras.Model):
    def __init__(self,**kwargs):
        super(Transformer, self).__init__()

        # Set up attributes
        ratio = kwargs["ratio"]
        self.num_stations = kwargs["num_stations"]
        self.prev_days_data = kwargs["prev_days_data"]
        self.num_features = kwargs["num_features"]
        self.batch_size = kwargs["batch_size"]
        self.cat_features_number = kwargs["cat_features_number"]

        # set bias for last dense layer
        bias_values = [np.log(1-ratio),np.log(ratio)]

        # Set up layers
        self.reshape_batch = ReshapeBatch(self.prev_days_data,self.num_features)

        self.seperate_cat_features = SeperateCatFeatures(self.cat_features_number)
        self.reshape_and_slice_cat = ReshapeAndSliceStaticFeatures(
            self.batch_size,
            self.cat_features_number
        )

        self.positional_encoding = PositionalEncoding(
            position=self.prev_days_data,
            d_model=self.num_features - self.cat_features_number
        )

        self.batch_inverse = ReshapeBatchInverse(
            self.prev_days_data,
            self.num_features - self.cat_features_number,
            self.batch_size)

        # Set up Transformer layers
        self.transformer_1 = EncoderLayer(d_model=256, num_heads=8, dff=16)
        #self.add_1 = tf.keras.layers.Add()
        self.transformer_2 = EncoderLayer(d_model=256, num_heads=8, dff=16)
        #self.add_2 = tf.keras.layers.Add()
        self.transformer_3 = EncoderLayer(d_model=256, num_heads=8, dff=16)
        #self.add_3 = tf.keras.layers.Add()
        self.transformer_4= EncoderLayer(d_model=256, num_heads=8, dff=16)
        #self.add_4 = tf.keras.layers.Add()
        # self.transformer_5= EncoderLayer(d_model=128, num_heads=8, dff=16)
        # self.add_5 = tf.keras.layers.Add()
        # self.transformer_6= EncoderLayer(d_model=128, num_heads=8, dff=16)
        # self.add_6 = tf.keras.layers.Add()
        # self.transformer_7= EncoderLayer(d_model=128, num_heads=8, dff=16)
        # self.add_7 = tf.keras.layers.Add()
        self.max_across_time = MaxReduction(axis=2)

        self.transformer_aggr= EncoderLayer(d_model=16, num_heads=8, dff=16)

        # Set up reshape LSTM output layer
        # #self.reshape_lstm_output = ReshapeLSTMOutput(
        #     self.batch_size,
        #     lstm_output=int(self.num_features-self.cat_features_number)
        #     )
        
        # Set up maximum reduction layer
        self.maximum = MaxReduction(axis=1)

        # Define Static Branch
        self.dense_static_1 = tf.keras.layers.Dense(32)
        self.dense_static_2 = tf.keras.layers.Dense(16)

        # Concatenate output from two branches
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        # Add Dense layer after concatenation 
        self.dense_concat = tf.keras.layers.Dense(16)

        # Set up dense and softmax layers
        self.dense = tf.keras.layers.Dense(
            2,
            bias_initializer=tf.keras.initializers.Constant(bias_values)
            )
        self.softmax = tf.keras.layers.Softmax()


    def call(self, inputs, training):
        if training == True:
            inputs = inputs[:,0:random.randint(1,self.num_stations),:,:]

        inputs = self.reshape_batch(inputs)
        temporal_features,static_features = self.seperate_cat_features(inputs)
        print("here is len of tem and sta", (temporal_features.shape), (static_features.shape) )
        temporal_features = self.positional_encoding(temporal_features)


        temporal_features = self.batch_inverse(temporal_features)
        temporal_features = self.transformer_1(temporal_features)
        #temporal_features = self.add_1([temporal_features,temporal_features_1])
        temporal_features = self.transformer_2(temporal_features)
        #temporal_features = self.add_2([temporal_features,temporal_features_2])
        temporal_features = self.transformer_3(temporal_features)
        #temporal_features = self.add_3([temporal_features,temporal_features_3])
        temporal_features = self.transformer_4(temporal_features)
        # temporal_features = self.add_4([temporal_features,temporal_features_4])
        # temporal_features_5 = self.transformer_5(temporal_features)
        # temporal_features = self.add_5([temporal_features,temporal_features_5])
        # temporal_features_6 = self.transformer_6(temporal_features)
        # temporal_features = self.add_6([temporal_features,temporal_features_6])
        # temporal_features_7 = self.transformer_7(temporal_features)
        # temporal_features = self.add_7([temporal_features,temporal_features_7])
        temporal_features = self.max_across_time(temporal_features)
        temporal_features = self.transformer_aggr(temporal_features)
        
        #temporal_features = self.reshape_lstm_output(temporal_features)
        temporal_features = self.maximum(temporal_features)
        
        static_features = self.reshape_and_slice_cat(static_features)
        static_features = self.dense_static_1(static_features)
        static_features = self.dense_static_2(static_features)
        
        combined = self.concat([temporal_features,static_features])
        combined = self.dense_concat(combined)
        combined = self.dense(combined)
        combined = self.softmax(combined)
        
        return combined


class Transformer(tf.keras.Model):
    def __init__(self,**kwargs):
        super(Transformer, self).__init__()

        # Set up attributes
        ratio = kwargs["ratio"]
        self.num_stations = kwargs["num_stations"]
        self.prev_days_data = kwargs["prev_days_data"]
        self.num_features = kwargs["num_features"] + 1
        self.batch_size = kwargs["batch_size"]
        self.cat_features_number = kwargs["cat_features_number"]

        # set bias for last dense layer
        bias_values = [np.log(1-ratio),np.log(ratio)]

        # Set up layers
        self.reshape_batch = ReshapeBatch(self.prev_days_data,self.num_features)

        self.seperate_cat_features = SeperateCatFeatures(self.cat_features_number)
        self.reshape_and_slice_cat = ReshapeAndSliceStaticFeatures(
            self.batch_size,
            self.cat_features_number
        )
        
        # Set up Transformer layers
        self.transformer_1 = EncoderLayer(d_model=128, num_heads=8, dff=128)
        #self.add_1 = tf.keras.layers.Add()
        self.transformer_2 = EncoderLayer(d_model=64, num_heads=8, dff=64)
        #self.add_2 = tf.keras.layers.Add()
        self.transformer_3 = EncoderLayer(d_model=32, num_heads=8, dff=32)
        #self.add_3 = tf.keras.layers.Add()
        self.transformer_4= EncoderLayer(d_model=17, num_heads=8, dff=17)
        #self.add_4 = tf.keras.layers.Add()

        self.batch_inverse = ReshapeBatchInverse(
            self.prev_days_data,
            self.num_features - self.cat_features_number,
            self.batch_size)
        
        self.max_across_time = MaxReduction(axis=2)
        #self.sum_across_time = SumReduction(axis=2)
        
        self.transformer_aggr= EncoderLayer(d_model=17, num_heads=8, dff=17)
        # self.add_7 = tf.keras.layers.Add()
        
        
        # Set up maximum reduction layer
        self.maximum = MaxReduction(axis=1)

        # Define Static Branch
        self.dense_static_1 = tf.keras.layers.Dense(32)
        self.dense_static_2 = tf.keras.layers.Dense(17)

        # Concatenate output from two branches
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        # Add Dense layer after concatenation 
        self.dense_concat = tf.keras.layers.Dense(16)

        # Set up dense and softmax layers
        self.dense = tf.keras.layers.Dense(
            2,
            bias_initializer=tf.keras.initializers.Constant(bias_values)
            )
        self.softmax = tf.keras.layers.Softmax()


    def call(self, inputs, training):
        if training == True:
            inputs = inputs[:,0:random.randint(1,self.num_stations),:,:]

        inputs = self.reshape_batch(inputs)
        temporal_features,static_features = self.seperate_cat_features(inputs)
        print("here is len of tem and sta", (temporal_features.shape), (static_features.shape) )
        temporal_features = self.transformer_1(temporal_features)
        #temporal_features = self.add_1([temporal_features,temporal_features_1])
        temporal_features = self.transformer_2(temporal_features)
        #temporal_features = self.add_2([temporal_features,temporal_features_2])
        temporal_features = self.transformer_3(temporal_features)
        #temporal_features = self.add_3([temporal_features,temporal_features_3])
        temporal_features = self.transformer_4(temporal_features)
        #temporal_features = self.add_4([temporal_features,temporal_features_4])

        temporal_features = self.batch_inverse(temporal_features)

        temporal_features = self.max_across_time(temporal_features)
        #temporal_features = self.sum_across_time(temporal_features)
        temporal_features = self.transformer_aggr(temporal_features)
        
        #temporal_features = self.reshape_lstm_output(temporal_features)
        temporal_features = self.maximum(temporal_features)
        
        static_features = self.reshape_and_slice_cat(static_features)
        static_features = self.dense_static_1(static_features)
        static_features = self.dense_static_2(static_features)
        
       # combined = self.concat([temporal_features,static_features])
       # combined = self.dense_concat(combined)
        combined = self.dense(combined)
        combined = self.softmax(combined)
        return combined
    

#@register_keras_serializable()
class TransformerTimeseries(tf.keras.Model):
    def __init__(self,**kwargs):
        super(TransformerTimeseries, self).__init__()

        # Set up attributes
        ratio = kwargs.get("ratio", 1.0)  # <-- set a sensible default
        self.num_stations = kwargs["num_stations"]
        self.prev_days_data = kwargs["prev_days_data"]
        self.num_features = kwargs["num_features"] + 1
        self.batch_size = kwargs["batch_size"]
        self.cat_features_number = kwargs["cat_features_number"]

        # set bias for last dense layer
        bias_values = [np.log(1-ratio),np.log(ratio)]

        # Set up layers
        self.reshape_batch = ReshapeBatch(self.prev_days_data,self.num_features)

        self.seperate_cat_features = SeperateCatFeatures(self.cat_features_number)
        self.reshape_and_slice_cat = ReshapeAndSliceStaticFeatures(
            self.batch_size,
            self.cat_features_number
        )

      
        self.positional_encoding_1=PositionalEncoding(
            position=self.prev_days_data,
            d_model=17
        )
        
        self.transformer_encoder_1 = TransformerEncoder(
            head_size=32,
            num_heads=4,
            ff_dim=32,
            num_features=17,
            dropout=0.0
            )
        
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        

        # Set up reshape LSTM output layer
        self.reshape_lstm_output = ReshapeLSTMOutput(
            self.batch_size,
            lstm_output=self.num_features - self.cat_features_number
            )
        
        
        # Set up maximum reduction layer
        self.maximum = MaxReduction(axis=1)

        # Define Static Branch
        self.dense_static_1 = tf.keras.layers.Dense(32)
        self.dense_static_2 = tf.keras.layers.Dense(17)

        # Concatenate output from two branches
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        # Add Dense layer after concatenation 
        self.dense_concat = tf.keras.layers.Dense(16)

        # Set up dense and softmax layers
        self.dense = tf.keras.layers.Dense(
            2,
            bias_initializer=tf.keras.initializers.Constant(bias_values)
            )
        self.softmax = tf.keras.layers.Softmax()


    def call(self, inputs, training):
       # training= False
        if training == True:
           # inputs= tf.convert_to_tensor(inputs)
            #inputs=tf.reshape(inputs,[1024,3,4,84])
           
            #inputs = inputs[:,0:random.randint(1,self.num_stations),:,:]
            inputs = inputs[:,0:random.randint(1,self.num_stations),:,:]
            print("ninput shape:",inputs.shape)
        #else:
           # inputs= tf.convert_to_tensor(inputs)
           # inputs=tf.reshape(inputs,[1024,3,4,84])
             
        
        inputs = inputs[:,0:random.randint(1,self.num_stations),:,:]
        #print("after if shape:",inputs.shape)
        inputs = self.reshape_batch(inputs)
        # Seperate categorical, continuous weather and continuous link features
        
        temporal_features,static_features = self.seperate_cat_features(inputs)
        print("here is len of tem and sta", (temporal_features.shape), (static_features.shape) )
       # temporal_features = self.positional_encoding_1(temporal_features)
        # Convert tensor to numpy array (if running eagerly)
        

# Heatmap visualization
        start = time.perf_counter()
        # Encode continuous weather and continuous link features
        temporal_features = self.transformer_encoder_1(temporal_features)

        # global pooling
        temporal_features = self.global_pool(temporal_features)
        
        temporal_features = self.reshape_lstm_output(temporal_features)
        temporal_features = self.maximum(temporal_features)
        
        static_features = self.reshape_and_slice_cat(static_features)
        static_features = self.dense_static_1(static_features)
        static_features = self.dense_static_2(static_features)
        print(temporal_features.shape,"tc and st" , static_features.shape)
        combined = self.concat([temporal_features,static_features])
        print(combined.shape)
        combined = self.dense_concat(combined)
        combined = self.dense(combined)
        combined = self.softmax(combined)
        total_sec = time.perf_counter() - start
        print(f"Total training time: {total_sec:.2f} s ({total_sec/60:.2f} min)")
        return combined
    
class TransformerTimeseries_K2(tf.keras.Model):
    def __init__(self,**kwargs):
        super(TransformerTimeseries_K2, self).__init__()

        # Set up attributes
        ratio = kwargs["ratio"]
       # self.num_stations = kwargs["num_stations"]
        self.num_stations =2
        self.prev_days_data = kwargs["prev_days_data"]
        self.num_features = kwargs["num_features"] + 1
        self.batch_size = kwargs["batch_size"]
        self.cat_features_number = kwargs["cat_features_number"]

        # set bias for last dense layer
        bias_values = [np.log(1-ratio),np.log(ratio)]

        # Set up layers
        self.reshape_batch = ReshapeBatch(self.prev_days_data,self.num_features)

        self.seperate_cat_features = SeperateCatFeatures(self.cat_features_number)
        self.reshape_and_slice_cat = ReshapeAndSliceStaticFeatures(
            self.batch_size,
            self.cat_features_number
        )

      
        self.positional_encoding_1=PositionalEncoding(
            position=self.prev_days_data,
            d_model=17
        )
        
        self.transformer_encoder_1 = TransformerEncoder(
            head_size=32,
            num_heads=4,
            ff_dim=32,
            num_features=17,
            dropout=0.0
            )
        
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        

        # Set up reshape LSTM output layer
        self.reshape_lstm_output = ReshapeLSTMOutput(
            self.batch_size,
            lstm_output=self.num_features - self.cat_features_number
            )
        
        
        # Set up maximum reduction layer
        self.maximum = MaxReduction(axis=1)

        # Define Static Branch
        self.dense_static_1 = tf.keras.layers.Dense(32)
        self.dense_static_2 = tf.keras.layers.Dense(17)

        # Concatenate output from two branches
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        # Add Dense layer after concatenation 
        self.dense_concat = tf.keras.layers.Dense(16)

        # Set up dense and softmax layers
        self.dense = tf.keras.layers.Dense(
            2,
            bias_initializer=tf.keras.initializers.Constant(bias_values)
            )
        self.softmax = tf.keras.layers.Softmax()


    def call(self, inputs, training):
       # training= False
        if training == True:
            #inputs= tf.convert_to_tensor(inputs)
           # inputs=tf.reshape(inputs,[1024,3,4,84])
           
            inputs = inputs[:,0:random.randint(1,2),:,:]
           # inputs = inputs[:,0:1,:,:]
            print("ninput shape:",inputs.shape)
        else:
           # inputs= tf.convert_to_tensor(inputs)
            inputs = inputs[:,0:2,:,:]
            # inputs=tf.reshape(inputs,[1024,3,4,84])
             
        
        #inputs = inputs[:,0:random.randint(1,self.num_stations),:,:]
        #print("after if shape:",inputs.shape)
       
        inputs = self.reshape_batch(inputs)
        # Seperate categorical, continuous weather and continuous link features
        
        temporal_features,static_features = self.seperate_cat_features(inputs)
        print("here is len of tem and sta", (temporal_features.shape), (static_features.shape) )
        #temporal_features = self.positional_encoding_1(temporal_features)

        # Encode continuous weather and continuous link features
        temporal_features = self.transformer_encoder_1(temporal_features)

        # global pooling
        temporal_features = self.global_pool(temporal_features)
        
        temporal_features = self.reshape_lstm_output(temporal_features)
        temporal_features = self.maximum(temporal_features)
        
        static_features = self.reshape_and_slice_cat(static_features)
        static_features = self.dense_static_1(static_features)
        static_features = self.dense_static_2(static_features)
        print(temporal_features.shape,"tc and st" , static_features.shape)
        combined = self.concat([temporal_features,static_features])
        print(combined.shape)
        combined = self.dense_concat(combined)
        combined = self.dense(combined)
        combined = self.softmax(combined)
        return combined
class TransformerTimeseries_K1(tf.keras.Model):
    def __init__(self,**kwargs):
        super(TransformerTimeseries_K1, self).__init__()

        # Set up attributes
        ratio = kwargs["ratio"]
       # self.num_stations = kwargs["num_stations"]
        self.num_stations =1
        self.prev_days_data = kwargs["prev_days_data"]
        self.num_features = kwargs["num_features"] + 1
        self.batch_size = kwargs["batch_size"]
        self.cat_features_number = kwargs["cat_features_number"]

        # set bias for last dense layer
        bias_values = [np.log(1-ratio),np.log(ratio)]

        # Set up layers
        self.reshape_batch = ReshapeBatch(self.prev_days_data,self.num_features)

        self.seperate_cat_features = SeperateCatFeatures(self.cat_features_number)
        self.reshape_and_slice_cat = ReshapeAndSliceStaticFeatures(
            self.batch_size,
            self.cat_features_number
        )

      
        self.positional_encoding_1=PositionalEncoding(
            position=self.prev_days_data,
            d_model=17
        )
        
        self.transformer_encoder_1 = TransformerEncoder(
            head_size=32,
            num_heads=4,
            ff_dim=32,
            num_features=17,
            dropout=0.0
            )
        
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        

        # Set up reshape LSTM output layer
        self.reshape_lstm_output = ReshapeLSTMOutput(
            self.batch_size,
            lstm_output=self.num_features - self.cat_features_number
            )
        
        
        # Set up maximum reduction layer
        self.maximum = MaxReduction(axis=1)

        # Define Static Branch
        self.dense_static_1 = tf.keras.layers.Dense(32)
        self.dense_static_2 = tf.keras.layers.Dense(17)

        # Concatenate output from two branches
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        # Add Dense layer after concatenation 
        self.dense_concat = tf.keras.layers.Dense(16)

        # Set up dense and softmax layers
        self.dense = tf.keras.layers.Dense(
            2,
            bias_initializer=tf.keras.initializers.Constant(bias_values)
            )
        self.softmax = tf.keras.layers.Softmax()


    def call(self, inputs, training):
       # training= False
        if training == True:
            #inputs= tf.convert_to_tensor(inputs)
           # inputs=tf.reshape(inputs,[1024,3,4,84])
           
            inputs = inputs[:,0:1,:,:]
           # inputs = inputs[:,0:1,:,:]
            print("ninput shape:",inputs.shape)
        else:
           # inputs= tf.convert_to_tensor(inputs)
            inputs = inputs[:,0:1,:,:]
            # inputs=tf.reshape(inputs,[1024,3,4,84])
             
        
        #inputs = inputs[:,0:random.randint(1,self.num_stations),:,:]
        #print("after if shape:",inputs.shape)
       
        inputs = self.reshape_batch(inputs)
        # Seperate categorical, continuous weather and continuous link features
        
        temporal_features,static_features = self.seperate_cat_features(inputs)
        print("here is len of tem and sta", (temporal_features.shape), (static_features.shape) )
        #temporal_features = self.positional_encoding_1(temporal_features)

        # Encode continuous weather and continuous link features
        temporal_features = self.transformer_encoder_1(temporal_features)

        # global pooling
        temporal_features = self.global_pool(temporal_features)
        
        temporal_features = self.reshape_lstm_output(temporal_features)
        temporal_features = self.maximum(temporal_features)
        
        static_features = self.reshape_and_slice_cat(static_features)
        static_features = self.dense_static_1(static_features)
        static_features = self.dense_static_2(static_features)
        print(temporal_features.shape,"tc and st" , static_features.shape)
        combined = self.concat([temporal_features,static_features])
        print(combined.shape)
        combined = self.dense_concat(combined)
        combined = self.dense(combined)
        combined = self.softmax(combined)
        return combined
           
class TransformerTimeseries_KC(tf.keras.Model):
    def __init__(self,**kwargs):
        super(TransformerTimeseries_KC, self).__init__()

        # Set up attributes
        ratio = kwargs["ratio"]
        self.num_stations = kwargs["num_stations"]
        self.prev_days_data = kwargs["prev_days_data"]
        self.num_features = kwargs["num_features"] + 1
        self.batch_size = kwargs["batch_size"]
        self.cat_features_number = kwargs["cat_features_number"]
        # set bias for last dense layer
        bias_values = [np.log(1-ratio),np.log(ratio)]

        # Set up layers
        self.reshape_batch = ReshapeBatch(self.prev_days_data,self.num_features)

        self.seperate_cat_features = SeperateCatFeatures(self.cat_features_number)
        self.reshape_and_slice_cat = ReshapeAndSliceStaticFeatures(
            self.batch_size,
            35
        )

      
        self.positional_encoding_1=PositionalEncoding(
            position=self.prev_days_data,
            d_model=32
        )
        
        self.transformer_encoder_1 = TransformerEncoder(
            head_size=32,
            num_heads=2,
            ff_dim=128,
            num_features=9,
            dropout=0.1
            )
        
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        

        # Set up reshape LSTM output layer
        self.reshape_lstm_output = ReshapeLSTMOutput(
            self.batch_size,
           # lstm_output=self.num_features - self.cat_features_number
            lstm_output=9
            )
        
        
        # Set up maximum reduction layer
        self.maximum = MaxReduction(axis=1)

        # Define Static Branch
        self.dense_static_1 = tf.keras.layers.Dense(32)
        self.dense_static_2 = tf.keras.layers.Dense(9)

        # Concatenate output from two branches
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        # Add Dense layer after concatenation 
        self.dense_concat = tf.keras.layers.Dense(16)

        # Set up dense and softmax layers
        self.dense = tf.keras.layers.Dense(
            2,
            bias_initializer=tf.keras.initializers.Constant(bias_values)
            )
        self.softmax = tf.keras.layers.Softmax()


    def call(self, inputs, training):
       # training= False
        if training == True:
            #inputs= tf.convert_to_tensor(inputs)
           # inputs=tf.reshape(inputs,[1024,3,4,84])
           
            inputs = inputs[:,0:1,:,:]
            #inputs = inputs[:,0:random.randint(1,self.num_stations),:,:]
            print("ninput shape:",inputs.shape)
      #  else:
           # inputs= tf.convert_to_tensor(inputs)
            # inputs=tf.reshape(inputs,[1024,3,4,84])
             
        
        inputs = inputs[:,0:1,:,:]
        #print("after if shape:",inputs.shape)
        inputs = self.reshape_batch(inputs)
        # Seperate categorical, continuous weather and continuous link features
        
        temporal_features,static_features = self.seperate_cat_features(inputs)
        print("here is len of tem and sta", (temporal_features.shape), (static_features.shape) )
        #temporal_features = self.positional_encoding_1(temporal_features)
        
        temporal_features = temporal_features[:, :, :9] # for selecting only the kpi features
        # Encode continuous weather and continuous link features
        start = time.perf_counter()
    
    
        temporal_features = self.transformer_encoder_1(temporal_features)

        # global pooling
        temporal_features = self.global_pool(temporal_features)
        
        temporal_features = self.reshape_lstm_output(temporal_features)
        temporal_features = self.maximum(temporal_features)
        
       # static_features =static_features[ :, :35] # for selecting only the static features
      #  static_features = self.reshape_and_slice_cat(static_features)
      #  static_features = self.dense_static_1(static_features)
      #  static_features = self.dense_static_2(static_features)
       # print("st" , static_features[0][34])
      #  combined = self.concat([temporal_features,static_features])
        #print(combined.shape)
        #combined = self.dense_concat(combined)
        combined = self.dense_concat(temporal_features)
        combined = self.dense(combined)
        combined = self.softmax(combined)
        total_sec = time.perf_counter() - start
        print(f"Total training time: {total_sec:.2f} s ({total_sec/60:.2f} min)")
        return combined

       
class TransformerTimeseries_X(tf.keras.Model):
    def __init__(self,**kwargs):
        super(TransformerTimeseries_X, self).__init__()

        # Set up attributes
        ratio = kwargs["ratio"]
        self.num_stations = kwargs["num_stations"]
        self.prev_days_data = kwargs["prev_days_data"]
        self.num_features = kwargs["num_features"] + 1
        self.batch_size = kwargs["batch_size"]
        self.cat_features_number = kwargs["cat_features_number"]

        # set bias for last dense layer
        bias_values = [np.log(1-ratio),np.log(ratio)]

        # Set up layers
        self.reshape_batch = ReshapeBatch(self.prev_days_data,self.num_features)

        self.seperate_cat_features = SeperateCatFeatures(self.cat_features_number)
        self.reshape_and_slice_cat = ReshapeAndSliceStaticFeatures(
            self.batch_size,
            self.cat_features_number
        )

      
        self.positional_encoding_1=PositionalEncoding(
            position=self.prev_days_data,
            d_model=17
        )
        
        self.transformer_encoder_1 = TransformerEncoder(
            head_size=32,
            num_heads=4,
            ff_dim=32,
            num_features=17,
            dropout=0.0
            )
        
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        

        # Set up reshape LSTM output layer
        self.reshape_lstm_output = ReshapeLSTMOutput(
            self.batch_size,
            lstm_output=17
            )
        
        
        # Set up maximum reduction layer
        self.maximum = MaxReduction(axis=1)

        # Define Static Branch
        self.dense_static_1 = tf.keras.layers.Dense(32)
        self.dense_static_2 = tf.keras.layers.Dense(17)

        # Concatenate output from two branches
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        # Add Dense layer after concatenation 
        self.dense_concat = tf.keras.layers.Dense(16)

        # Set up dense and softmax layers
        self.dense = tf.keras.layers.Dense(
            2,
            bias_initializer=tf.keras.initializers.Constant(bias_values)
            )
        self.softmax = tf.keras.layers.Softmax()


    def call(self, inputs, training):
       # training= False
        if training == True:
            inputs= tf.convert_to_tensor(inputs)
         #   inputs=tf.reshape(inputs,[1024,3,4,84])
            inputs=tf.reshape(inputs,[1024,3,4,82])
            inputs = inputs[:,0:1,:,:]
        else:
            inputs= tf.convert_to_tensor(inputs)
            inputs=tf.reshape(inputs,[1024,3,4,82])
            inputs = inputs[:,0:1,:,:]
             
        
        #inputs = inputs[:,0:random.randint(1,self.num_stations),:,:]
        inputs = self.reshape_batch(inputs)
        # Seperate categorical, continuous weather and continuous link features
        temporal_features,static_features = self.seperate_cat_features(inputs)
        print("here is len of tem and sta", (temporal_features.shape), (static_features.shape) )
        #temporal_features = self.positional_encoding_1(temporal_features)
       # temporal_features = temporal_features[:, :, :9] 
        # Encode continuous weather and continuous link features
        temporal_features = self.transformer_encoder_1(temporal_features)

        # global pooling
        temporal_features = self.global_pool(temporal_features)
        
        temporal_features = self.reshape_lstm_output(temporal_features)
        temporal_features = self.maximum(temporal_features)
        
        static_features = self.reshape_and_slice_cat(static_features)
        static_features = self.dense_static_1(static_features)
        static_features = self.dense_static_2(static_features)
        print(temporal_features.shape,"tc and st" , static_features.shape)
        combined = self.concat([temporal_features,static_features])
       # print(combined.shape)
        combined = self.dense_concat(combined)
        combined = self.dense(combined)
        combined = self.softmax(combined)
        return combined
class TransformerTimeseries_KCX(tf.keras.Model):
    def __init__(self,**kwargs):
        super(TransformerTimeseries_KCX, self).__init__()

        # Set up attributes
        ratio = kwargs["ratio"]
        self.num_stations = kwargs["num_stations"]
        self.prev_days_data = kwargs["prev_days_data"]
        self.num_features = kwargs["num_features"] + 1
        self.batch_size = kwargs["batch_size"]
        self.cat_features_number = kwargs["cat_features_number"]
        # set bias for last dense layer
        bias_values = [np.log(1-ratio),np.log(ratio)]

        # Set up layers
        self.reshape_batch = ReshapeBatch(self.prev_days_data,self.num_features)

        self.seperate_cat_features = SeperateCatFeatures(self.cat_features_number)
        self.reshape_and_slice_cat = ReshapeAndSliceStaticFeatures(
            self.batch_size,
            35
        )

      
        self.positional_encoding_1=PositionalEncoding(
            position=self.prev_days_data,
            d_model=32
        )
        
        self.transformer_encoder_1 = TransformerEncoder(
            head_size=32,
            num_heads=2,
            ff_dim=128,
            num_features=8,
            dropout=0.0
            )
        
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        

        # Set up reshape LSTM output layer
        self.reshape_lstm_output = ReshapeLSTMOutput(
            self.batch_size,
           # lstm_output=self.num_features - self.cat_features_number
            lstm_output=8
            )
        
        
        # Set up maximum reduction layer
        self.maximum = MaxReduction(axis=1)

        # Define Static Branch
        self.dense_static_1 = tf.keras.layers.Dense(32)
        self.dense_static_2 = tf.keras.layers.Dense(8)

        # Concatenate output from two branches
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        # Add Dense layer after concatenation 
        self.dense_concat = tf.keras.layers.Dense(16)

        # Set up dense and softmax layers
        self.dense = tf.keras.layers.Dense(
            2,
            bias_initializer=tf.keras.initializers.Constant(bias_values)
            )
        self.softmax = tf.keras.layers.Softmax()


    def call(self, inputs, training):
       # training= False
        ft=80
        if training == True:
            inputs= tf.convert_to_tensor(inputs)
            inputs=tf.reshape(inputs,[1024,3,4,ft])
           
            inputs = inputs[:,0:1,:,:]
            #inputs = inputs[:,0:random.randint(1,self.num_stations),:,:]
            print("ninput shape:",inputs.shape)
        else:
             inputs= tf.convert_to_tensor(inputs)
             inputs=tf.reshape(inputs,[1024,3,4,ft])
             
        
        inputs = inputs[:,0:1,:,:]
        #print("after if shape:",inputs.shape)
        inputs = self.reshape_batch(inputs)
        # Seperate categorical, continuous weather and continuous link features
        
        temporal_features,static_features = self.seperate_cat_features(inputs)
        print("here is len of tem and sta", (temporal_features.shape), (static_features.shape) )
        #temporal_features = self.positional_encoding_1(temporal_features)
        temporal_features = temporal_features[:, :, :8] # for selecting only the kpi features
        # Encode continuous weather and continuous link features
        temporal_features = self.transformer_encoder_1(temporal_features)

        # global pooling
        temporal_features = self.global_pool(temporal_features)
        
        temporal_features = self.reshape_lstm_output(temporal_features)
        temporal_features = self.maximum(temporal_features)
        
       # static_features =static_features[ :, :35] # for selecting only the static features
      #  static_features = self.reshape_and_slice_cat(static_features)
      #  static_features = self.dense_static_1(static_features)
      #  static_features = self.dense_static_2(static_features)
       # print("st" , static_features[0][34])
      #  combined = self.concat([temporal_features,static_features])
        #print(combined.shape)
        #combined = self.dense_concat(combined)
        combined = self.dense_concat(temporal_features)
        combined = self.dense(combined)
        combined = self.softmax(combined)
        return combined

class FCN(tf.keras.Model):
    def __init__(self,**kwargs):
        super(FCN, self).__init__()

        # Set up attributes
        ratio = kwargs["ratio"]
        self.num_stations = kwargs["num_stations"]
        self.prev_days_data = kwargs["prev_days_data"]
        self.num_features = kwargs["num_features"] + 1
        self.batch_size = kwargs["batch_size"]
        self.cat_features_number = kwargs["cat_features_number"]

        # set bias for last dense layer
        bias_values = [np.log(1-ratio),np.log(ratio)]

        # Set up layers
        self.reshape_batch = ReshapeBatch(self.prev_days_data,self.num_features)

        self.seperate_cat_features = SeperateCatFeatures(self.cat_features_number)
        self.reshape_and_slice_cat = ReshapeAndSliceStaticFeatures(
            self.batch_size,
            self.cat_features_number
        )

        '''
        self.positional_encoding_1=PositionalEncoding(
            position=self.prev_days_data,
            d_model=self.num_features - self.cat_features_number
        )
        '''
       # Dense layers
        self.dense_1 = tf.keras.layers.Dense(128,activation="relu")
        self.dense_2 = tf.keras.layers.Dense(64,activation="relu")
        self.dense_3 = tf.keras.layers.Dense(self.num_features-self.cat_features_number,activation="relu")

        
        #self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        # Set up reshape LSTM output layer
        self.reshape_lstm_output = ReshapeLSTMOutput(
            self.batch_size,
            lstm_output=int(self.num_features-self.cat_features_number)
            )
        self.reshape_out = ReshapeLSTMOutput(
            self.batch_size,
            lstm_output=int(self.num_features)
            )
        self.maximum = MaxReduction(axis=1)
        # Define Static Branch
        self.dense_static_1 = tf.keras.layers.Dense(64)
        self.dense_static_2 = tf.keras.layers.Dense(17)

        # Concatenate output from two branches
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        # Add Dense layer after concatenation 
        self.dense_concat = tf.keras.layers.Dense(16)

        # Set up dense and softmax layers
        self.dense = tf.keras.layers.Dense(
            2,
            bias_initializer=tf.keras.initializers.Constant(bias_values)
            )
        self.softmax = tf.keras.layers.Softmax()


    def call(self, inputs, training):
        if training == True:
            inputs = inputs[:,0:random.randint(1,self.num_stations),:,:]

        inputs = self.reshape_batch(inputs)
        # Seperate categorical, continuous weather and continuous link features
        temporal_features,static_features = self.seperate_cat_features(inputs)

       # temporal_features = self.positional_encoding_1(temporal_features)

        # Encode continuous weather and continuous link features
        temporal_features = self.dense_1(temporal_features)
        temporal_features = self.dense_2(temporal_features)
        temporal_features = self.dense_3(temporal_features)
      #  temporal_features = self.global_pool(temporal_features)
        
        temporal_features = self.reshape_lstm_output(temporal_features)
       # print("temporal ", temporal_features.shape)
        temporal_features = self.maximum(temporal_features)
        #print("temporal ", temporal_features.shape)
        static_features = self.reshape_and_slice_cat(static_features)
        static_features = self.dense_static_1(static_features)
        static_features = self.dense_static_2(static_features)
       # print("categorical ", static_features.shape)
        combined = self.concat([temporal_features,static_features])
      #  print("combined", combined.shape)
        combined = self.dense_concat(combined)
       # print("Before reshape", combined.shape)
        #combined=self.reshape_out(combined)
       # print("completed reshape", combined.shape)
        combined = self.dense(combined)
        combined = self.softmax(combined)
        return combined

class VarLSTMPlusTransformerV0(tf.keras.Model):
    """
    
    """

    def __init__(self,**kwargs):
        super(VarLSTMPlusTransformerV0, self).__init__()

        # Set up attributes
        ratio = kwargs["ratio"]
        self.num_stations = kwargs["num_stations"]
        self.prev_days_data = kwargs["prev_days_data"]
        self.num_features = kwargs["num_features"]
        self.batch_size = kwargs["batch_size"]
        self.cat_features_number = kwargs["cat_features_number"]

        # set bias for last dense layer
        bias_values = [np.log(1-ratio),np.log(ratio)]

        # Set up layers
        self.reshape_batch = ReshapeBatch(self.prev_days_data,self.num_features)

        self.seperate_cat_features = SeperateCatFeatures(self.cat_features_number)
        self.reshape_and_slice_cat = ReshapeAndSliceStaticFeatures(
            self.batch_size,
            self.cat_features_number
        )

        # Set up LSTM layers
        self.lstm_1 = tf.keras.layers.LSTM(128,return_sequences=True)
        self.lstm_2 = tf.keras.layers.LSTM(64,return_sequences=True)
        self.lstm_3 = tf.keras.layers.LSTM(32,return_sequences=True)
        self.lstm_4 = tf.keras.layers.LSTM(16)

        # Set up reshape LSTM output layer
        self.reshape_lstm_output = ReshapeLSTMOutput(
            self.batch_size,
            lstm_output=int(self.num_features-self.cat_features_number)
            )
        
        self.transformer_aggr = EncoderLayer(d_model=32, num_heads=8, dff=16)
        # Set up maximum reduction layer
        self.maximum = MaxReduction(axis=1)

        # Define Static Branch
        self.dense_static_1 = tf.keras.layers.Dense(32)
        self.dense_static_2 = tf.keras.layers.Dense(16)

        # Concatenate output from two branches
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        # Add Dense layer after concatenation 
        self.dense_concat = tf.keras.layers.Dense(16)

        # Set up dense and softmax layers
        self.dense = tf.keras.layers.Dense(
            2,
            bias_initializer=tf.keras.initializers.Constant(bias_values)
            )
        self.softmax = tf.keras.layers.Softmax()


    def call(self, inputs, training):
        if training == True:
            inputs = inputs[:,0:random.randint(1,self.num_stations),:,:]

        inputs = self.reshape_batch(inputs)
        temporal_features,static_features = self.seperate_cat_features(inputs)
        
        temporal_features = self.lstm_1(temporal_features)
        temporal_features = self.lstm_2(temporal_features)
        temporal_features = self.lstm_3(temporal_features)
        temporal_features = self.lstm_4(temporal_features)
        temporal_features = self.reshape_lstm_output(temporal_features)
        temporal_features = self.transformer_aggr(temporal_features)
        temporal_features = self.maximum(temporal_features)
        
        static_features = self.reshape_and_slice_cat(static_features)
        static_features = self.dense_static_1(static_features)
        static_features = self.dense_static_2(static_features)
        
        combined = self.concat([temporal_features,static_features])
        combined = self.dense_concat(combined)
        combined = self.dense(combined)
        combined = self.softmax(combined)
        return combined

class VarLSTMPlusTransformer(tf.keras.Model):
    """
    
    """

    def __init__(self,**kwargs):
        super(VarLSTMPlusTransformer, self).__init__()

        # Set up attributes
        ratio = kwargs["ratio"]
        self.num_stations = kwargs["num_stations"]
        self.prev_days_data = kwargs["prev_days_data"]
        self.num_features = kwargs["num_features"]
        self.batch_size = kwargs["batch_size"]
        self.cat_features_number = kwargs["cat_features_number"]

        # set bias for last dense layer
        bias_values = [np.log(1-ratio),np.log(ratio)]

        # Set up layers
        self.reshape_batch = ReshapeBatch(self.prev_days_data,self.num_features)

        self.seperate_cat_features = SeperateCatFeatures(self.cat_features_number)
        self.reshape_and_slice_cat = ReshapeAndSliceStaticFeatures(
            self.batch_size,
            self.cat_features_number
        )

        # Set up LSTM layers
        self.lstm_1 = tf.keras.layers.LSTM(128,return_sequences=True)
        self.lstm_2 = tf.keras.layers.LSTM(64,return_sequences=True)
        self.lstm_3 = tf.keras.layers.LSTM(32,return_sequences=True)
        self.lstm_4 = tf.keras.layers.LSTM(16)

        # Set up reshape LSTM output layer
        self.reshape_lstm_output = ReshapeLSTMOutput(
            self.batch_size,
            lstm_output=int(self.num_features-self.cat_features_number)
            )
        
        self.transformer_aggr1 = EncoderLayer(d_model=64, num_heads=8, dff=16)
        self.transformer_aggr2 = EncoderLayer(d_model=64, num_heads=8, dff=16)
        # Set up maximum reduction layer
        self.maximum = MaxReduction(axis=1)

        # Define Static Branch
        self.dense_static_1 = tf.keras.layers.Dense(32)
        self.dense_static_2 = tf.keras.layers.Dense(16)

        # Concatenate output from two branches
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        # Add Dense layer after concatenation 
        self.dense_concat = tf.keras.layers.Dense(16)

        # Set up dense and softmax layers
        self.dense = tf.keras.layers.Dense(
            2,
            bias_initializer=tf.keras.initializers.Constant(bias_values)
            )
        self.softmax = tf.keras.layers.Softmax()


    def call(self, inputs, training):
        if training == True:
            inputs = inputs[:,0:random.randint(1,self.num_stations),:,:]

        inputs = self.reshape_batch(inputs)
        temporal_features,static_features = self.seperate_cat_features(inputs)
        
        temporal_features = self.lstm_1(temporal_features)
        temporal_features = self.lstm_2(temporal_features)
        temporal_features = self.lstm_3(temporal_features)
        temporal_features = self.lstm_4(temporal_features)
        temporal_features = self.reshape_lstm_output(temporal_features)
        temporal_features = self.transformer_aggr1(temporal_features)
        temporal_features = self.transformer_aggr2(temporal_features)
        temporal_features = self.maximum(temporal_features)
        
        static_features = self.reshape_and_slice_cat(static_features)
        static_features = self.dense_static_1(static_features)
        static_features = self.dense_static_2(static_features)
        
        combined = self.concat([temporal_features,static_features])
        combined = self.dense_concat(combined)
        combined = self.dense(combined)
        combined = self.softmax(combined)
        return combined

class ModelManager:
    def __init__(self,**kwargs):
        self.models = {
            'FCN': FCN,
            'SVM0': SVM0,
            'VarLSTM': VarLSTM,
            'VanillaLSTM': VanillaLSTM,
            'LSTMPlus': LSTMPlus,
            'LSTMPlus_KC': LSTMPlus_KC,
            'LSTMPlus2': LSTMPlus2,
            'LSTMPlusExp': LSTMPlusExp,
            'VarLSTMPlus': VarLSTMPlus,
            'ULSTMPlus': ULSTMPlus,
            'LSTMAutoencoder': LSTMAutoencoder,
            'VarLSTMAutoencoder': VarLSTMAutoencoder,
            'LSTMAutoencoderPlus':LSTMAutoencoderPlus,
            'Transformer': Transformer,
            'VarLSTMPlusTransformer': VarLSTMPlusTransformer,
            'TransformerTimeseries': TransformerTimeseries,
            'TransformerTimeseries_X': TransformerTimeseries_X,
            'TransformerTimeseries_KC': TransformerTimeseries_KC,
            'TransformerTimeseries_KCX': TransformerTimeseries_KCX,
            'TransformerTimeseries_K1': TransformerTimeseries_K1,
            'TransformerTimeseries_K2': TransformerTimeseries_K2,
            }

    def get_model(self,name,**kwargs):
        """
        Get the model identified by name
        """
        if name not in self.models:
            raise ValueError(f"Model {name} not found.")
        return self.models[name](**kwargs)




if __name__ == '__main__':
    pass