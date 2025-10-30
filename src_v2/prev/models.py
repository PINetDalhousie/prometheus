from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import numpy as np
import sys
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from report import report_performance_of_trained_model, box_plot, calculate_num_outliers
import math
from sklearn.svm import SVC
from preprocessing import inverse_processing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from balancing import SMOTE_undersampling
from collections import Counter
from preprocessing import preprocessing
from sklearn.preprocessing import MinMaxScaler
from split import *

def get_model(model_name):
    if model_name == "ExtraTree":
        model = ExtraTreeModel()
    elif model_name == "RandomForest":
        model = RandomForest()
    elif model_name == "LGBM":
        model = LGBM()
    elif model_name == "XGB":
        model = XGB()
    elif model_name == "Ensemble":
        model = EnsembleModel()
    elif model_name == "LSTM_AE":
        model = LSTM_AE()
    elif model_name == "LogisticRegression":
        model = Logistic_Regression()
    elif model_name == "VanillaDenseNN":
        model = VanillaDenseNN()
    elif model_name == "SVC":
        model = SupportVectorClassifier()
    elif model_name == "DenseNN":
        model = DenseNN()
    return model

# class DenseNN:
#     def __init__(self):
        

#     def initialize(self):

#     def time_series_cross_validate(self, X, Y):
    
#     def cross_validate(self, X, Y):

#     def fit(self, X_train, Y_train):

#     def predict_proba(self, X_train, pred_threshold):

#     def get_params(self):


class GNN_dense(tf.keras.Model):
    def __init__(self):
        super(vanilla_dense, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='elu')
        self.dense2 = tf.keras.layers.Dense(32, activation='elu')
        self.dense3 = tf.keras.layers.Dense(2, activation='elu')
        self.soft_max = tf.keras.layers.Softmax()

    def call(self, inputs, training=False):
        # split features into numeric radio site and numeric weather features


        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.soft_max(x)
        return x


class vanilla_dense(tf.keras.Model):
    def __init__(self):
        super(vanilla_dense, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='elu')
        self.dense2 = tf.keras.layers.Dense(32, activation='elu')
        self.dense3 = tf.keras.layers.Dense(2, activation='elu')
        self.soft_max = tf.keras.layers.Softmax()

    def call(self, inputs, training=False):     
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.soft_max(x)
        return x

class VanillaDenseNN:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.time = None
        self.batch_size = None
        self.epochs = None
        self.learning_rate = None
        self.X_val = None
        self.Y_val = None
        self.X_train = None
        self.Y_train = None
        self.save_model = False
        self.pred_threshold = None
        self.loss_function_name = 'sparse_categorical_cross_entropy'
        self.loss_function = None
        self.global_step = 0
        self.optimizer = None

        self.undersample = None
        self.numerical_features = None
        self.categorical_features = None
        self.val_size = None
        

    def initialize(self):
        self.model = vanilla_dense()

    def define_opt_and_loss(self):
        # define the optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        # define the loss function
        if self.loss_function_name == 'sparse_categorical_cross_entropy':
            self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

    def f1_score_calc_step(self, epoch):
        # report performance of trained model on train data
        train_report = report_performance_of_trained_model(self, "VanillaDenseNN", self.X_train, self.Y_train, 
            self.pred_threshold, self.pred_threshold, mode='train')
        with self.train_summary_writer.as_default():
            tf.summary.scalar('f1_score', train_report['macro avg']['f1-score'], step=(epoch+1))

        # report performance of trained model on validation data
        validation_report = report_performance_of_trained_model(self, "VanillaDenseNN", self.X_val, self.Y_val, 
            self.pred_threshold, self.pred_threshold, mode='validation')
        with self.val_summary_writer.as_default():
            tf.summary.scalar('f1_score', validation_report['macro avg']['f1-score'], step=(epoch+1))

    def train_step(self, epoch, dataset, train_loss_metric):
        # Iterate over the batches of the dataset.
        for step, (batch_input, batch_labels) in enumerate(dataset):
            with tf.GradientTape() as tape:
                predictions = self.model(batch_input, training=True)
                loss = self.loss_function(batch_labels, predictions)
            # update the gradients
            grads = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

            
            train_loss_metric.update_state(loss)
            # log the loss metric in tensorboard 
            if (step % math.floor((self.X_train.shape[0]/self.batch_size)/4)) == 0:
                print("step %d: mean train loss = %.4f" % ((step+1), train_loss_metric.result()))
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss_metric.result(), step=self.global_step)
            self.global_step += 1
        return train_loss_metric

    def validation_step(self, epoch, dataset, val_loss_metric):
        # Iterate over the batches of the dataset.
        for step, (batch_input, batch_labels) in enumerate(dataset):
            predictions = self.model(batch_input)
            loss = self.loss_function(batch_labels, predictions)
            val_loss_metric.update_state(loss)

        # log the loss metric in tensorboard
        print("step %d: mean validation loss = %.4f" % (step, val_loss_metric.result()))
        with self.val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss_metric.result(), step=self.global_step)
        return val_loss_metric


    def fit(self, X_train, Y_train):
        # set optimizer and loss function
        self.define_opt_and_loss()

        # define the logging metrics
        train_loss_metric = tf.keras.metrics.Mean()
        val_loss_metric = tf.keras.metrics.Mean()

        # turn training data into dataset object
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024, seed=42,
            reshuffle_each_iteration=True).batch(self.batch_size, drop_remainder=True)

        # log metrics in tensorboard
        #train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        self.train_summary_writer = tf.summary.create_file_writer(f"../report/tensorboard/{self.time}/train/")
        self.val_summary_writer = tf.summary.create_file_writer(f"../report/tensorboard/{self.time}/val/")

        # train the model
        # Iterate over epochs.
        for epoch in range(self.epochs):
            print("Start of epoch %d" % (epoch,))
            train_loss_metric  = self.train_step(epoch, train_dataset, train_loss_metric)
            with self.train_summary_writer.as_default():
                tf.summary.scalar('epoch_loss', train_loss_metric.result(), step=(epoch+1))
            # print out the model summary
            if epoch==0:
                print(self.model.summary())


            # prepare validation dataset
            val_dataset = tf.data.Dataset.from_tensor_slices((self.X_val,self.Y_val))
            val_dataset = val_dataset.batch(self.batch_size, drop_remainder=False)
            val_loss_metric = self.validation_step(epoch, val_dataset, val_loss_metric)

            with self.val_summary_writer.as_default():
                tf.summary.scalar('epoch_loss', val_loss_metric.result(), step=(epoch+1))

            # reset logging metrics after each epoch
            #train_loss_metric.reset_states()
            #val_loss_metric.reset_states()

            if self.save_model:
                if epoch % 3 == 0:
                    self.model.save(f'../models/{self.time}/')

            # calculate f1 score on train and validation data
            if (epoch+1) % 20 == 0:
                print('calculating f1 scores on trained model .....')
                self.f1_score_calc_step(epoch)
        
        if self.save_model:
            self.model.save(f'../models/{self.time}/')

    def predict_proba(self, X_train, pred_threshold):
        dataset = tf.data.Dataset.from_tensor_slices(X_train)
        dataset = dataset.batch(self.batch_size, drop_remainder=False)

        prediction_list = []
        pred_binary = []
        for _, batch_input in enumerate(dataset):
            # preprocess input based on model type
            predictions = self.model(batch_input)
            binary_predictions = tf.math.argmax(predictions, 1)
            prediction_list.extend(list(predictions[:,1]))
            pred_binary.extend(list(binary_predictions))

        # turn list into numpy array
        prediction_list = np.array(prediction_list)
        pred_binary = np.array(pred_binary)
    
        return prediction_list, pred_binary

    def get_params(self):
        print(self.model.to_json())

class SupportVectorClassifier:
    def __init__(self):
        self.model = None
        self.time = None

        self.X_val = None
        self.Y_val = None
        self.X_train = None
        self.Y_train = None

        self.y_pred_proba = None
        self.undersample = None
        self.model_name = None
        self.numerical_features = None
        self.categorical_features = None
        self.pred_threshold = None

    def initialize(self):
        self.model = SVC(gamma='auto', probability=True)

    def time_series_cross_validate(self, X, Y):
        tkf = TemporalKFold(X, Y, self.val_size)
        
        counter = 0
        for i in range(5):
            X_train, X_val, Y_train, Y_val = tkf.get_split(i)

            print(f"counter y train after tkfold {Counter(Y_train)}")
            print(f"counter y test after tkfold {Counter(Y_val)}")

            X_train, Y_train = SMOTE_undersampling(X_train, Y_train, minority_ratio = self.undersample)
            print(f"after undersampling x.shape: {X_train.shape}")
            print(Counter(Y_train))

            # feature scaling and encoding
            min_max_scalar = MinMaxScaler()
            min_max_scalar.fit(X_train)
            X_train = min_max_scalar.transform(X_train)
            print(f"after encoding X_train.shape: {X_train.shape}")

            self.model.fit(X_train, Y_train)

            min_to_major_ratio = Counter(Y_train)[1] / Counter(Y_train)[0]
            recons_prctl_thresh = (100 - min_to_major_ratio)

            # report performance of trained model on train data
            _ = report_performance_of_trained_model(self, self.model_name, X_train, Y_train, 
                recons_prctl_thresh, self.pred_threshold, mode='train')

            ####### Validation Steps #######
            X_val = min_max_scalar.transform(X_val)

            # report performance of trained model on validation data
            _ = report_performance_of_trained_model(self, self.model_name, X_val, Y_val, 
                recons_prctl_thresh, self.pred_threshold, mode='validation')

            # sample non failure validation cases to match the number of failure cases
            # df = pd.DataFrame(data={'y_val':Y_val, 'y_pred_proba':self.y_pred_proba})
            # num_failures = len(df[df['y_val']==1].index)
            # df_non_failure = df[df['y_val']==0].sample(n=num_failures, random_state=42)
            # df = pd.concat([df[df['y_val']==1],df_non_failure])

            # print(f"failure cases {len(df[df['y_val']==1].index)}")
            # print(f"non failure cases {len(df[df['y_val']==0].index)}")

            # # get roc auc
            # self.get_roc_auc(df['y_val'], df['y_pred_proba'], id=counter)

            counter += 1

    def cross_validate(self, X, Y):
        X = X.to_numpy()

        # define stratifiedkfold where majority minority class ratios are maintained
        # in each fold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        counter = 0
        for train_index, test_index in skf.split(X, Y):
            X_train, X_val = X[train_index], X[test_index]
            Y_train, Y_val = Y[train_index], Y[test_index]

            print(f"counter y train after kfold {Counter(Y_train)}")
            print(f"counter y test after kfold {Counter(Y_val)}")

            X_train, Y_train = SMOTE_undersampling(X_train, Y_train, minority_ratio = self.undersample)
            print(f"after undersampling x.shape: {X_train.shape}")
            print(Counter(Y_train))

            # feature scaling and encoding
            min_max_scalar = MinMaxScaler()
            min_max_scalar.fit(X_train)
            X_train = min_max_scalar.transform(X_train)
            print(f"after encoding X_train.shape: {X_train.shape}")

            self.model.fit(X_train, Y_train)

            min_to_major_ratio = Counter(Y_train)[1] / Counter(Y_train)[0]
            recons_prctl_thresh = (100 - min_to_major_ratio)

            # report performance of trained model on train data
            _ = report_performance_of_trained_model(self, self.model_name, X_train, Y_train, 
                recons_prctl_thresh, self.pred_threshold, mode='train')

            ####### Validation Steps #######
            X_val = min_max_scalar.transform(X_val)

            # report performance of trained model on validation data
            _ = report_performance_of_trained_model(self, self.model_name, X_val, Y_val, 
                recons_prctl_thresh, self.pred_threshold, mode='validation')

            # sample non failure validation cases to match the number of failure cases
            df = pd.DataFrame(data={'y_val':Y_val, 'y_pred_proba':self.y_pred_proba})
            num_failures = len(df[df['y_val']==1].index)
            df_non_failure = df[df['y_val']==0].sample(n=num_failures, random_state=42)
            df = pd.concat([df[df['y_val']==1],df_non_failure])

            # get roc auc
            self.get_roc_auc(df['y_val'], df['y_pred_proba'], id=counter)

            counter += 1
        
    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict_proba(self, X_train, pred_threshold):
        # we take second column as it corresponds the probability
        # of sample belonging to failure class
        pred_proba = self.model.predict_proba(X_train)[:,1]
        pred_binary = np.where(pred_proba >= pred_threshold, 1, 0)

        # save pobability predictions to calculate auc score and create roc curve
        self.y_pred_proba = pred_proba

        return pred_proba, pred_binary

    def get_params(self):
        print("Logistic Regression Model Parameters")
        print(self.model.get_params())

    def get_roc_auc(self, Y_val, y_pred_proba, id):
        # Calculate the False Positive Rate (FPR) and True Positive Rate (TPR)
        fpr, tpr, _ = roc_curve(Y_val, y_pred_proba, pos_label=1)

        # save false positive rate and true positive rate values
        df = pd.DataFrame(data={'fpr':fpr, 'tpr':tpr})
        df.to_csv(f'../report/tensorboard/{self.time}/fpr_tpr_{id}.csv')

        # Plot the ROC curve
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.savefig(f'../report/tensorboard/{self.time}/roc_curve_{id}.png')
        plt.close()

        # Calculate the AUC (Area Under the Curve)
        auc = roc_auc_score(Y_val, y_pred_proba)
        print('AUC: ', auc)

class Logistic_Regression:
    def __init__(self):
        self.model = None
        self.time = None
        
        self.X_val = None
        self.Y_val = None
        self.X_train = None
        self.Y_train = None

        self.y_pred_proba = None
        self.undersample = None
        self.model_name = None
        self.numerical_features = None
        self.categorical_features = None
        self.pred_threshold = None
        self.val_size = None

    def initialize(self):
        self.model = LogisticRegression(max_iter=1000,random_state=42)
    
    def time_series_cross_validate(self, X, Y):
        tkf = TemporalKFold(X, Y, self.val_size)
        
        counter = 0
        for i in range(5):
            X_train, X_val, Y_train, Y_val = tkf.get_split(i)

            print(f"counter y train after tkfold {Counter(Y_train)}")
            print(f"counter y test after tkfold {Counter(Y_val)}")

            X_train, Y_train = SMOTE_undersampling(X_train, Y_train, minority_ratio = self.undersample)
            print(f"after undersampling x.shape: {X_train.shape}")
            print(Counter(Y_train))

            # feature scaling and encoding
            min_max_scalar = MinMaxScaler()
            min_max_scalar.fit(X_train)
            X_train = min_max_scalar.transform(X_train)
            print(f"after encoding X_train.shape: {X_train.shape}")

            self.model.fit(X_train, Y_train)

            min_to_major_ratio = Counter(Y_train)[1] / Counter(Y_train)[0]
            recons_prctl_thresh = (100 - min_to_major_ratio)

            # report performance of trained model on train data
            _ = report_performance_of_trained_model(self, self.model_name, X_train, Y_train, 
                recons_prctl_thresh, self.pred_threshold, mode='train')

            ####### Validation Steps #######
            X_val = min_max_scalar.transform(X_val)

            # report performance of trained model on validation data
            _ = report_performance_of_trained_model(self, self.model_name, X_val, Y_val, 
                recons_prctl_thresh, self.pred_threshold, mode='validation')

            # sample non failure validation cases to match the number of failure cases
            # df = pd.DataFrame(data={'y_val':Y_val, 'y_pred_proba':self.y_pred_proba})
            # num_failures = len(df[df['y_val']==1].index)
            # df_non_failure = df[df['y_val']==0].sample(n=num_failures, random_state=42)
            # df = pd.concat([df[df['y_val']==1],df_non_failure])

            # print(f"failure cases {len(df[df['y_val']==1].index)}")
            # print(f"non failure cases {len(df[df['y_val']==0].index)}")

            # get roc auc
            #self.get_roc_auc(df['y_val'], df['y_pred_proba'], id=counter)

            counter += 1

    def cross_validate(self, X, Y):
        X = X.to_numpy()

        # define stratifiedkfold where majority minority class ratios are maintained
        # in each fold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        counter = 0
        for train_index, test_index in skf.split(X, Y):
            X_train, X_val = X[train_index], X[test_index]
            Y_train, Y_val = Y[train_index], Y[test_index]

            print(f"counter y train after kfold {Counter(Y_train)}")
            print(f"counter y test after kfold {Counter(Y_val)}")

            X_train, Y_train = SMOTE_undersampling(X_train, Y_train, minority_ratio = self.undersample)
            print(f"after undersampling x.shape: {X_train.shape}")
            print(Counter(Y_train))

            # feature scaling and encoding
            min_max_scalar = MinMaxScaler()
            min_max_scalar.fit(X_train)
            X_train = min_max_scalar.transform(X_train)
            print(f"after encoding X_train.shape: {X_train.shape}")

            self.model.fit(X_train, Y_train)

            min_to_major_ratio = Counter(Y_train)[1] / Counter(Y_train)[0]
            recons_prctl_thresh = (100 - min_to_major_ratio)

            # report performance of trained model on train data
            _ = report_performance_of_trained_model(self, self.model_name, X_train, Y_train, 
                recons_prctl_thresh, self.pred_threshold, mode='train')

            ####### Validation Steps #######
            X_val = min_max_scalar.transform(X_val)

            # report performance of trained model on validation data
            _ = report_performance_of_trained_model(self, self.model_name, X_val, Y_val, 
                recons_prctl_thresh, self.pred_threshold, mode='validation')

            # sample non failure validation cases to match the number of failure cases
            df = pd.DataFrame(data={'y_val':Y_val, 'y_pred_proba':self.y_pred_proba})
            num_failures = len(df[df['y_val']==1].index)
            df_non_failure = df[df['y_val']==0].sample(n=num_failures, random_state=42)
            df = pd.concat([df[df['y_val']==1],df_non_failure])

            print(f"failure cases {len(df[df['y_val']==1].index)}")
            print(f"non failure cases {len(df[df['y_val']==0].index)}")

            # get roc auc
            self.get_roc_auc(df['y_val'], df['y_pred_proba'], id=counter)

            counter += 1

    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)


    def predict_proba(self, X_train, pred_threshold):
        # we take second column as it corresponds the probability
        # of sample belonging to failure class
        pred_proba = self.model.predict_proba(X_train)[:,1]
        pred_binary = np.where(pred_proba >= pred_threshold, 1, 0)

        # save pobability predictions to calculate auc score and create roc curve
        self.y_pred_proba = pred_proba

        return pred_proba, pred_binary

    def get_roc_auc(self, Y_val, y_pred_proba, id):
        # Calculate the False Positive Rate (FPR) and True Positive Rate (TPR)
        fpr, tpr, _ = roc_curve(Y_val, y_pred_proba, pos_label=1, drop_intermediate=False)

        # save false positive rate and true positive rate values
        df = pd.DataFrame(data={'fpr':fpr, 'tpr':tpr})
        df.to_csv(f'../report/tensorboard/{self.time}/fpr_tpr_{id}.csv')

        # Plot the ROC curve
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.savefig(f'../report/tensorboard/{self.time}/roc_curve_{id}.png')
        plt.close()

        # Calculate the AUC (Area Under the Curve)
        auc = roc_auc_score(Y_val, y_pred_proba)
        print('AUC: ', auc)


    def get_params(self):
        print("Logistic Regression Model Parameters")
        print(self.model.get_params())

class LSTM_AE:
    def __init__(self):
        # parameters to change for the model
        # is also used to perform data preprocessing for the model
        self.batch_size = None
        self.epochs = None
        self.num_features = None
        self.learning_rate = None
        self.calculate_threshold_flag = True
        self.reconstruction_threshold = None
        self.X_train = None
        self.Y_train = None
        self.resume_training = False
        self.save_model = True
        self.X_val = None
        self.Y_val = None
        self.pred_threshold = None
        self.model = None
        self.reconstruction_losses = None
        self.loss_function_name = 'mae'
        # threshold method to use; can be 'std' or 'percentile' based
        self.threshold_method = None
        self.optimizer = None
        self.loss_function = None
        self.global_step = 0
        self.time = None
        self.model_type = 'DenseAE'
        self.threshold_reduce = None

    def initialize(self):
        if self.resume_training:
            self.model = tf.keras.models.load_model('../models/')
        else:
            if self.model_type == 'LSTMAE':
                self.model = lstm_ae_model(self.num_features)
            elif self.model_type == 'DenseAE':
                self.model = dense_ae_model(self.num_features)
    
    def filter_normal_links(self, X, Y):
        X = pd.DataFrame(X)
        # filter only non-failure links
        X['1-day-predict'] = Y
        X = X[X['1-day-predict'] == 0]
        X = X.drop(['1-day-predict'],axis=1)
        X = X.to_numpy()

        # enable box plot 
        # box_plot(pd.DataFrame(X), table_name='train')
        # calculate_num_outliers(pd.DataFrame(X), table_name='train')
        # print(asd)
        return X


    def define_opt_and_loss(self):
        # define the optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        # define the loss function
        if self.loss_function_name == 'mae':
            self.loss_function = tf.keras.losses.MeanAbsoluteError()
        elif self.loss_function_name == 'mse':
            self.loss_function = tf.keras.losses.MeanSquaredError()

    def preprocess_input(self, batch_input):
        if self.model_type == 'DenseAE':
            batch_input = tf.squeeze(batch_input)
            if tf.rank(batch_input).numpy() == 1:
                batch_input = tf.expand_dims(batch_input, axis=0)
        return batch_input

    def train_step(self, epoch, dataset, train_loss_metric, train_eval_metric):
        # Iterate over the batches of the dataset.
        for step, batch_input in enumerate(dataset):
            # preprocess input based on model type
            batch_input = self.preprocess_input(batch_input)
            with tf.GradientTape() as tape:
                reconstructed = self.model(batch_input, training=True)
                batch_input = tf.squeeze(batch_input)
                loss = self.loss_function(batch_input, reconstructed)
                train_eval_metric.update_state(batch_input, reconstructed)
            
            # update the gradients
            grads = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

            train_loss_metric.update_state(loss)
            
            # log the loss metric in tensorboard 
            if (step % math.floor((self.X_train.shape[0]/self.batch_size)/4)) == 0:
                print("step %d: mean train loss = %.4f" % ((step+1), train_loss_metric.result()))
                print("step %d: mean absolute train eval loss = %.4f" % ((step+1), train_eval_metric.result()))
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss_metric.result(), step=self.global_step)
                    tf.summary.scalar('eval_metric', train_eval_metric.result(), step=self.global_step)

            self.global_step += 1
            
        return train_loss_metric, train_eval_metric


    def validation_step(self, epoch, dataset, val_loss_metric, val_eval_metric):
        # Iterate over the batches of the dataset.
        for step, batch_input in enumerate(dataset):
            batch_input = self.preprocess_input(batch_input)
            reconstructed = self.model(batch_input)
            batch_input = tf.squeeze(batch_input)

            loss = self.loss_function(batch_input, reconstructed)
            val_eval_metric.update_state(batch_input, reconstructed)
            val_loss_metric.update_state(loss)

        # log the loss metric in tensorboard
        print("step %d: mean validation loss = %.4f" % (step, val_loss_metric.result()))
        print("step %d: mean absolute validation eval loss = %.4f" % (step, val_eval_metric.result()))
        with self.val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss_metric.result(), step=self.global_step)
            tf.summary.scalar('eval_metric', val_eval_metric.result(), step=self.global_step)
        
        return val_loss_metric, val_eval_metric

    def f1_score_calc_step(self, epoch):
        # report performance of trained model on train data
        train_report = report_performance_of_trained_model(self, "LSTM_AE", self.X_train, self.Y_train, 
            self.pred_threshold, self.pred_threshold, mode='train')
        with self.train_summary_writer.as_default():
            tf.summary.scalar('f1_score', train_report['macro avg']['f1-score'], step=(epoch+1))

        # report performance of trained model on validation data
        validation_report = report_performance_of_trained_model(self, "LSTM_AE", self.X_val, self.Y_val, 
            self.pred_threshold, self.pred_threshold, mode='validation')
        with self.val_summary_writer.as_default():
            tf.summary.scalar('f1_score', validation_report['macro avg']['f1-score'], step=(epoch+1))
        

    def fit(self, X_train, Y_train):
        # keep normal links only from train data
        X_train = self.filter_normal_links(X_train, Y_train)

        # resize to match lstm input shape
        X_train = np.reshape(X_train, (X_train.shape[0],1,X_train.shape[1]))
        
        # set optimizer and loss function
        self.define_opt_and_loss()

        # define the logging metrics
        train_eval_metric = tf.keras.metrics.MeanAbsoluteError()
        val_eval_metric = tf.keras.metrics.MeanAbsoluteError()
        train_loss_metric = tf.keras.metrics.Mean()
        val_loss_metric = tf.keras.metrics.Mean()

        # turn training data into dataset object
        train_dataset = tf.data.Dataset.from_tensor_slices(X_train)
        train_dataset = train_dataset.shuffle(buffer_size=1024, seed=42,
            reshuffle_each_iteration=True).batch(self.batch_size, drop_remainder=True)

        # log metrics in tensorboard
        #train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        self.train_summary_writer = tf.summary.create_file_writer(f"../report/tensorboard/{self.time}/train/")
        self.val_summary_writer = tf.summary.create_file_writer(f"../report/tensorboard/{self.time}/val/")

        # train the model
        # Iterate over epochs.
        for epoch in range(self.epochs):
            print("Start of epoch %d" % (epoch,))

            train_loss_metric, train_eval_metric = self.train_step(epoch, train_dataset, train_loss_metric, train_eval_metric)
            
            with self.train_summary_writer.as_default():
                tf.summary.scalar('epoch_loss', train_loss_metric.result(), step=(epoch+1))
            
            # print out the model summary
            if epoch==0:
                print(self.model.summary())
                

            # keep normal links only from validation data
            X_val = self.filter_normal_links(self.X_val, self.Y_val)
            # resize to match lstm input shape
            X_val = np.reshape(X_val, (X_val.shape[0],1,X_val.shape[1]))
            # prepare validation dataset
            val_dataset = tf.data.Dataset.from_tensor_slices(X_val)
            val_dataset = val_dataset.batch(self.batch_size, drop_remainder=False)
            val_loss_metric, val_eval_metric = self.validation_step(epoch, val_dataset, val_loss_metric, val_eval_metric)

            with self.val_summary_writer.as_default():
                tf.summary.scalar('epoch_loss', val_loss_metric.result(), step=(epoch+1))

            # reset logging metrics after each epoch
            train_loss_metric.reset_states()
            val_loss_metric.reset_states()
            train_eval_metric.reset_states()
            val_eval_metric.reset_states()

            if self.save_model:
                if epoch % 3 == 0:
                    self.model.save(f'../models/{self.time}/')

            # calculate f1 score on train and validation data
            if (epoch+1) % 20 == 0:
                print('calculating f1 scores on trained model .....')
                self.calculate_threshold()
                self.f1_score_calc_step(epoch)
        
        self.calculate_threshold()
        if self.save_model:
            self.model.save(f'../models/{self.time}/')
        

    def calculate_threshold(self):
        # get loss for normal links in train data using a trained model
        X_train = self.filter_normal_links(self.X_train, self.Y_train)
        X_train = np.reshape(X_train, (X_train.shape[0],1,X_train.shape[1]))
        dataset = tf.data.Dataset.from_tensor_slices(X_train)
        dataset = dataset.batch(self.batch_size, drop_remainder=False)
        loss_list = []
        for _, batch_input in enumerate(dataset):
            # preprocess input based on model type
            batch_input = self.preprocess_input(batch_input)
            reconstructed = self.model(batch_input)
            batch_input = tf.squeeze(batch_input)
            loss = tf.math.abs(batch_input - reconstructed)
            if self.threshold_reduce == 'sum':
                loss = tf.math.reduce_sum(loss, axis=1).numpy().tolist()
            elif self.threshold_reduce == 'mean':
                loss = tf.math.reduce_mean(loss, axis=1).numpy().tolist()
            loss_list.extend(loss)
        # calculate reconstruction threshold from reconstruction losses 
        loss_list = np.array(loss_list)
        print(f"mean reconstruction loss over train data {np.mean(loss_list)}")
        if self.threshold_method == 'percentile':
            self.reconstruction_threshold = np.percentile(loss_list, self.pred_threshold)
        elif self.threshold_method == 'std':
            self.reconstruction_threshold = np.mean(loss_list) + np.std(loss_list)
        print(f"reconstruction threshold using {self.threshold_method} calculated to be {self.reconstruction_threshold}")


    def predict_proba(self, X_train, pred_threshold):
        X_train = np.reshape(X_train, (X_train.shape[0],1,X_train.shape[1]))
        dataset = tf.data.Dataset.from_tensor_slices(X_train)
        dataset = dataset.batch(self.batch_size, drop_remainder=False)

        loss_list = []
        for _, batch_input in enumerate(dataset):
            # preprocess input based on model type
            batch_input = self.preprocess_input(batch_input)
            reconstructed = self.model(batch_input)
            batch_input = tf.squeeze(batch_input)

            loss = tf.math.abs(batch_input - reconstructed)
            if self.threshold_reduce == 'sum':
                loss = tf.math.reduce_sum(loss, axis=1).numpy().tolist()
            elif self.threshold_reduce == 'mean':
                loss = tf.math.reduce_mean(loss, axis=1).numpy().tolist()
            loss_list.extend(loss)

        # turn list into numpy array
        loss_list = np.array(loss_list)
        pred_binary = np.where(loss_list >= self.reconstruction_threshold, 1, 0)
    
        return loss_list, pred_binary
        
    def get_params(self):
        print(self.model.to_json())

    def plot_reconstruction_hist(self,mode):
        # plot reconstruction histogram
        plt.hist(self.reconstruction_losses, bins='auto')
        plt.savefig(f'../report/reconstruction_hist_{mode}.png')
        plt.close()

class dense_ae_model(tf.keras.Model):
    def __init__(self, num_features):
        super(dense_ae_model, self).__init__()
        self.dense2 = tf.keras.layers.Dense(90, activation='elu')
        #self.dense3 = tf.keras.layers.Dense(70, activation='elu')
        self.dense4 = tf.keras.layers.Dense(40, activation='elu')
        #self.batch_norm1 = tf.keras.layers.BatchNormalization()
        #self.dense5 = tf.keras.layers.Dense(70, activation='elu')
        self.dense6 = tf.keras.layers.Dense(90, activation='elu')
        self.dense8 = tf.keras.layers.Dense(num_features, activation='elu')

    def call(self, inputs, training=False):     
        x = self.dense2(inputs)
        #x = self.dense3(x)
        x = self.dense4(x)
        #x = self.dense5(x)
        #x = self.dense12(x)
        x = self.dense6(x)
        #x = self.dense7(x)
        x = self.dense8(x)
        return x

class lstm_ae_model(tf.keras.Model):
    def __init__(self, num_features):
        super(lstm_ae_model, self).__init__()
        self.encoder_lstm1 = tf.keras.layers.LSTM(90,activation='elu',return_sequences=True)
        #self.encoder_lstm2 = tf.keras.layers.LSTM(70,activation='elu',return_sequences=True)  
        self.encoder_lstm3 = tf.keras.layers.LSTM(60,activation='elu',return_sequences=True)
        #self.decoder_lstm1 = tf.keras.layers.LSTM(70,activation='elu',return_sequences=True)
        self.decoder_lstm2 = tf.keras.layers.LSTM(90,activation='elu',return_sequences=True)
        self.decoder_lstm3 = tf.keras.layers.LSTM(num_features, activation='elu')

    def call(self, inputs, training=False):
        x = self.encoder_lstm1(inputs)
        #x = self.encoder_lstm2(x)
        x = self.encoder_lstm3(x)
        #x = self.decoder_lstm1(x)
        x = self.decoder_lstm2(x)
        x = self.decoder_lstm3(x)
        return x


class EnsembleModel:
    def __init__(self):
        self.rf = RandomForest()
        self.lgbm = LGBM()
        self.xgb = XGB()

        self.model = None
        self.time = None
        
        self.X_val = None
        self.Y_val = None
        self.X_train = None
        self.Y_train = None

        self.y_pred_proba = None
        self.undersample = None
        self.model_name = None
        self.numerical_features = None
        self.categorical_features = None
        self.pred_threshold = None

    def initialize(self):
        self.rf.initialize()
        self.lgbm.initialize()
        self.xgb.initialize()

    def fit(self, X_train, Y_train):
        self.rf.fit(X_train, Y_train)
        self.lgbm.fit(X_train, Y_train)
        self.xgb.fit(X_train, Y_train)
    
    def predict_proba(self, X_train, pred_threshold):
        # we take second column as it corresponds the probability
        # of sample belonging to failure class
        rf_pred_proba, _ = self.rf.predict_proba(X_train, pred_threshold)
        lgbm_pred_proba, _ = self.lgbm.predict_proba(X_train, pred_threshold)
        xgb_pred_proba, _ = self.xgb.predict_proba(X_train, pred_threshold)
        
        pred_proba = np.column_stack([rf_pred_proba, lgbm_pred_proba,
                                xgb_pred_proba])
        pred_proba = np.average(pred_proba, axis=1)
        pred_binary = np.where(pred_proba >= pred_threshold, 1, 0)

        # save pobability predictions to calculate auc score and create roc curve
        self.y_pred_proba = pred_proba

        return pred_proba, pred_binary

    def time_series_cross_validate(self, X, Y):
        tkf = TemporalKFold(X, Y, self.val_size)
        
        counter = 0
        for i in range(5):
            X_train, X_val, Y_train, Y_val = tkf.get_split(i)

            print(f"counter y train after tkfold {Counter(Y_train)}")
            print(f"counter y test after tkfold {Counter(Y_val)}")

            X_train, Y_train = SMOTE_undersampling(X_train, Y_train, minority_ratio = self.undersample)
            print(f"after undersampling x.shape: {X_train.shape}")
            print(Counter(Y_train))

            # feature scaling and encoding
            min_max_scalar = MinMaxScaler()
            min_max_scalar.fit(X_train)
            X_train = min_max_scalar.transform(X_train)
            print(f"after encoding X_train.shape: {X_train.shape}")

            self.fit(X_train, Y_train)

            min_to_major_ratio = Counter(Y_train)[1] / Counter(Y_train)[0]
            recons_prctl_thresh = (100 - min_to_major_ratio)

            # report performance of trained model on train data
            _ = report_performance_of_trained_model(self, self.model_name, X_train, Y_train, 
                recons_prctl_thresh, self.pred_threshold, mode='train')

            ####### Validation Steps #######
            X_val = min_max_scalar.transform(X_val)

            # report performance of trained model on validation data
            _ = report_performance_of_trained_model(self, self.model_name, X_val, Y_val, 
                recons_prctl_thresh, self.pred_threshold, mode='validation')

            # sample non failure validation cases to match the number of failure cases
            #df = pd.DataFrame(data={'y_val':Y_val, 'y_pred_proba':self.y_pred_proba})
            #num_failures = len(df[df['y_val']==1].index)
            #df_non_failure = df[df['y_val']==0].sample(n=num_failures, random_state=42)
            #df = pd.concat([df[df['y_val']==1],df_non_failure])

            #print(f"failure cases {len(df[df['y_val']==1].index)}")
            #print(f"non failure cases {len(df[df['y_val']==0].index)}")

            # get roc auc
            #self.get_roc_auc(df['y_val'], df['y_pred_proba'], id=counter)

            counter += 1

    def cross_validate(self, X, Y):
        X = X.to_numpy()

        # define stratifiedkfold where majority minority class ratios are maintained
        # in each fold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        counter = 0
        for train_index, test_index in skf.split(X, Y):
            X_train, X_val = X[train_index], X[test_index]
            Y_train, Y_val = Y[train_index], Y[test_index]

            print(f"counter y train after kfold {Counter(Y_train)}")
            print(f"counter y test after kfold {Counter(Y_val)}")

            X_train, Y_train = SMOTE_undersampling(X_train, Y_train, minority_ratio = self.undersample)
            print(f"after undersampling x.shape: {X_train.shape}")
            print(Counter(Y_train))

            # feature scaling and encoding
            min_max_scalar = MinMaxScaler()
            min_max_scalar.fit(X_train)
            X_train = min_max_scalar.transform(X_train)
            print(f"after encoding X_train.shape: {X_train.shape}")

            self.fit(X_train, Y_train)

            min_to_major_ratio = Counter(Y_train)[1] / Counter(Y_train)[0]
            recons_prctl_thresh = (100 - min_to_major_ratio)

            # report performance of trained model on train data
            _ = report_performance_of_trained_model(self, self.model_name, X_train, Y_train, 
                recons_prctl_thresh, self.pred_threshold, mode='train')

            ####### Validation Steps #######
            X_val = min_max_scalar.transform(X_val)

            # report performance of trained model on validation data
            _ = report_performance_of_trained_model(self, self.model_name, X_val, Y_val, 
                recons_prctl_thresh, self.pred_threshold, mode='validation')

            # sample non failure validation cases to match the number of failure cases
            df = pd.DataFrame(data={'y_val':Y_val, 'y_pred_proba':self.y_pred_proba})
            num_failures = len(df[df['y_val']==1].index)
            df_non_failure = df[df['y_val']==0].sample(n=num_failures, random_state=42)
            df = pd.concat([df[df['y_val']==1],df_non_failure])

            print(f"failure cases {len(df[df['y_val']==1].index)}")
            print(f"non failure cases {len(df[df['y_val']==0].index)}")

            # get roc auc
            self.get_roc_auc(df['y_val'], df['y_pred_proba'], id=counter)

            counter += 1
    

    def get_roc_auc(self, Y_val, y_pred_proba, id):
        # Calculate the False Positive Rate (FPR) and True Positive Rate (TPR)
        fpr, tpr, _ = roc_curve(Y_val, y_pred_proba, pos_label=1)

        # save false positive rate and true positive rate values
        df = pd.DataFrame(data={'fpr':fpr, 'tpr':tpr})
        df.to_csv(f'../report/tensorboard/{self.time}/fpr_tpr_{id}.csv')

        # Plot the ROC curve
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.savefig(f'../report/tensorboard/{self.time}/roc_curve_{id}.png')
        plt.close()

        # Calculate the AUC (Area Under the Curve)
        auc = roc_auc_score(Y_val, y_pred_proba)
        print('AUC: ', auc)

    def get_params(self):
        print("RF Model Parameters")
        print(self.rf.get_params())
        
        print("LGBM Model Parameters")
        print(self.lgbm.get_params())

        print("XGB Model Parameters")
        print(self.xgb.get_params())
    

class XGB:
    def __init__(self):
        self.model = None
        self.max_depth = 6
        self.time = None

    def initialize(self):
        self.model = XGBClassifier(n_estimators=100, max_depth=self.max_depth,
            n_jobs=-1, verbose=1)

    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict_proba(self, X_train, pred_threshold):
        # we take second column as it corresponds the probability
        # of sample belonging to failure class
        pred_proba = self.model.predict_proba(X_train)[:,1]
        pred_binary = np.where(pred_proba >= pred_threshold, 1, 0)
        return pred_proba, pred_binary

    def get_params(self):
        print(self.model.get_params())
        print(self.model.get_xgb_params())

class LGBM:
    def __init__(self):
        self.model = None
        self.max_depth = -1
        self.time = None
    
    def initialize(self):
        self.model = LGBMClassifier(n_estimators=100, max_depth=self.max_depth,
            n_jobs=-1, verbose = 1)
    
    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict_proba(self, X_train, pred_threshold):
        # we take second column as it corresponds the probability
        # of sample belonging to failure class
        pred_proba = self.model.predict_proba(X_train)[:,1]
        pred_binary = np.where(pred_proba >= pred_threshold, 1, 0)

        return pred_proba, pred_binary

    def get_params(self):
        print(self.model.get_params())


class RandomForest:
    def __init__(self):
        self.model = None
        self.max_depth = None
        self.time = None

    def initialize(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=self.max_depth,
            n_jobs=-1, verbose =1)    
    
    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict_proba(self, X_train, pred_threshold):
        # we take second column as it corresponds the probability
        # of sample belonging to failure class
        pred_proba = self.model.predict_proba(X_train)[:,1]
        pred_binary = np.where(pred_proba >= pred_threshold, 1, 0)
        
        return pred_proba, pred_binary

    def get_params(self):
        print(self.model.get_params())

class ExtraTreeModel:
    def __init__(self):
        self.model = None
        self.time = None
    
    def initialize(self):
        # build model
        self.model = ExtraTreesClassifier(n_estimators=100, bootstrap=True, n_jobs=-1, verbose = 1)
    
    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict_proba(self, X_train, pred_threshold):
        # we take second column as it corresponds the probability
        # of sample belonging to failure class
        pred_proba = self.model.predict_proba(X_train)[:,1]
        pred_binary = np.where(pred_proba >= pred_threshold, 1, 0)
        
        return pred_proba, pred_binary

    def get_params(self):
        print(self.model.get_params())



if __name__ == '__main__':
    pass



