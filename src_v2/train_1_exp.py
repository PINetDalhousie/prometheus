import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import pandas as pd
import numpy as np
from models import ModelManager
from losses import WeightedCategoricalCrossentropy
from data_preprocess import DataPreprocess
import argparse
import datetime
from to_tfdata import ToTfData
from metrics import F1Score, ReconstructionError, TruncateReconstructionError
from callbacks import Callbacks
import os
import sys
from sklearn.model_selection import cross_val_score
from losses import TruncatedMSE
from sklearn import svm
from sklearn import datasets
from sklearn import metrics as m 
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import pymannkendall as mk
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
import shap
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15000)])
  except RuntimeError as e:
    print(e)


def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--approach', type=str, default="new",
        help=f"String identifying which approach to use. 'prev' indicates "
            f"previous approach with derived features.")
    parser.add_argument('--autoencoder', type=bool, default=False,
                        help='Use autoencoder approach')
    parser.add_argument('--drop_cat_features', type=bool, default=False,
                        help='Drop categorical features')
    parser.add_argument('--undersample', type=float, default=0.0,
                        help='Undersample majority class')
    parser.add_argument('--train_csv',
        type=str,
        default=f"/users/grad/papry/MS_thesis-main/data/20240310_151048/train_new_rural_time4_None.csv",
        help='Path to train csv file')
    parser.add_argument(
        '--val_csv',
        type=str,
        default=f"/users/grad/papry/MS_thesis-main/data/20240310_151048/validation_new_rural_time4_None.csv",
        help='Path to validation csv file'
        )
    parser.add_argument('--model', type=str, default="TransformerTimeseries",
        help='model to use')
    parser.add_argument('--positional_encoding', type=bool, default=True,
        help='Use positional encoding in transformer model')
    parser.add_argument('--batch_size', type=int, default=1024,
        help='batch size')
    parser.add_argument('--epochs', type=int, default=2,
        help='Number of epochs to train model')
    parser.add_argument('--initial_epoch', type=int, default=0,
        help='Epoch to start training from. Required for resuming training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
        help='Learning rate for the optimizer')
    parser.add_argument('--num_stations', type=int, default=3,
        help='Number of K closest weather stations to consider')
    parser.add_argument('--prev_days_data', type=int, default=4,
        help='Number of previous days data to use')
    parser.add_argument(
        '--resume_from_ckpt',
        type=str,
        default= None, #"/users/grad/papry/MS_thesis-main/logs/20240422_125231/ckpt",
        help='Resume training from a checkpoint'
        )
    args = parser.parse_args()
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.system(f"mkdir -p /users/grad/papry/MS_thesis-main/logs/{current_time}")
    with open(f'/users/grad/papry/MS_thesis-main/logs/{current_time}/train_args.txt', 'w') as file:
        for arg, value in vars(args).items():
            file.write(f"{arg}: {value}\n")


    # Preprocess data
    data_preprocess = DataPreprocess(args.approach)
    data_preprocess.read_train_val_csv(args.train_csv,args.val_csv)
    if args.undersample:
        data_preprocess.undersample(args.undersample)
    if args.drop_cat_features:
        data_preprocess.drop_cat_features()
    if args.autoencoder:
        data_preprocess.remove_failures()
    data_preprocess.seperate_kpis_labels()
    #data_preprocess.normalize(current_time,log_path=args.resume_from_ckpt)
    data_preprocess.min_max_scale(current_time,log_path=args.resume_from_ckpt)
    data_preprocess.one_hot_encode(current_time,log_path=args.resume_from_ckpt)
    train_kpis, train_labels = data_preprocess.get_train_kpis_labels()
    val_kpis, val_labels = data_preprocess.get_val_kpis_labels()


    # Prepare data with tf.data.Dataset
    to_tfdata = ToTfData(
        args.batch_size,
        args.approach,
        args.autoencoder,
        args.positional_encoding
        )
    train_ds = to_tfdata.train_to_tfdata(
        train_kpis,
        train_labels,
        )
    val_ds = to_tfdata.val_to_tfdata(
        val_kpis,
        val_labels,
        )
    

    # model params
    
    if args.resume_from_ckpt != None:
        # Retrieve model from a checkpoint
        model = tf.keras.models.load_model(args.resume_from_ckpt,compile=False)
    else :
        # Initialize model manager and create the model
        model_manager = ModelManager()
        if args.autoencoder:
            model = model_manager.get_model(
                name=args.model,
                num_stations=args.num_stations,
                prev_days_data=args.prev_days_data,
                num_features=data_preprocess.feature_number,
                batch_size=args.batch_size,
                cat_features_number=data_preprocess.cat_feature_number,
                feature_number=data_preprocess.feature_number,
            )
        
        else:
            model = model_manager.get_model(
                name=args.model,
                ratio=WeightedCategoricalCrossentropy.calculate_ratio(train_labels),
                num_stations=args.num_stations,
                prev_days_data=args.prev_days_data,
                num_features=data_preprocess.feature_number,
                batch_size=args.batch_size,
                cat_features_number=data_preprocess.cat_feature_number,
                latent_dim=32,
                )
    

    # Define loss function and optimizer
    if args.autoencoder:
        #loss_object = tf.keras.losses.MeanSquaredError()
        #loss_object = tf.keras.losses.MeanAbsoluteError()
        if args.model == "VarLSTMAutoencoder":
            loss_object = TruncatedMSE()
        elif args.model == "LSTMAutoencoder":
            loss_object = tf.keras.losses.MeanSquaredError()
        elif args.model == "LSTMAutoencoderPlus":
            loss_object = tf.keras.losses.MeanAbsoluteError()
    else:
            loss_object = WeightedCategoricalCrossentropy(
            ratio=WeightedCategoricalCrossentropy.calculate_ratio(train_labels)
            )
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    
    # define metrics
    if args.autoencoder:
        if args.model == "VarLSTMAutoencoder":
            metrics = [
                TruncateReconstructionError(name='reconstruction_error')
            ]
        elif args.model == "LSTMAutoencoder":
            metrics = [
                ReconstructionError(name='reconstruction_error')
            ]
        elif args.model == "LSTMAutoencoderPlus":
            metrics = [
                ReconstructionError(name='reconstruction_error')
            ]
    else:
        metrics = [
            tf.keras.metrics.Precision(name='precision_0',class_id=0),
            tf.keras.metrics.Recall(name='recall_0',class_id=0),
            F1Score(name='f1score_0',class_id=0),
            tf.keras.metrics.Precision(name='precision_1',class_id=1),
            tf.keras.metrics.Recall(name='recall_1',class_id=1),
            F1Score(name='f1score_1',class_id=1)
            ]


    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=loss_object,
        metrics=metrics
    )


    # Define callbackss
    callbacks = Callbacks(current_time)
    if args.autoencoder:
        callbacks.add_callback(
            name='model_checkpoint',
            callback=tf.keras.callbacks.ModelCheckpoint(    
                filepath=f'../logs/{current_time}/ckpt',
                save_weights_only=False,
                monitor='val_loss',
                verbose=1,
                mode='min',
                save_best_only=True
                )
            )
        #callbacks.remove_callback('model_checkpoint')
    
    
    # Run training and validation
    if args.model=="SVM0":
        model.fit(
        x=train_ds,
        epochs=args.epochs,
        verbose=2,
        callbacks=callbacks.get_callbacks(),
        initial_epoch=args.initial_epoch,
        validation_data=val_ds,
        workers=-1,
        use_multiprocessing=True,
        )

    else:   
        model.fit(
            x=train_ds,
            epochs=args.epochs,
            verbose=2,
            callbacks=callbacks.get_callbacks(),
            initial_epoch=args.initial_epoch,
            validation_data=val_ds,
            workers=-1,
            use_multiprocessing=True,
        )
       # attrib_data =train_ds[:200]
       
        explainer = shap.DeepExplainer(model,train_ds)
        num_explanations = 20
        shap_vals = explainer.shap_values(train_ds[ :20])
        shap.plots.waterfall(shap_vals, max_display=14)
    
if __name__ == '__main__':
    main()