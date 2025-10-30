import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import pandas as pd
import numpy as np
from models import ModelManager
from losses import WeightedCategoricalCrossentropy
from data_preprocess_svm import DataPreprocess
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
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15000)])
  except RuntimeError as e:
    print(e)


def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--approach', type=str, default="prev",
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
        default=f"/users/grad/papry/MS_thesis-main/data/20240313_061512/train_prev_rural_time0_None.csv",
        help='Path to train csv file')
    parser.add_argument(
        '--val_csv',
        type=str,
        default=f"/users/grad/papry/MS_thesis-main/data/20240313_061512/validation_prev_rural_time0_None.csv",
        help='Path to validation csv file'
        )
    parser.add_argument(
        '--test_csv',
        type=str,
        default="/users/grad/papry/MS_thesis-main/data/20240313_061512/test_prev_rural_time0_None.csv",
        help='Path to test csv file'
        )
    parser.add_argument('--model', type=str, default="TransformerTimeseries",
        help='model to use')
    parser.add_argument('--positional_encoding', type=bool, default=True,
        help='Use positional encoding in transformer model')
    parser.add_argument('--batch_size', type=int, default=None,
        help='batch size')
    parser.add_argument('--epochs', type=int, default=1000,
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
        default=None, #"../logs/20230809_014657/ckpt",
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
    data_preprocess.read_train_val_csv(args.train_csv,args.val_csv, args.test_csv)
   # data_preprocess.read_test_csv(args.test_csv)
    if args.undersample:
        data_preprocess.undersample(args.undersample)
    if args.drop_cat_features:
        data_preprocess.drop_cat_features()
    if args.autoencoder:
        data_preprocess.remove_failures()
    data_preprocess.seperate_kpis_labels()
   # if args.drop_cat_features:
       # data_preprocess.drop_test_cat_features()
   # data_preprocess.seperate_test_kpis_labels()
    data_preprocess.normalize(current_time,log_path=args.resume_from_ckpt)
    data_preprocess.one_hot_encode(current_time,log_path=args.resume_from_ckpt)
    train_kpis, train_labels = data_preprocess.get_train_kpis_labels()
    val_kpis, val_labels = data_preprocess.get_val_kpis_labels()
    test_kpis, test_labels = data_preprocess.get_test_kpis_labels()
    #print(train_kpis.shape, val_kpis.shape, test_kpis.shape)  
    #data_preprocess.min_max_scale_test_data(args.log_path)
   # data_preprocess.normalize_test_data(log_path=args.resume_from_ckpt)
    #data_preprocess.one_hot_encode_test_data(alog_path=args.resume_from_ckpt,ignore_cat=False)
    #if args.fix_miss_cat:
        #data_preprocess.pad_missing_catgorical_columns_test(args.fix_miss_cat)
    
    #train_ds=pd.concat([train_kpis, val_kpis])
    #train_labels=pd.concat([train_labels, val_labels])
    #ratio_1=WeightedCategoricalCrossentropy.calculate_ratio(train_labels),
    d_0=val_labels.value_counts()[0]
    d_1=val_labels.value_counts()[1]
    d_0=train_labels.value_counts()[0]
    d_1=train_labels.value_counts()[1]
   # print(d_0, d_1)
    d_0=test_labels.value_counts()[0]
    d_1=test_labels.value_counts()[1]
    print(train_kpis.shape)
   # train_ds=tf.concat([train_kpis, val_kpis], axis=0)
   # train_labels=tf.concat([train_labels, val_labels], axis=0)
    #d_0=train_labels.value_counts()[0]
    #d_1=train_labels.value_counts()[1]
    #print(m.get_scorer_names());
    #print(train_labels.shape, train_ds.shape)
    #clf = svm.SVC(class_weight="balanced" )
   # clf.fit(train_ds, train_labels)
    #clf.fit(train_ds, train_labels)
   # print(train_labels.shape)
   # train_labels=tf.reshape(train_labels,(-1,1))
  #  test_labels=tf.reshape(test_labels, (-1,1))
    
   # X= np.append(train_ds, train_labels, axis=1)
  #  y=np.append(test_kpis, test_labels, axis=1)
   # print("shape for x", X.shape)
   # X, y = datasets.load_iris(return_X_y=True)
    print(test_kpis.shape, test_labels.shape)
    #temporal_features,static_features = data_preprocess.seperate_cat_features(train_kpis)
    #result= mk.original_test(temporal_features)
   # print(result.p)
    c = 1
    
    while (c < test_kpis.shape[0]-1):
       # y=c+1
        train_X = test_kpis.iloc[0:, c:c+1]
        a= np.array(train_X)
        series=a.ravel()
        #print(series.shape)
        #result = adfuller(series, autolag='AIC')
        #print(f'ADF Statistic: {result[0]}')
        #print(f'n_lags: {result[1]}')
        #print(test_kpis.columns[c])
        #SSSprint(test_kpis.columns[c], ":    ",result[1])
        result2= mk.original_test(series)
        #pvalue=result2.p
        #print(test_kpis.iloc[0].values(c))
        #print(result[1])
        print(test_kpis.columns[c], ":    ", result2.p, "   ", result2.h )
        c=c+1
     
    #clf = svm.SVC(kernel='poly', gamma=10.0, C=1, random_state=42,class_weight='balanced', degree=10)
    #scores = cross_val_score(clf, X, y, cv=5)
   # scores = cross_val_score(clf, train_kpis,train_labels, cv=1, scoring='f1')
    #print(scores)

'''
    #skf = StratifiedKFold(n_splits=10)
    b=10
    fold = 0
    count = 0
    while (count < 5): 
        step_size=1024*2
        #a[0::step_size]
        train_X = train_kpis.iloc[fold*step_size:(fold+8)*step_size]
        train_y=train_labels.iloc[fold*step_size:(fold+8)*step_size]
        print(train_X.shape, train_y.shape)
        test_X = train_kpis.iloc[(fold+8)*step_size:(fold+10)*step_size]
        test_y = train_labels.iloc[(fold+8)*step_size:(fold+10)*step_size]
        print(test_X.shape, test_y.shape)
        #train_model(train,test,fold_no)
        fold += 10
        count+=1
        clf.fit(train_X,train_y)
        predictions = clf.predict(test_X)
        matrix = confusion_matrix(test_y,predictions)
        print(matrix)
        precision=precision_score(test_y,predictions)
        print('Fold',str(fold),'f1:',f1_score(test_y,predictions), 'pre:', precision)

   # predict=clf.predict(test_kpis)
   # result=f1_score(predict, test_labels)
    #precision=precision_score(predict, test_labels)
    #ccuracy=accuracy_score(predict, test_labels)
   # matrix = confusion_matrix(predict, test_labels)
    #matrix.diagonal()/matrix.sum(axis=1)
   # print(matrix)
   
    #print(result)
    


    # Prepare data with tf.data.Dataset
    to_tfdata = ToTfData(
        args.batch_size,
        args.approach,
        args.autoencoder,
        args.positional_encoding)
    #test_ds = to_tfdata.test_to_tfdata(test_kpis,test_labels)


    # Prepare data with tf.data.Dataset
    
 train_ds = to_tfdata.train_to_tfdata(
        train_kpis,
        train_labels,
        )
    val_ds = to_tfdata.val_to_tfdata(
        val_kpis,
        val_labels,
        )
    
    num_steps = len(test_kpis) // args.batch_size
    counter = 1
    last_batch_size = len(test_kpis) % args.batch_size


    # perform inference on model
    for kpis, labels in test_ds:
        if counter > num_steps:
            kpis = DataPreprocess.pad_to_batch_size(
                kpis,
                args.batch_size,
                args.approach
                )
        
        # get k nearest weather stations
        kpis = DataPreprocess.get_k_nearest_WS(
            kpis,
            args.num_weather_stations,
            args.approach
            )
   

  
   
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
    '''
if __name__ == '__main__':
    main()