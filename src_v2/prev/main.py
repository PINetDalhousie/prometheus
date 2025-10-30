from data import *
from data import get_input_feature_reports
from merge import *
from report import report_performance_of_trained_model,continuous_hist,categorical_bar,box_plot
from utility import *
from cleaning import *
from process_tables import process_all_tables
from preprocessing import *
from models import *
from balancing import *
from split import *
import numpy as np
from collections import Counter
import pandas as pd
from data_preparation import *
import seaborn as sns
import argparse
import sys
from datetime import datetime
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import classification_report
import tensorflow as tf
from ws_association import get_closest_ws
tf.random.set_seed(42)
import os
import matplotlib.pyplot as plt
from models import get_model
from aggregate_ws import aggregate_met_real_features, seperate_rl_met_features
pd.options.display.max_rows = 1000
pd.options.mode.chained_assignment = None  # default='warn'


def tune_hyperparameters(max_evals):
    space = {'eta': hp.uniform('eta', 0.01, 0.2),
        'max_depth': hp.quniform('max_depth', 3, 18, 1),
        'gamma': hp.quniform('gamma', 1, 9, 1),
        'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
        'reg_lambda': hp.uniform('reg_lambda', 0, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': hp.quniform('n_estimators', 3, 250, 1)
        }

    #space = {'undersample': hp.uniform('undersample', 0.005, 0.5),
    #    'oversample': hp.uniform('oversample', 0.005, 0.5),
    #    'optimal_distance': hp.uniform('optimal_distance', 2, 50)}
    
    trials = Trials()
    best_hyperparams = fmin(fn=main, space=space, algo=tpe.suggest,
        max_evals=max_evals, trials=trials)

    print(best_hyperparams)

def main(space):
    # args
    parser = argparse.ArgumentParser(description="radio link failure")
    parser.add_argument('--dataset', type=str, default='new',
                        help='input dataset to use')
    parser.add_argument('--oversample', type=float, default=0.0,
                        help='oversampling minority class ratio')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='validation data ratio for train/validation split')
    parser.add_argument('--optimal-distance', type=float, default=None,
                        help='optimal distance for ws to merge tables')
    parser.add_argument('--prev-days', type=int, default=10,
                        help='number of prev day features to use')
    parser.add_argument('--forecast-days', type=int, default=1,
                        help='number of forecast days to consider')
    parser.add_argument('--undersample', type=float, default=0.05,
                        help='undersampling minority class ratio')
    parser.add_argument('--model', type=str, default='Ensemble',
                        help='model to use')
    parser.add_argument('--sample', type=float, default=0.0,
                        help='sample from the full dataset for testing full pipeline')
    parser.add_argument('--time_sample', type=float, default=0.0,
                        help='sample across time from the full dataset for testing full pipeline')
    parser.add_argument('--exp_name', type=str, default='VanillaDenseNN',
                        help='experiment name for log files')
    parser.add_argument('--pca-components', type=int, default=None,
                        help='number of components to keep in performing PCA')
    parser.add_argument('--split', type=str, default='temporal_conservative_split',
                        help=f'splitting method for the train validation split.options:time_series_split;random,temporal_conservative_split')
    parser.add_argument('--logging', type=bool, default=True,
                        help='flag to indicate if logs will be saved or not')
    parser.add_argument('--generate_data_dist', type=bool, default=False,
                        help='flag to generate raw and processed data distributions of features')
    parser.add_argument('--keep_features', type=bool, default=True,
                        help='keep most important features for testing purpose')
    parser.add_argument('--remove_outliers', type=bool, default=False,
                        help='before model, outliers are calculated and removed based on non failure links')
    parser.add_argument('--pred_threshold', type=float, default=0.5,
                        help='threshold to use for predicted probability scores')
    parser.add_argument('--perform_split', type=bool, default=True,
                        help='perform train validation split based on val_size. This allows us to do Cross Validation')
    parser.add_argument('--cross_validate_type', type=str, default="time_series",
                        help='type of cross validation to use. options: kfold;time_series')
    parser.add_argument('--k_nearest_ws', type=int, default=1,
                        help='k number of weather stations to consider for each radio site')



    args = parser.parse_args()
    
    # save logs
    current_time = str(datetime.now()).replace(" ","_")
    if args.logging:
        sys.stdout = open(f"../report/logs/{args.exp_name}/{current_time}.txt", 'wt')
    print(sys.argv)
    print(args)
    if args.logging:
        os.system(f'mkdir -p ../report/tensorboard/{current_time}')
    
    if args.dataset == 'new':
        # file path for zipped RegionA train data
        data_zip_path = os.path.join("../data/train/RegionA.zip")
        if args.optimal_distance == 0.0:
            optimal_distance = 15.82
        else:
            optimal_distance = args.optimal_distance
    elif args.dataset == 'prev':
        data_zip_path = os.path.join("../data/train/prev_data.zip")
        if args.optimal_distance == 0.0:
            optimal_distance = 22.71
        else:
            optimal_distance = args.optimal_distance

    print(f"optimal distance for radio site to weather station association is {optimal_distance}")

    print('reading and processing different tables')
    rl_kpis, rl_sites, distances, met_forecast, met_stations, met_real = process_all_tables(data_zip_path,
        data_split='train', generate_histograms=args.generate_data_dist, num_prev_days=args.prev_days,
        forecast_days=args.forecast_days, handle_scalibility=True, dataset=args.dataset)
    
    # print out some relevant stats from the tables that has been read
    print(f"shape of imputed radio link kpi table with rlf columns: {rl_kpis.shape}")
    print(f"stats on radio link failure events: {rl_kpis['1-day-predict'].value_counts()}")
    print(f"number of radio link sites from rl sites table: {rl_sites['site_id'].unique().shape}")
    print(f"number of radio link sites from kpi table: {rl_kpis['site_id'].unique().shape}")
    print(f"number of radio link sites from distance table: {distances.shape[0]}")
    print(f"number of ws sites from distance table: {distances.shape[1]}")
    print(f"number of ws sites from forecast table: {met_forecast['station_no'].unique().shape}")
    print(f"number of sites from ws real table: {met_real['station_no'].unique().shape}")
    print(f"number of ws sites from ws sites table: {met_stations['station_no'].unique().shape}")
    print(f"shape of imputed forecast table: {met_forecast.shape}")
    
    
    # sample to test out. it's better to sample across time
    if args.time_sample != 0.0:
        min_date = rl_kpis.datetime.min()
        max_date = rl_kpis.datetime.max()
        print(f"minimum date observed : {min_date}")
        print(f"maximum date observed : {max_date}")
        time_between = max_date - min_date
        time_cutoff = min_date + args.time_sample*time_between
        rl_kpis = rl_kpis[rl_kpis.datetime <= time_cutoff]
        met_forecast = met_forecast[met_forecast.datetime <= time_cutoff]
        met_real = met_real[met_real.datetime <= time_cutoff]
        print(f"after sampling rl kpis shape {rl_kpis.shape}")
        print(f"after sampling met forecast shape {met_forecast.shape}")
        print(f"after sampling met real shape {met_real.shape}")


    # get closest ws for each radio site
    distances = get_closest_ws(distances, optimal_distance, met_forecast, met_real, method="k_nearest",
         num_neighbors=args.k_nearest_ws, use_ws_list="real")
    print(f"shape of table used for distance/weather station association {distances.shape}")
    

    # aggregate associated weather station features over the number of weather station 
    # associated with each radio site. So derived features for each WS feature is 
    # calculated (min,max,mean,std)
    # met_real = aggregate_met_real_features(distances, met_real)
    # print(f"calculated derived features for neighboring weather stations; shape {met_real.shape}")


    # merge tables based on distance, kpi and forecast table
    rl_kpis = merge_tables(rl_kpis, rl_sites, distances, met_forecast, met_stations, met_real, 
                            include_forecast=False, include_real=True)
    del rl_sites
    del distances
    del met_forecast
    del met_stations
    del met_real
    print(f"after merging tables rl_kpis.shape {rl_kpis.shape}")


    
    # # aggregate associated weather station features over the number of weather station 
    # # associated with each radio site. So derived features for each WS feature is 
    # # calculated (min,max,mean,std)
    if args.k_nearest_ws > 1:
        # seperate rl features and met real features
        rl_kpis, df_met = seperate_rl_met_features(rl_kpis, args.prev_days)
        print(f"radio link features and weather station features have been seperated")

        rl_kpis = aggregate_met_real_features(rl_kpis, df_met)
        del df_met
        print(f"calculated derived features for neighboring weather stations; shape {rl_kpis.shape}")



    # clean final merged table
    #features_to_drop = ["direction","modulation", "clutter_class", "freq_band"]
    features_to_drop = []
    rl_kpis = clean_final_merged_table(rl_kpis, features_to_drop, prev_days=args.prev_days, model=args.model,
        keep_features=args.keep_features, keep_feature_method="kpi_real_continuous")
    print(f"cleaned final dataframe shape shape: {rl_kpis.shape}") 
    

    if args.sample != 0.0:
        rl_kpis = rl_kpis.sample(frac=args.sample, random_state=42)
        print(f"after sampling small dataset {rl_kpis.shape}")



    # get feature reports for train validation data
    if args.generate_data_dist:
        get_input_feature_reports(rl_kpis, data_type = 'train_val')
        ### generate feature distributions ###
        continuous_hist(rl_kpis, table_name='train_val')
        categorical_bar(rl_kpis, table_name='train_val')
        ### generate box plots and view outliers 
        box_plot(rl_kpis[rl_kpis['1-day-predict']==0], table_name='non_failure_train_val_before_removal')
        calculate_num_outliers(rl_kpis[rl_kpis['1-day-predict']==0], table_name='train_val')
        non_failure_clean = remove_outliers(rl_kpis[rl_kpis['1-day-predict']==0], mode='non_failure', num_times=1)
        box_plot(non_failure_clean[non_failure_clean['1-day-predict']==0], table_name='non_failure_train_val_after_removal')
    
    
  
    ### calculate and remove outliers 
    # before model training continuous features should have normal distributions 
    # remove outliers of continuous features of non failure links 
    # as this is anomaly prediction, we do not remove outliers based on the whole dataset;
    # instead we remove outliers of continuous features based on only non failure links    
    if args.remove_outliers:
        rl_kpis = remove_outliers(rl_kpis, mode='non_failure', num_times=1)
        print(f"outliers have been removed. After removal shape {rl_kpis.shape}")
    
    

    # helper code to prepare data for preprocessing
    X = rl_kpis.drop(['1-day-predict'],axis=1)
    Y = rl_kpis['1-day-predict']
    Y = Y.astype('int').to_numpy()
    # seperate categorical and numerical features
    categorical_features = X.select_dtypes('object').columns.to_list()
    numerical_features = X.select_dtypes('number').columns.to_list()
    all_features = numerical_features + categorical_features
    print(f"all features as list {all_features}")
    

    if args.perform_split:
        # split into train and validation data
        X_train, X_val, Y_train, Y_val = split_dataset(args.split, X, Y, args.val_size)
        print(f"after train validation split using: X_train.shape: {X_train.shape} | X_val.shape: {X_val.shape}")
    else:
        X_train = X
        Y_train = Y
        X_val = None
        Y_val = None


    
    if args.perform_split:
        # undersample majority class
        if args.undersample != 0.0:
            X_train, Y_train = SMOTE_undersampling(X_train, Y_train, minority_ratio = args.undersample)
            print(f"after undersampling x.shape: {X_train.shape}")
            print(Counter(Y_train))
    
    
    if args.perform_split:
        # feature scaling and encoding
        X_train, one_hot_encoder, min_max_scalar = preprocessing(X_train, Y_train,
            numerical_columns=numerical_features,
            categorical_columns=categorical_features,
            model_name=args.model)
        print(f"after encoding X_train.shape: {X_train.shape}")
    
    if args.perform_split:
        # generate feature correlation matrix
        columns_for_corr = [*numerical_features, *list(one_hot_encoder.get_feature_names_out(categorical_features))]
        columns_for_corr.append('rlf')
        corr_df = pd.DataFrame(np.append(X_train,np.expand_dims(Y_train,axis=1),axis=1), columns=columns_for_corr).corr()
        print('Feature correlation coefficients after encoding')
        pd.set_option('display.max_rows', None)
        print(corr_df['rlf'].abs().sort_values())
        #sns.heatmap(corr_df).get_figure().savefig('../report/encoded_corr.png', dpi=1200)
    
    # use dimensionality reduction technique
    #if args.pca_components != None:
    #X_train, pca = apply_pca(X_train, num_components=args.pca_components)
    #print(f"after applying PCA X_train.shape: {X_train.shape}")
    

    
    # oversample minority class
    min_to_major_ratio = Counter(Y_train)[1] / Counter(Y_train)[0]
    recons_prctl_thresh = (100 - min_to_major_ratio)
    # set ratio as threshold value for percentile based autoencoder approach
    if args.perform_split:
        if args.oversample > min_to_major_ratio:
            if args.oversample != 0.0:
                X_train, Y_train = SMOTE_oversampling(X_train, Y_train, minority_ratio = args.oversample)
                print(f"after oversampling X_train.shape: {X_train.shape}")
                print(Counter(Y_train))


    # choose model
    print(f"using model {args.model}")
    clf_1_day_pred = get_model(args.model)

    # configure model
    clf_1_day_pred.model_name = args.model
    clf_1_day_pred.time = current_time
    clf_1_day_pred.X_val = X_val
    clf_1_day_pred.Y_val = Y_val
    clf_1_day_pred.X_train = X_train
    clf_1_day_pred.Y_train = Y_train
    clf_1_day_pred.undersample = args.undersample
    clf_1_day_pred.numerical_features = numerical_features
    clf_1_day_pred.categorical_features = categorical_features
    clf_1_day_pred.pred_threshold = args.pred_threshold
    clf_1_day_pred.val_size = args.val_size

    
    if args.model == "LSTM_AE":
        # set model specific parameters
        clf_1_day_pred.num_features = X_train.shape[1]
        clf_1_day_pred.batch_size = 32
        clf_1_day_pred.epochs = 1
        clf_1_day_pred.learning_rate = 1e-3
        clf_1_day_pred.pred_threshold = recons_prctl_thresh
        clf_1_day_pred.threshold_method = 'percentile'
        clf_1_day_pred.threshold_reduce = 'mean'
    elif args.model == "VanillaDenseNN":
        clf_1_day_pred.batch_size = 32
        clf_1_day_pred.epochs = 10
        clf_1_day_pred.learning_rate = 1e-3
        

    # initialize model
    #clf_1_day_pred.initialize()
    clf_1_day_pred.initialize()

    print("Before training, Model Parameters")
    clf_1_day_pred.get_params()
    
    #box_plot(pd.DataFrame(X_train), table_name='train_before_model')
    #box_plot(pd.DataFrame(X_val), table_name='val_before_model')
    if args.perform_split:
        clf_1_day_pred.fit(X_train, Y_train)
    else:
        if args.cross_validate_type == "kfold":
            clf_1_day_pred.cross_validate(X_train, Y_train)
        elif args.cross_validate_type == "time_series":
            clf_1_day_pred.time_series_cross_validate(X_train, Y_train)
        sys.exit()
    
    print("After training Model Parameters")
    clf_1_day_pred.get_params()
    
    if args.perform_split:
        # report performance of trained model on train data
        _ = report_performance_of_trained_model(clf_1_day_pred, args.model, X_train, Y_train, 
            recons_prctl_thresh, args.pred_threshold, mode='train')

        
        ####### Validation Steps #######
        X_val, _, _ = preprocessing(X_val, Y_val, numerical_columns=numerical_features, 
                                    categorical_columns=categorical_features, 
                                    one_hot_encoder=one_hot_encoder,
                                    min_max_scalar=min_max_scalar,
                                    model_name=args.model)
        #if args.pca_components != None:
        #X_val = pca.transform(X_val)
        #print(Counter(Y_val))
        # report performance of trained model on validation data
        _ = report_performance_of_trained_model(clf_1_day_pred, args.model, X_val, Y_val, 
            recons_prctl_thresh, args.pred_threshold, mode='validation')

    
    ####### test data #######
    if args.dataset == 'new':
        # prepare test data
        data_zip_path = [os.path.join("../data/test/RegionA_test_20210426.zip"),
                        os.path.join("../data/test/RegionA_test_20210525.zip"),
                        os.path.join("../data/test/RegionA_test_20210614.zip"),
                        os.path.join("../data/test/RegionA_test_20210817.zip")]
    elif args.dataset == 'prev':
        data_zip_path = [os.path.join("../data/test/test1.zip"),
                        os.path.join("../data/test/test2.zip"),
                        os.path.join("../data/test/test3.zip"),
                        os.path.join("../data/test/test4.zip"),
                        os.path.join("../data/test/test5.zip"),
                        os.path.join("../data/test/test6.zip")]


    for i in range(len(data_zip_path)):
        X_test, Y_test, identifiers = get_test_results(data_zip_path[i], 
            optimal_dist=optimal_distance,
            prev_days = args.prev_days,
            forecast_days = args.forecast_days,
            model=args.model,
            features_to_drop=features_to_drop,
            keep_features=args.keep_features,
            dataset=args.dataset)

        if i == 0 :
            test_input = X_test
            test_labels = Y_test
            test_identifiers = identifiers
        else :
            test_input = pd.concat([test_input, X_test], ignore_index=True)
            test_labels = np.append(test_labels, Y_test, axis=0)
            test_identifiers = pd.concat([test_identifiers, identifiers], ignore_index=True)
    
    print(f"test input shape {test_input.shape}")


    # align test feature list to be the same as train feature list
    test_input = align_test_features(test_input, all_features)
    
    print(f"after aligning features {test_input.shape}")
    # get test feature reports
    #get_input_feature_reports(test_input, data_type = 'test')

    test_input, _, _ = preprocessing(test_input, test_labels, numerical_columns=numerical_features,
        categorical_columns=categorical_features, 
        one_hot_encoder=one_hot_encoder,
        min_max_scalar=min_max_scalar,
        model_name=args.model)
    
    print(f"after encoding and scaling {test_input.shape}")
    #if args.pca_components != None:
    #test_input = pca.transform(test_input)
    #box_plot(pd.DataFrame(test_input), table_name='test_before_model')
    # report performance of trained model on test data
    test_report = report_performance_of_trained_model(clf_1_day_pred, args.model, test_input, test_labels, 
        recons_prctl_thresh, args.pred_threshold, mode='test', identifiers=test_identifiers)
    
    #report_evaluation_metrics(predictions, test_labels, "Test", hparam=0.1) 
    return {'loss': -test_report['macro avg']['f1-score'], 'status': STATUS_OK} 
    
if __name__ == '__main__':
    #tune_hyperparameters(125)
    #for i in [99.1,99.2,99.3,99.4,99.5,99.6,99.7,99.8,99.9]:
    #    main({'recons_prctl_thresh':i})
    #generate_min_rl2ws_dist('prev')
    main({})
  
    
    
    
    
    
    
    
