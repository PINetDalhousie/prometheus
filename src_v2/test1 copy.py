import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from data_preprocess_v1 import DataPreprocess
from metrics_manager import MetricsManager,TestMetricsManager
import argparse
from to_tfdata import ToTfData
import datetime
from utils import plot_distribution
import tensorflow_probability as tfp
import numpy as np
import sys
from metrics_manager import OptimalThresholdCalculator

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15000)])
  except RuntimeError as e:
    print(e)

def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--autoencoder', type=bool, default=False,
                        help='Use autoencoder approach')
    parser.add_argument('--drop_cat_features', type=bool, default=False,
                        help='Drop categorical features')
    parser.add_argument('--calculate_threshold', type=bool, default=False,
                        help='whether to calculate threshold for autoencoder')
    parser.add_argument('--positional_encoding', type=bool, default=False,
        help='Use positional encoding in transformer model')
    parser.add_argument('--fix_miss_cat', type=str, default=None,
                        help='Zero pad missing categorical feature values when compared with the maximum possible categorical class values. Set value to deployment type: rural or urban')
    parser.add_argument(
        '--threshold',
        type=float,
        default=42.162086486816406,
        help='Threshold for autoencoder'
        )
    parser.add_argument(
        '--test_csv',
        type=str,
        default="/home/papry/Exp_Project/MS_thesis-main/data/20250719_180144/test_prev_urban_time3_None.csv",
        help='Path to test csv file'
        )
    parser.add_argument(
        '--model_path',
        type=str,
        default="/home/papry/Exp_Project/MS_thesis-main/logs/20250729_101737/ckpt",
        help='Path to trained model'
        )
    parser.add_argument(
        '--log_path',
        type=str,
        default="/home/papry/Exp_Project/MS_thesis-main/logs/20250729_101737/",
        help='Path to log directory')
    parser.add_argument(
        '--approach',
        type=str,
        default="prev",
        help=f"String identifying which approach to use. 'prev' indicates"
            f"previous approach with derived features.")
    parser.add_argument('--num_weather_stations', type=int, default=6,
                        help='Closest number of weather stations to consider')
    args = parser.parse_args()
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(args.log_path+f'{current_time}_test_args.txt', 'w') as file:
        for arg, value in vars(args).items():
            file.write(f"{arg}: {value}\n")
    
    # Preprocess data
    data_preprocess = DataPreprocess(args.approach)
    data_preprocess.read_test_csv(args.test_csv)
    if args.drop_cat_features:
        data_preprocess.drop_test_cat_features()
    data_preprocess.seperate_test_kpis_labels()
    data_preprocess.min_max_scale_test_data(args.log_path)
    data_preprocess.one_hot_encode_test_data(args.log_path,ignore_cat=False)
    if args.fix_miss_cat:
        data_preprocess.pad_missing_catgorical_columns_test(args.fix_miss_cat)
    test_kpis, test_labels = data_preprocess.get_test_kpis_labels()


    # Prepare data with tf.data.Dataset
    to_tfdata = ToTfData(
        args.batch_size,
        args.approach,
        args.autoencoder,
        args.positional_encoding)
    test_ds = to_tfdata.test_to_tfdata(test_kpis,test_labels)

    # load the model
    model = tf.keras.models.load_model(args.model_path,compile=False)

    # initialize test metrics
    if args.autoencoder:
        test_metrics = TestMetricsManager(
            threshold=args.threshold,
            approach=args.approach
            )
    else:
        test_metrics = TestMetricsManager(approach=args.approach)

    # Calculate the number of steps required to cover all samples in the test set
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

        # run inference
        predictions = model(kpis, training=False)
        
        # get first valid elements using prev approach
        if counter > num_steps:
            predictions = predictions[:last_batch_size]
            labels = labels[:last_batch_size]
            kpis = kpis[:last_batch_size]

        if args.autoencoder:
            test_metrics.update_autoencoder_test_metrics(
                labels,
                predictions,
                kpis
                )
        else:
            test_metrics.update_test_metrics(labels,predictions)
        counter += 1

    # Plot the distribution of reconstruction errors
    if args.autoencoder:
        if args.calculate_threshold:
            optimal_threshold_calculator = OptimalThresholdCalculator(test_metrics)
            optimal_threshold = optimal_threshold_calculator.calculate_threshold()
            print(optimal_threshold)
        else:
            optimal_threshold = 0.0 
    else:
        optimal_threshold = 0.0
    
    # print metric results
    with open(args.log_path+f'{current_time}_results.txt', 'w') as file:
        file.write(
            f'Test Precision 0: {test_metrics.precision_0_test.result() * 100}, \n'
            f'Test Recall 0: {test_metrics.recall_0_test.result() * 100}, \n'
            f'Test F1score 0: {MetricsManager.calculate_f1score(test_metrics.precision_0_test.result(),test_metrics.recall_0_test.result()) *100}, \n'
            f'Test Precision 1: {test_metrics.precision_1_test.result() * 100}, \n'
            f'Test Recall 1: {test_metrics.recall_1_test.result() * 100}, \n'
            f'Test F1score 1: {MetricsManager.calculate_f1score(test_metrics.precision_1_test.result(),test_metrics.recall_1_test.result()) *100}, \n'
            f'Optimal Threshold for reconstruction errors: {optimal_threshold}'
        )


if __name__ == '__main__':
    main()