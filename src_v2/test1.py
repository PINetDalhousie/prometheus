from collections import Counter
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from data_preprocess_v1 import DataPreprocess
from metrics_manager import MetricsManager,TestMetricsManager
import argparse
from to_tfdata import ToTfData
import datetime
import time
from utils import plot_distribution
import tensorflow_probability as tfp
import numpy as np
import sys
from metrics_manager import OptimalThresholdCalculator
from models import TransformerTimeseries
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
    parser.add_argument('--positional_encoding', type=bool, default=True,
        help='Use positional encoding in transformer model')
    parser.add_argument('--fix_miss_cat', type=str, default=None,
                        help='Zero pad missing categorical feature values when compared with the maximum possible categorical class values. Set value to deployment type: rural or urban')
    parser.add_argument(
        '--threshold',
        type=float,
        default=3.0413200855255127,
        help='Threshold for autoencoder'
        )
    parser.add_argument(
        '--test_csv',
        type=str,
        default="/home/papry/Exp_Project/MS_thesis-main/data/20250513_230206_34/test_new_urban_time2_None.csv",
        help='Path to test csv file'
        )
    parser.add_argument(
        '--model_path',
        type=str,
        default="/home/papry/Exp_Project/MS_thesis-main/logs/20250527_233848/ckpt",
        help='Path to trained model'
        )
    parser.add_argument(
        '--log_path',
        type=str,
        default="/home/papry/Exp_Project/MS_thesis-main/logs/20250527_233848/",
        help='Path to log directory')
    parser.add_argument(
        '--approach',
        type=str,
        default="new",
        help=f"String identifying which approach to use. 'prev' indicates"
            f"previous approach with derived features.")
    parser.add_argument('--num_weather_stations', type=int, default=3,
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
   # model = tf.keras.models.load_model(args.model_path,compile=False)
    model = tf.keras.models.load_model(
    args.model_path,
    compile=False
   # custom_objects={'TransformerTimeseries': TransformerTimeseries}
)
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
    results_files = []
    num_time_steps = 6
    for t in range(num_time_steps):
        path = os.path.join(args.log_path, f'{current_time}_deletion_results_time{t}.csv')
        f = open(path, 'w')
        f.write('feature_removed,precision_0,recall_0,f1_0,precision_1,recall_1,f1_1,optimal_threshold\n')
        results_files.append(f)
    features_sorted = [[3,5,6,7,2,1,4],[4,2,1,7,6,5,3], [5,3,2,1,4,6,7],[6,7,1,5,4,3,2],[1,2,3,4,5,6,7],[4,2,1,7,6,3,5]]  # Example feature indices sorted by importance
    iteration=1
    for i in range(iteration):  # set to your model's time dimension
        for time_idx in range(num_time_steps):
            features_to_delete = []
            all_features = features_sorted[time_idx]
            print(f"\nEvaluating by removing features at time step {time_idx}: {features_to_delete}\n")
            for features in list(all_features):
                print("Deleting feature:", features)
                features_to_delete.append(features)
                print("Deleting feature:", features_to_delete)
                counter = 1
            # perform inference on model
                if args.autoencoder:
                    test_metrics = TestMetricsManager(
                        threshold=args.threshold,
                        approach=args.approach
                    )
                else:
                    test_metrics = TestMetricsManager(approach=args.approach)
                test_ds2=test_ds
                for kpis, labels in test_ds2:
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
                    pc=0
                    for i in range(len(labels)):
                        if labels[i][1]>0:
                        # print("pdata", labels[i], kpis[i])
                            pc=pc+1
                # print("Positive class in test set: ", pc)
                    # run inference
                    

                # indices to zero out
                    # Option A: numpy in-place (safe, then convert back to tensor)
                    kpis = kpis.numpy()
                    #kpis = kpis.astype(np.float32)  
                #  for t in range(kpis.shape[2]):
                    # v = kpis[..., 3][:, :, t].ravel()
                    # print("time", t, Counter(np.round(v, 2)).most_common(5))
                # print("Before Zeroed time-step 3 shape check:", kpis[:, :, :, 3].shape, "sum:", kpis[:, :, :, 3].sum())  # ensure numeric dtype
                    kpis[:,:,:, features_to_delete] = 0.0             # zero selected features across last axis
                    #print("Zeroed time-step 3 shape check:", kpis[:, :, :, 3].shape, "sum:", kpis[:, :, :, 3].sum())
                    
                    kpis = tf.convert_to_tensor(kpis)
                    predictions = model(kpis, training=False)
                # end = time.time()
                    #inference_time = 1000*(end - start)
                    #print(f"Inference time for batch {counter}: {inference_time:.4f} mseconds")
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
                '''with open(args.log_path+f'{current_time}_{features}_results_{time_idx}.txt', 'w') as file:
                    file.write(
                        f'Test Precision 0: {test_metrics.precision_0_test.result() * 100}, \n'
                        f'Test Recall 0: {test_metrics.recall_0_test.result() * 100}, \n'
                        f'Test F1score 0: {MetricsManager.calculate_f1score(test_metrics.precision_0_test.result(),test_metrics.recall_0_test.result()) *100}, \n'
                        f'Test Precision 1: {test_metrics.precision_1_test.result() * 100}, \n'
                        f'Test Recall 1: {test_metrics.recall_1_test.result() * 100}, \n'
                        f'Test F1score 1: {MetricsManager.calculate_f1score(test_metrics.precision_1_test.result(),test_metrics.recall_1_test.result()) *100}, \n'
                        f'Optimal Threshold for reconstruction errors: {optimal_threshold}'
                    )
                    '''
                f1_0 = MetricsManager.calculate_f1score(test_metrics.precision_0_test.result(), test_metrics.recall_0_test.result()) * 100
                f1_1 = MetricsManager.calculate_f1score(test_metrics.precision_1_test.result(), test_metrics.recall_1_test.result()) * 100
                # write to the file for this time_idx
                results_files[time_idx].write(
                    f'"{features}",'
                    f'{test_metrics.precision_0_test.result() * 100:.4f},'
                    f'{test_metrics.recall_0_test.result() * 100:.4f},'
                    f'{f1_0:.4f},'
                    f'{test_metrics.precision_1_test.result() * 100:.4f},'
                    f'{test_metrics.recall_1_test.result() * 100:.4f},'
                    f'{f1_1:.2f},'
                    f'{optimal_threshold}\n'
                )

    for f in results_files:
        f.close()

if __name__ == '__main__':
    main()