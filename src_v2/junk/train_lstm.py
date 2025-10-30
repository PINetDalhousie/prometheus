

import tensorflow as tf
from models import ModelManager
from junk.train_val import TrainVal
from metrics_manager import MetricsManager
from early_stopping import EarlyStopping
from losses import WeightedCategoricalCrossentropy
from data_preprocess import DataPreprocess
import argparse
import datetime
from to_tfdata import ToTfData


def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--approach', type=str, default="prev",
                        help='String identifying which approach to use. "prev" indicates previous approach with derived features.')
    parser.add_argument('--train_csv', type=str, default="../data/20230421_190706/train_prev_rural_time0_with_cat.csv",
                        help='Path to train csv file')
    parser.add_argument('--val_csv', type=str, default="../data/20230421_190706/validation_prev_rural_time0_with_cat.csv",
                        help='Path to validation csv file')
    parser.add_argument('--model', type=str, default="LSTMPlus",
                        help='model to use')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=1500,
                        help='Number of epochs to train model')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--num_stations', type=int, default=3,
                        help='Number of K closest weather stations to consider')
    parser.add_argument('--prev_days_data', type=int, default=5,
                        help='Number of previous days data to use')
    parser.add_argument('--num_features', type=int, default=16,
                        help='Number of features per sample')
    args = parser.parse_args()
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Preprocess data
    data_preprocess = DataPreprocess(args.approach)
    data_preprocess.read_train_val_csv(args.train_csv,args.val_csv)
    data_preprocess.seperate_kpis_labels()
    data_preprocess.min_max_scale()
    per_step_cat_features = data_preprocess.one_hot_encode()
    train_kpis, train_labels = data_preprocess.get_train_kpis_labels()
    val_kpis, val_labels = data_preprocess.get_val_kpis_labels()
    
    # Prepare data with tf.data.Dataset
    to_tfdata = ToTfData(args.batch_size,args.approach)
    train_ds = to_tfdata.train_to_tfdata(train_kpis,train_labels)
    val_ds = to_tfdata.val_to_tfdata(val_kpis,val_labels)

    # Initialize model manager and get the model
    model_manager = ModelManager()
    model = model_manager.get_model(
        name=args.model,
        ratio=WeightedCategoricalCrossentropy.calculate_ratio(train_labels),
        num_stations=args.num_stations,
        prev_days_data=args.prev_days_data,
        num_features=args.num_features,
        batch_size=args.batch_size,
        per_step_cat_features=per_step_cat_features
        )   
    loss_object = WeightedCategoricalCrossentropy(ratio=WeightedCategoricalCrossentropy.calculate_ratio(train_labels))
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    train_val = TrainVal(model,loss_object,optimizer)

    # add early stopping
    #early_stopping = EarlyStopping(patience=5)
    # initialize metrics
    metrics_manager = MetricsManager(current_time)


    for epoch in range(args.epochs):
        # Reset the metrics at the start of the next epoch
        metrics_manager.reset_metrics()

        for kpis, labels in train_ds:
            loss, predictions =  train_val.train_step(kpis, labels)
            metrics_manager.update_train_meteric(loss,labels,predictions)

        for kpis, labels in val_ds:
            loss, predictions =  train_val.validation_step(kpis, labels)
            metrics_manager.update_validation_meteric(loss,labels,predictions)

        # Write metrics to tensorboards
        metrics_manager.write_metrics(epoch)

        print(
            f'Epoch {epoch + 1}, \n'
            f'Train Loss: {metrics_manager.train_metrics["loss"].result()}, \n'
            f'Train F1score 0: {MetricsManager.calculate_f1score(metrics_manager.train_metrics["precision_0"].result(),metrics_manager.train_metrics["recall_0"].result()) *100}, \n'    
            f'Train F1score 1: {MetricsManager.calculate_f1score(metrics_manager.train_metrics["precision_1"].result(),metrics_manager.train_metrics["recall_1"].result()) *100}, \n'
            f'Validation Loss: {metrics_manager.validation_metrics["loss"].result()}, \n'
            f'Validation F1score 0: {MetricsManager.calculate_f1score(metrics_manager.validation_metrics["precision_0"].result(),metrics_manager.validation_metrics["recall_0"].result()) *100}, \n'    
            f'Validation F1score 1: {MetricsManager.calculate_f1score(metrics_manager.validation_metrics["precision_1"].result(),metrics_manager.validation_metrics["recall_1"].result()) *100}, \n'
        )

        # check for early stopping
        #if early_stopping(metrics_manager.validation_metrics["loss"].result()):
        #    break

    # save model
    model.save(f"../models/{args.model}_{current_time}")

    
if __name__ == '__main__':
    main()