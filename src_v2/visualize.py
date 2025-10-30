import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
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
from losses import TruncatedMSE

def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--approach', type=str, default="prev",
        help=f"String identifying which approach to use. 'prev' indicates "
            f"previous approach with derived features.")
    parser.add_argument('--autoencoder', type=bool, default=True,
                        help='Use autoencoder approach')
    parser.add_argument('--drop_cat_features', type=bool, default=True,
                        help='Drop categorical features')
    parser.add_argument('--undersample', type=float, default=0.1,
                        help='Undersample majority class')
    parser.add_argument('--train_csv',
        type=str,
        default="../data/20230421_190706_prev_rural_cat/train_prev_rural_time0_with_cat.csv",
        help='Path to train csv file')
    parser.add_argument(
        '--val_csv',
        type=str,
        default=f"../data/20230421_190706_prev_rural_cat/validation_prev_rural_time0_with_cat.csv",
        help='Path to validation csv file'
        )
    parser.add_argument('--model', type=str, default="LSTMAutoencoder",
        help='model to use')
    parser.add_argument('--positional_encoding', type=bool, default=False,
        help='Use positional encoding in transformer model')
    parser.add_argument('--batch_size', type=int, default=16,
        help='batch size')
    parser.add_argument('--epochs', type=int, default=1000,
        help='Number of epochs to train model')
    parser.add_argument('--initial_epoch', type=int, default=0,
        help='Epoch to start training from. Required for resuming training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
        help='Learning rate for the optimizer')
    parser.add_argument('--num_stations', type=int, default=3,
        help='Number of K closest weather stations to consider')
    parser.add_argument('--prev_days_data', type=int, default=7,
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
    data_preprocess.read_train_val_csv(args.train_csv,args.val_csv)

if __name__ == '__main__':
    main()