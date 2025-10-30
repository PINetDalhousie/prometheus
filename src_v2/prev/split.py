import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit

def split_dataset(method, X, Y, val_size):
    if method=="random":
        X_train, X_val, Y_train, Y_val = random_split(X, Y, val_size)
    elif method=="time_series_split":
        X_train, X_val, Y_train, Y_val = time_series_split(X, Y, val_size)
    elif method=="temporal_conservative_split":
        X_train, X_val, Y_train, Y_val = temporal_conservative_split(X, Y, val_size)
    
    return X_train, X_val, Y_train, Y_val

def random_split(X, Y, val_size):
    # drop identifier column
    X = rl_kpis.drop(["datetime"], axis=1)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val_size, random_state = 42)
    return X_train, X_val, Y_train, Y_val

def time_series_split(X, Y, val_size):
    X['1-day-predict'] = Y
    min_date = X.datetime.min()
    max_date = X.datetime.max()
    #print("Min:", min_date, "Max:", max_date)

    train_percent = (1-val_size)
    time_between = max_date - min_date
    train_cutoff = min_date + train_percent*time_between

    train_df = X[X.datetime <= train_cutoff]
    val_df = X[X.datetime > train_cutoff]
    print("Train:", train_df.datetime.min(), train_df.datetime.max())
    print("Val:", val_df.datetime.min(), val_df.datetime.max())

    # shuffle train data
    train_df = train_df.sample(frac = 1)

    # remove datetime column as it's not needed anymore
    train_df = train_df.drop(['datetime'], axis=1)
    val_df = val_df.drop(['datetime'], axis=1)

    X_train = train_df.drop(['1-day-predict'],axis=1)
    Y_train = train_df['1-day-predict']
    Y_train = Y_train.astype('int').to_numpy()
    X_val = val_df.drop(['1-day-predict'],axis=1)
    Y_val = val_df['1-day-predict']
    Y_val = Y_val.astype('int').to_numpy()

    return X_train, X_val, Y_train, Y_val

def temporal_conservative_split(X, Y, val_size):
    # concatenate label to train (X) dataframe
    X['1-day-predict'] = Y

    # calculate index to split at to get train and val set
    train_percent = (1-val_size)
    train_split_index = int(train_percent*len(X))

    # sort values by datetime and split usint train_split_index
    X = X.sort_values(by="datetime") 
    train_df = X.iloc[:train_split_index]
    val_df = X.iloc[train_split_index:]

    print("Train:", train_df.datetime.min(), train_df.datetime.max())
    print("Val:", val_df.datetime.min(), val_df.datetime.max())
    # shuffle train data
    train_df = train_df.sample(frac = 1,random_state=42)

    # remove datetime column as it's not needed anymore
    train_df = train_df.drop(['datetime'], axis=1)
    val_df = val_df.drop(['datetime'], axis=1)

    X_train = train_df.drop(['1-day-predict'],axis=1)
    Y_train = train_df['1-day-predict']
    Y_train = Y_train.astype('int').to_numpy()
    X_val = val_df.drop(['1-day-predict'],axis=1)
    Y_val = val_df['1-day-predict']
    Y_val = Y_val.astype('int').to_numpy()

    return X_train, X_val, Y_train, Y_val

class TemporalKFold:
    def __init__(self, X, Y, val_size):
        self.current_split_index = None
        self.X = X
        self.Y = Y
        self.val_size = None
        self.num_val_samples = int(val_size*(len(X)))
        self.train_end_index = int((1-val_size)*len(X))
        self.val_end_index = self.train_end_index + self.num_val_samples

        # concatenate label to train (X) dataframe
        self.X['1-day-predict'] = Y
        # sort values by datetime and split usint train_split_index
        self.X = self.X.sort_values(by="datetime")

    def get_split(self,num_split):
        if num_split != 0:
            # calculate index to split at to get train and val set
            self.train_end_index = self.train_end_index - self.num_val_samples
            self.val_end_index = self.val_end_index - self.num_val_samples
         
        train_df = self.X.iloc[:self.train_end_index]
        val_df = self.X.iloc[self.train_end_index:self.val_end_index]

        print("Train:", train_df.datetime.min(), train_df.datetime.max())
        print("Val:", val_df.datetime.min(), val_df.datetime.max())
        # shuffle train data
        train_df = train_df.sample(frac = 1, random_state=42)

        # remove datetime column as it's not needed anymore
        train_df = train_df.drop(['datetime'], axis=1)
        val_df = val_df.drop(['datetime'], axis=1)

        X_train = train_df.drop(['1-day-predict'],axis=1)
        Y_train = train_df['1-day-predict']
        Y_train = Y_train.astype('int').to_numpy()
        X_val = val_df.drop(['1-day-predict'],axis=1)
        Y_val = val_df['1-day-predict']
        Y_val = Y_val.astype('int').to_numpy()

        return X_train, X_val, Y_train, Y_val
        
