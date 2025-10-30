from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
import pandas as pd
from report import continuous_hist, categorical_bar

def preprocessing(df, df_Y, numerical_columns=[], categorical_columns=[], one_hot_encoder=None,
                 min_max_scalar=None, model_name=None):
    
    if (one_hot_encoder is None):
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        one_hot_encoder.fit(df[categorical_columns])

    arr_numerical = np.array(df[numerical_columns], dtype=np.float32)
    arr_categorical = np.array(one_hot_encoder.transform(df[categorical_columns]),
                        dtype=np.float32)
    
    df = np.concatenate((arr_numerical, arr_categorical), axis=1, dtype=np.float32)


    if (min_max_scalar is None):
        if model_name == "LSTM_AE":
            print(f"using only non-failure link data for standardization")
            df_fit = pd.DataFrame(df)
            # filter only non-failure links
            df_fit['1-day-predict'] = df_Y
            df_fit = df_fit[df_fit['1-day-predict'] == 0]
            df_fit = df_fit.drop(['1-day-predict'],axis=1)
            df_fit = df_fit.to_numpy()
        else :
            df_fit = df

        # normalize data to have the same scale
        min_max_scalar = MinMaxScaler()
        min_max_scalar.fit(df_fit)

    df = min_max_scalar.transform(df)

    return df, one_hot_encoder, min_max_scalar

def inverse_processing(X, one_hot_encoder, min_max_scalar):
    print(f"before inverse scaling shape {X.shape}")
    X = min_max_scalar.inverse_transform(X)
    X = one_hot_encoder.inverse_transform(X)
    return X

if __name__ == '__main__':
    pass
