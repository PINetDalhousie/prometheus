import os
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import csv
import matplotlib.pyplot as plt
import math
import warnings
from sklearn.metrics import classification_report
import seaborn as sns
warnings.simplefilter(action='ignore', category=UserWarning)

def avg_filter_multi_ws_predictions(df):
    ''' One radio site can be associated with multiple weather stations at test time.
    In order to compute probability predictions for one site at one datetime, this
    function calculates the average over predictions per day per mini link.

    Args:
        df: dataframe with multiple predictions per link per day
    '''
    #df = pd.read_csv("../report/tensorboard/2022-11-23_12:00:21.401611/test_csv.csv")

    df_pred = df[['datetime','RL_Sites','mlid','predicted']].groupby(['RL_Sites','mlid','datetime']).mean()
    df_pred = df_pred.reset_index()

    df_label = df[['datetime','RL_Sites','mlid','labels']].groupby(['RL_Sites','mlid','datetime']).mean()
    df_label = df_label.reset_index()

    df = df_pred.merge(df_label,
                        how = "inner",
                        left_on = ('RL_Sites','mlid','datetime'),
                        right_on = ('RL_Sites','mlid','datetime'),
                        )
    return df


def write_predictions(reconstruction_losses, Y, predict_scores, clf_1_day_pred, 
    mode, identifiers):
    ''' predictions are saved to csv files, along with labels and binary predictions
    '''
    # write data with identifiers if mode is test
    if mode == 'test':
        df = pd.DataFrame(data={'datetime':identifiers['datetime'], 'RL_Sites':identifiers['RL_Sites'],
            'mlid':identifiers['mlid'], 'predicted': reconstruction_losses, 
            'labels': Y, 'binary_pred': predict_scores})

        df = avg_filter_multi_ws_predictions(df)
    else:
        df = pd.DataFrame(data={'predicted': reconstruction_losses, 
            'labels': Y, 'binary_pred': predict_scores})

    df.to_csv(f'../report/tensorboard/{clf_1_day_pred.time}/{mode}_csv.csv')


def report_performance_of_trained_model(clf_1_day_pred, model, X, Y, 
        recons_prctl_thresh, pred_threshold, mode, identifiers=None):
    '''Create classification report using a trained model and labels
    '''

    print(f'starting inference on {mode} data')
    if model == "LSTM_AE":
        reconstruction_losses, predict_scores = clf_1_day_pred.predict_proba(X, recons_prctl_thresh)
    else:
        reconstruction_losses, predict_scores = clf_1_day_pred.predict_proba(X, pred_threshold)

    # calculate roc auc for validation or test data
    #if mode != 'train':
    #    clf_1_day_pred.get_roc_auc()


    print(f'finished inference on {mode} data')
    print(f"predict scores {predict_scores} and shape {predict_scores.shape}")
    print(f"labels values {Y} and shape {Y.shape}")

    print(f'Evaluation metrics on {mode} Data')
    print(classification_report(Y,predict_scores,
                                labels=[0,1],
                                target_names=['non-failure','failure']))

    performance_report = classification_report(Y,predict_scores,
                                labels=[0,1],
                                target_names=['non-failure','failure'],
                                output_dict=True)
    print('finished reporting evaluation mertrics')

    write_predictions(reconstruction_losses, Y, predict_scores, clf_1_day_pred, mode, identifiers)

    return performance_report


def report_evaluation_metrics(y_true, y_pred, report, hparam):
    precision1, recall1, fscore1, _ = precision_recall_fscore_support(y_true, 
                                                               y_pred, 
                                                               average="binary", # 
                                                               labels=[0, 1], # labels
                                                               beta=1) # f1 score
    
    print(f"*********** {report} SCORE for 1-DAY predict")
    print(f"precision : {precision1:.4f}")
    print(f"recall    : {recall1:.4f}")
    print(f"f-score   : {fscore1:.4f}")
    
    if (report=="Train") and (not os.path.exists("../report/eval_metrics/results.csv")):
        with open(f"../report/eval_metrics/results.csv", 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(["type", "precision", "recall", "f-score", "over_val"])
            writer.writerow([report, precision1, recall1, fscore1, hparam])
    else :
        with open(f"../report/eval_metrics/results.csv", 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([report, precision1, recall1, fscore1, hparam])

def visualize_topology(distances):
    #distance = distance.to_numpy()
    G = nx.from_pandas_adjacency(distances)
    G.name = "Graph from pandas adjacency matrix"
    print(nx.info(G))
    nx.write_gexf(G, "../report/test.gexf")
    #print(list(G.edges))
    
def continuous_hist(df, table_name):
    ''' Save the data distribution of the continuous features in df
    '''
    # make directory if needed
    os.system(f"mkdir -p ../report/data_dist/{table_name}_continuous_dist")
    cont_features = df.select_dtypes('number')
    for feature in cont_features:
        figure = df[feature].hist(bins=100) 
        figure.get_figure().savefig(f'../report/data_dist/{table_name}_continuous_dist/{feature}.png')
        plt.close()

def categorical_bar(df, table_name):
    ''' Save the data distribution of the cateogrical features in df
    '''
    # make directory if needed
    os.system(f"mkdir -p ../report/data_dist/{table_name}_categorical_bars")

    cat_features = df.select_dtypes(include='object').columns.to_list()
    for feature in cat_features:
        plt.subplots(constrained_layout=True)
        figure = df[feature].value_counts()[0:10].plot.bar()
        figure.get_figure().savefig(f'../report/data_dist/{table_name}_categorical_bars/{feature}.png')
        plt.close()

def box_plot(df, table_name):
    '''save box plot of the continuoys features in df
    '''
    # make directory if needed
    os.system(f"mkdir -p ../report/box_plots/{table_name}_continuous_dist")
    cont_features = df.select_dtypes('number')
    for feature in cont_features:
        figure = sns.boxplot(x=df[feature])
        figure.get_figure().savefig(f'../report/box_plots/{table_name}_continuous_dist/{feature}.png')
        plt.close()
    
def calculate_num_outliers(df, table_name):
    '''calcualte the number of outliers based on 3 standard deviation for continuos features
    '''
    cont_features = df.select_dtypes('number')
    feature_name = []
    num_of_outliers = []
    outlier_percentage = []
    upper = []
    lower = []
    for feature in cont_features:
        upper_limit = df[feature].mean() + 3 * df[feature].std()
        lower_limit = df[feature].mean() -3 * df[feature].std()

        upper.append(upper_limit)
        lower.append(lower_limit)
        feature_name.append(feature)
        
        outlier_rows = df[~((df[feature] < upper_limit) & (df[feature] > lower_limit))]
        df = df.drop(outlier_rows.index.values.tolist())
        num_of_outliers.append(outlier_rows.shape[0])
    
    # save dataframe of number of outliers
    pd.DataFrame(data={'feature':feature_name,'num_outliers':num_of_outliers, 'upper_limit':upper, 'lower_limit':lower}).to_csv(f'../report/box_plots/outliers_{table_name}.csv')
    print('number of outliers for each feature saved in csv file')

if __name__ == '__main__':
    avg_filter_multi_ws_predictions()
    pass
