import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_error_bars():
    var_lstm_plus = [85.23, 91.05, 91.66, 84.45, 77.18]
    lstm_plus = [70.49, 78.37, 71.98, 63.59, 72.14]

    combined_results = pd.DataFrame(np.transpose(np.array([var_lstm_plus, lstm_plus])), columns=['VarLSTMplus', 'LSTMplus'])

    
    fig, ax = plt.subplots()
    combined_results.boxplot(ax=ax)


    plt.savefig('/users/grad/papry/MS_thesis-main/exp/boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

    import scipy.stats as stats
    fvalue, pvalue = stats.f_oneway(var_lstm_plus, lstm_plus)
    print("f score : " + str(fvalue), "p value : " + str(pvalue))

def plot_multi_error_bars():
    gentrap = [0.9183,0.9431,0.9689,0.9166,0.9067]
    genlstmplus = [0.8523,0.9105,0.9105,0.8445,0.7731]
    lstmplus = [0.7049,0.7837,0.7198,0.6359,0.7214]

    combined_results = pd.DataFrame(np.transpose(np.array([gentrap, genlstmplus, lstmplus])), columns=['GenTrap', 'GenLSTMplus', 'LSTMplus'])

    fig, ax = plt.subplots()
    combined_results.boxplot(ax=ax, widths=0.2,positions=[1, 1.3, 1.6])
    plt.savefig('/users/grad/papry/MS_thesis-main/exp/boxplot_new.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_error_bars():
    values = [0.9183,0.9431,0.9689,0.9166,0.9067,0.9363,0.7803,0.6978,0.7360,0.8076,0.8523,0.9105,0.9105,0.8445,0.7731,0.8168,0.7407,0.6560,0.6583,0.6738,0.7049,0.7837,0.7198,0.6359,0.7214,0.7688,0.7441,0.5100,0.6273,0.6257]
    approach = ['GNNTrans','GenTrap','GenTrap','GenTrap','GenTrap','GenTrap','GenTrap','GenTrap','GenTrap','GenTrap','GenLSTM+','GenLSTM+','GenLSTM+','GenLSTM+','GenLSTM+', 'GenLSTM+','GenLSTM+','GenLSTM+','GenLSTM+','GenLSTM+', 'LSTM+','LSTM+','LSTM+','LSTM+','LSTM+','LSTM+','LSTM+','LSTM+','LSTM+','LSTM+']
    deployment = ['Rural','Rural','Rural','Rural','Rural','Urban','Urban','Urban','Urban','Urban','Rural','Rural','Rural','Rural','Rural','Urban','Urban','Urban','Urban','Urban','Rural','Rural','Rural','Rural','Rural','Urban','Urban','Urban','Urban','Urban']

    combined_results = pd.DataFrame(data={'F1-Score':values, 'Approach':approach,'Deployment':deployment})

    # Set the style to "whitegrid" to include gridlines
    sns.set_style("whitegrid")
    # Box plot by group
    sns.boxplot(
        data=combined_results,
        y = 'F1-Score',
        x = 'Deployment',
        hue = 'Approach',
        palette = "Set3"
        )
    plt.savefig('../exp/boxplot_comb.png', dpi=500, bbox_inches='tight')
    plt.close()

def plot_combined_error_bars_GNNTransformer():
    values = [0.9183,0.9431,0.9689,0.9166,0.9067,0.9363,0.7803,0.6978,0.7360,0.8076,0.8523,0.9105,0.9105,0.8445,0.7731,0.8168,0.7407,0.6560,0.6583,0.6738,0.7049,0.7837,0.7198,0.6359,0.7214,0.7688,0.7441,0.5100,0.6273,0.6257]
    approach = ['GNNTransformer','GNNTransformer','GNNTransformer','GNNTransformer','GNNTransformer','GNNTransformer','GNNTransformer','GNNTransformer','GNNTransformer','GNNTransformer','GenLSTM+','GenLSTM+','GenLSTM+','GenLSTM+','GenLSTM+', 'GenLSTM+','GenLSTM+','GenLSTM+','GenLSTM+','GenLSTM+', 'LSTM+','LSTM+','LSTM+','LSTM+','LSTM+','LSTM+','LSTM+','LSTM+','LSTM+','LSTM+']
    deployment = ['Rural','Rural','Rural','Rural','Rural','Urban','Urban','Urban','Urban','Urban','Rural','Rural','Rural','Rural','Rural','Urban','Urban','Urban','Urban','Urban','Rural','Rural','Rural','Rural','Rural','Urban','Urban','Urban','Urban','Urban']

    combined_results = pd.DataFrame(data={'F1-Score':values, 'Approach':approach,'Deployment':deployment})

    # Set the style to "whitegrid" to include gridlines
    sns.set_style("whitegrid")
    # Box plot by group
    sns.boxplot(
        data=combined_results,
        y = 'F1-Score',
        x = 'Deployment',
        hue = 'Approach',
        palette = "Set3"
        )
    plt.savefig('/users/grad/papry/MS_thesis-main/exp/boxplot_comb.png', dpi=500, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    plot_combined_error_bars_GNNTransformer()
    pass
