import os
import pandas as pd
from zipfile import ZipFile

# build feature report for continuous features
def continuous_feature_report(features, data_df, table):
    conHead = ['Count', 'Miss %', 'Card.', 'Min', '1st Qrt.',
            'Mean', 'Median', '3rd Qrt', 'Max', 'Std. Dev.']

    conOut_df = pd.DataFrame(index=features, columns=conHead)
    columns_df = data_df[features]

    #COUNT
    conOut_df[conHead[0]] = len(columns_df)

    #MISS % 
    conOut_df[conHead[1]] = columns_df.isna().sum() / len(columns_df) * 100

    #CARDINALITY
    conOut_df[conHead[2]] = columns_df.nunique()

    #MINIMUM
    conOut_df[conHead[3]] = columns_df.min()

    #1ST QUARTILE
    conOut_df[conHead[4]] = columns_df.quantile(0.25)

    #MEAN
    conOut_df[conHead[5]] = columns_df.mean()

    #MEDIAN
    conOut_df[conHead[6]] = columns_df.median()

    #3rd QUARTILE
    conOut_df[conHead[7]] = columns_df.quantile(0.75)

    #MAX
    conOut_df[conHead[8]] = columns_df.max()

    #STANDARD DEVIATION
    conOut_df[conHead[9]] = columns_df.std()

    conOut_df.to_csv(os.path.join(f"../report/data_stats/{table}_contFeatureReport.csv"))

    
    
# build feature report for categorical features
def categorical_feature_report(features, data_df, table):
    catHead = ['Count', 'Miss %', 'Card.', 'Mode', 'Mode Freq',
            'Mode %', '2nd Mode', '2nd Mode Freq', '2nd Mode %']

    columns_df = data_df[features]

    #preparing a dictionary for storing data
    stats_dict = {k: ['']*len(features) for k in catHead}

    #CARDINALITY
    stats_dict['Card.'] = columns_df.nunique()

    missing = columns_df.isna().sum() / len(columns_df) * 100

    for col in columns_df:
        values = columns_df[col].value_counts()
        index = features.index(col)

        #COUNT
        stats_dict['Count'][index] = len(columns_df)

        #MISS %
        stats_dict['Miss %'][index] = missing[col]

        #MODES
        mode = values.index[0]
        mode2 = values.index[1] if len(values.index) > 1 else mode
        stats_dict['Mode'][index] = mode
        stats_dict['2nd Mode'][index] = mode2

        #MODE FREQ
        modeCount = values.loc[mode]
        modeCount2 = values.loc[mode2]
        stats_dict['Mode Freq'][index] = modeCount
        stats_dict['2nd Mode Freq'][index] = modeCount2

        #MODE %
        miss = stats_dict['Miss %'][index]


        modePer = (modeCount/(len(columns_df)*((100-miss)/100)))*100
        stats_dict['Mode %'][index] = round(modePer, 2)

        modePer2 = (modeCount2/(len(columns_df)*((100-miss)/100)))*100
        stats_dict['2nd Mode %'][index] = round(modePer2, 2)

    output_df = pd.DataFrame.from_dict(stats_dict)
    output_df.to_csv(os.path.join(f"../report/data_stats/{table}_catFeatureReport.csv"))

def read_table_from_zip(zip_path, table_name):
    """
    read dataframe from zip file which contains multiple 
    tab seperated text files
    
    Parameters: 
        zip_path - string
            path to regionA zip file
        table_name - array-like of shape (n_samples,)
            Corresponding label for each sample in X.
    Returns:
        table_from_zip - pandas dataframe
            one of the tables from zip file
    """
    with ZipFile(zip_path) as zip_file:
        with zip_file.open(table_name+".tsv") as file:
            df = pd.read_csv(file, sep="\t", index_col=0, low_memory=False)
            # if datetime column is present, convert it into date type
            if "datetime" in df:
                df["datetime"] = pd.to_datetime(df["datetime"])
            return df

        
        
def get_kpi_feature_reports(rl_kpis, rlf_separate):
    # get continuous and categorical feature reports for rl-kpis data
    continuous_feature_report(rl_kpis.select_dtypes('number').columns.to_list(), rl_kpis, "rl_kpis")
    categorical_feature_report(rl_kpis.select_dtypes(include=['object','bool']).columns.to_list(), rl_kpis, "rl_kpis")
    
    if rlf_separate == True:
        # get continuous and categorical feature reports for failure rl-kpis data
        continuous_feature_report(rl_kpis[rl_kpis["rlf"]==True].select_dtypes('number').columns.to_list(), 
                                  rl_kpis[rl_kpis["rlf"]==True], "fail_rl_kpis")
        categorical_feature_report(rl_kpis[rl_kpis["rlf"]==True].select_dtypes(include=['object','bool']).columns.to_list(),
                                   rl_kpis[rl_kpis["rlf"]==True], "fail_rl_kpis")

        # get continuous and categorical feature reports for failure rl-kpis data
        continuous_feature_report(rl_kpis[rl_kpis["rlf"]==False].select_dtypes('number').columns.to_list(), 
                                  rl_kpis[rl_kpis["rlf"]==False], "non_fail_rl_kpis")
        categorical_feature_report(rl_kpis[rl_kpis["rlf"]==False].select_dtypes(include=['object','bool']).columns.to_list(),
                                   rl_kpis[rl_kpis["rlf"]==False], "non_fail_rl_kpis")
    
def get_forecast_feature_reports(met_forecast):
    # get continuous and categorical feature reports for rl-kpis data
    continuous_feature_report(met_forecast.select_dtypes('number').columns.to_list(), met_forecast, "met_forecast")
    categorical_feature_report(met_forecast.select_dtypes(include=['object','bool']).columns.to_list(), met_forecast, "met_forecast")
    
    
def get_input_feature_reports(df, data_type):
    continuous_feature_report(df.select_dtypes('number').columns.to_list(), df, data_type)
    categorical_feature_report(df.select_dtypes(include=['object','bool']).columns.to_list(), df, data_type)
    
def get_site_feature_reports(sites, site_type):
    continuous_feature_report(sites.select_dtypes('number').columns.to_list(), sites, site_type)
    categorical_feature_report(sites.select_dtypes(include=['object','bool']).columns.to_list(), sites, site_type)
        
if __name__ == '__main__':
    pass
