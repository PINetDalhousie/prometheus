from deployment import Deployment
import argparse
from eda import EDA
import argparse
from typecast_features import TypecastFeatures
from fix_features import FixFeatures
from remove_values import RemoveValues
from aggregate_to_daily import AggregateToDaily
from impute_values import ImputeValues
from assign_ws import AssignWS
from merge_table import MergedTable
from merge import Merge
from aggregate_ws import AggregateWS
from split import Split
import datetime
import os
import warnings
import pandas as pd
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--approach',
        type=str,
        default="prev", 
        help=f'"new" and "prev" refers to the new end to end'
        f'and previous derived WS approaches respectively'
        )
    
    parser.add_argument('--deployment', type=str, default="urban",
                        help='"rural" and "urban" refers to the previous rural and new urban deployment respectively')
    parser.add_argument('--drop_train_link_percentage', type=float, default=0.0,
                        help='randomly drop drop_train_link_percentage percent links from train data')
    parser.add_argument('--include_cat_features', type=bool, default=True,
                        help='Flag to include categorical features in output data csv')
    parser.add_argument('--name_ext', type=str, default=None,
                        help='Output file name extension that uniquely identifies the data preprocessing run')
    parser.add_argument('--time_splits', type=int, default=5,
                        help='Number of time series cross validation splits to generate')
    args = parser.parse_args()

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.system(f" mkdir -p /home/papry/Exp_Project/MS_thesis-main/data/{current_time}")


    # get deployment using zip path
    if args.deployment == "rural":
        deployment = Deployment("/home/papry/Exp_Project/MS_thesis-main/train/prev_data.zip", deployment_name=args.deployment)
    elif args.deployment == "urban":
       deployment = Deployment("/home/papry/Exp_Project/MS_thesis-main/train/RegionA.zip", deployment_name=args.deployment)
    
    # add the different tables to deployment 
    deployment.add_table('rl-kpis')
    deployment.add_table('rl-sites')
    deployment.add_table('met-forecast')
    deployment.add_table('met-real')
    deployment.add_table('met-stations')

    deployment.add_table('distances')

    for key, table in deployment.tables.items():
        EDA(f"/home/papry/Exp_Project/MS_thesis-main/data/{current_time}/").write_feature_reports(table)
        table = FixFeatures().fix_table(table)
        table = TypecastFeatures().cast_features(table)
        # remove time component from datetime feature
        table = RemoveValues().remove_time_from_datetime(table)
        # aggregate along time for some tables
        table = AggregateToDaily().aggregate(table)
        # remove columns with high number of missing values
        table = RemoveValues().remove_highly_missing_features(table)
        # impute continuous columns that have low number of missing values
        table = ImputeValues().time_series_imputation(table)
        # impute remaining continuous values using median
        table = ImputeValues().median_imputation(table)
        # impute categorical columns that have low number of missing values
        table = ImputeValues().mode_imputation(table)
        # get box plots for continuos features 
        EDA(f"/home/papry/Exp_Project/MS_thesis-main/data/{current_time}/").box_plot(table)
        # get time series plots for time series features
        # set transformed table back to deployment
        deployment.tables[key] = table

    # generate label column for rl-kpis table
    deployment.tables["rl-kpis"].calculate_label(forecast_days=1)
    # remove weather station rows from distances table
    deployment.tables["distances"].remove_ws_rows()

    # associate radio sites to weather stations
    deployment = AssignWS().k_nearest_method(deployment,num_neighbors=3)

    # merge data based on association
    merged_table = MergedTable(deployment)
    merged_table = Merge().merge(deployment,merged_table)
    
    # remove unnecessary features that do not provide useful 
    merged_table = RemoveValues().remove_unused_features(merged_table)
    
    if args.include_cat_features == False:
        # remove categorical features
        merged_table = RemoveValues().remove_categorical_features(merged_table)
    
    if args.approach == "prev":
        # calculate the derived weather feature values for radio sites (aggregation)
        merged_table = AggregateWS().aggregate(merged_table,
                                               derived_features=['mean','min','max','std'])
        # For unknown reasons the std derived feature can have some null values. We remove them
        merged_table.set_df(merged_table.get_df().dropna())

    # sort columns
    merged_table = AggregateWS().sort_columns(merged_table)
    
    # Get previous day features for merged_table dataframe
    merged_table.get_previous_day_features(num_prev_days=4)
    
    if args.approach == "new":
        # Get all weather station data 
        merged_table.get_assigned_ws_features()
        merged_table.set_df(merged_table.get_df().dropna())

    if args.approach == "prev":
        # remove assigned_WS column from merged table
        merged_table.data_df = merged_table.data_df.drop(["assigned_WS"], axis=1)

    # split into train, validation and test
    train, validation, test = Split().time_series_cross_validation(
        merged_table,
        val_ratio=0.2,
        test_ratio=0.1,
        splits=args.time_splits
        )
    
    # save train validation and test csv files
    print(train[0].info(verbose=True))
    print(f"Number of missing values {train[0].isna().sum().sum()}")

    # Visualize failures against a feature column for a weather station
    #EDA(f"/users/grad/papry/MS_thesis-main/data/{current_time}/").visualize_failures(train,"precipitation_mean")
    #print(asd)
    
    for i in range(args.time_splits):
        if args.drop_train_link_percentage:
            # randomly sample fraction of unique (side_id,mlid) pairs and drop them
            # get the unique (site_id, mlid) pairs and sample 50% of them
            unique_pairs = train[i][["site_id", "mlid"]].drop_duplicates()
            sampled_pairs = unique_pairs.sample(frac=args.drop_train_link_percentage)

            # join the original DataFrame with the sampled pairs and keep only the matching rows
            train[i] = pd.merge(train[i], sampled_pairs, on=["site_id", "mlid"], how="inner")

        # Remove identifiers or
        # Add integer identifier for (site_id,mlid) pairs and integer identifier for datetime
        train[i] = train[i].drop(merged_table.unique_identifiers, axis=1)
        validation[i] = validation[i].drop(merged_table.unique_identifiers, axis=1)
        test[i] = test[i].drop(merged_table.unique_identifiers, axis=1)

        # save 
        train[i].to_csv(f'/home/papry/Exp_Project/MS_thesis-main/data/{current_time}/train_{args.approach}_{args.deployment}_time{i}_{args.name_ext}.csv',index=False)
        validation[i].to_csv(f'/home/papry/Exp_Project/MS_thesis-main/data/{current_time}/validation_{args.approach}_{args.deployment}_time{i}_{args.name_ext}.csv',index=False)
        test[i].to_csv(f'/home/papry/Exp_Project/MS_thesis-main/data/{current_time}/test_{args.approach}_{args.deployment}_time{i}_{args.name_ext}.csv',index=False)


if __name__ == '__main__':
    main()