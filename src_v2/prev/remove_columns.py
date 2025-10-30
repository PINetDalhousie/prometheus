def kpi_remove_features(rl_kpis, data_type):
    ''' remove unnecessary features. There are features that will not be used in
    any of the data preprocessing and model training process. So, we remove these 
    features at the very begining of the data processing step
    '''
    if data_type == "train":
        # fix scalibility_score wrong typecasting by removing inconsistent data point
        if 'scalibility_score' in list(rl_kpis.columns):
            if rl_kpis.loc[686810]['scalibility_score'] == '2025-01-01 00:00:00':
                rl_kpis = rl_kpis.drop(686810)
    
    # remove mw_connection_no feature as it's an unnecessary identifier
    rl_kpis = rl_kpis.drop(["mw_connection_no"], axis=1)
    return rl_kpis

def forecast_remove_features(met_forecast):
    ''' remove report_time feature as it's never used in the following steps
    '''
    # remove the report_time column
    met_forecast = met_forecast.drop(['report_time'], axis=1)
    return met_forecast

def real_remove_features(met_real):
    # wind_dir_max column has some values as dtype string. These string values are more than 360
    # -the maximum allowable wind direction- and so we remove them.
    met_real["wind_dir_max"] = met_real["wind_dir_max"].replace(regex=r"[a-zA-Z,;]", value=None)

    # pressure values have commas in them and so we remove these commas
    met_real["pressure"] = met_real["pressure"].replace(regex=r"[a-zA-Z,;]", value="")
    
    # pressure values have commas in them and so we remove these commas
    met_real["pressure_sea_level"] = met_real["pressure_sea_level"].replace(regex=r"[a-zA-Z,;]", value="")

    # remove unnecessary date and time columns: (measured_date and measured_hour)
    removed_datetime_features = ["measured_date", "measured_hour"]
    for feature in removed_datetime_features:
        if feature in met_real.columns.tolist():
            met_real = met_real.drop([feature], axis=1)
    print(f"removed date time columns: measured_date and measured_hour")

    return met_real


if __name__ == '__main__':
    pass
