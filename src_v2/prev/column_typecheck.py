import pandas as pd

def kpi_feature_typecasting(rl_kpis):
    # properly cast scalibility_score as continuous feature
    if 'scalibility_score' in list(rl_kpis.columns):
        rl_kpis['scalibility_score'] = rl_kpis['scalibility_score'].astype('float32')
        print('scalability score feature was casted as float32')

    # typecast all categorical features to type object
    rl_kpis = rl_kpis.astype({
        "type":"object",
        "tip":"object",
        "mlid":"object",
        "site_id":"object",
        "card_type":"object",
        "adaptive_modulation":"object",
        "freq_band":"object",
        "modulation":"object"
        })

    # typecast all numerical features to type float32
    rl_kpis = rl_kpis.astype({
        "severaly_error_second":"float32",
        "error_second":"float32",
        "unavail_second":"float32",
        "avail_time":"float32",
        "bbe":"float32",
        "rxlevmax":"float32",
        "capacity":"float32"
        })

    # remove time component from datetime feature
    rl_kpis['datetime'] = pd.to_datetime(rl_kpis['datetime'].dt.date)

    return rl_kpis

def rl_sites_feature_typecasting(rl_sites):
    # typecast all categorical features to type object
    rl_sites = rl_sites.astype({"site_id":"object", "clutter_class":"object"})

    # typecast all numerical features to type float32
    rl_sites = rl_sites.astype({"groundheight":"float32"})

    return rl_sites


def met_forecast_typecasting(met_forecast):
    # typecast all categorical features to type object
    met_forecast = met_forecast.astype({
        "station_no":"object", 
        "weather_day1":"object", 
        "weather_day2":"object",
        "weather_day3":"object",
        "weather_day4":"object",
        "weather_day5":"object",
        })

    # typecast all numerical features to type float32
    for i in range(1,6):
        met_forecast = met_forecast.astype({
            f"temp_max_day{i}":"float32",
            f"temp_min_day{i}":"float32",
            f"humidity_max_day{i}":"float32",
            f"humidity_min_day{i}":"float32",
            f"wind_dir_day{i}":"float32",
            f"wind_speed_day{i}":"float32",
            })
    
    # remove time component from datetime feature
    met_forecast['datetime'] = pd.to_datetime(met_forecast['datetime'].dt.date)

    return met_forecast


def met_real_typecasting(met_real):
    # typecast all categorical features to type object
    met_real = met_real.astype({"station_no":"object"})

    # typecast all numerical features to type float32
    met_real = met_real.astype({
        "temp":"float32", 
        "temp_max":"float32",
        "temp_min":"float32",
        "wind_dir":"float32",
        "wind_dir_max":"float32",
        "wind_speed":"float32",
        "wind_speed_max":"float32",
        "humidity":"float32",
        "precipitation":"float32",
        "precipitation_coeff":"float32",
        "pressure":"float32",
        "pressure_sea_level":"float32",
        })
    
    # remove time component from datetime feature
    met_real['datetime'] = pd.to_datetime(met_real['datetime'].dt.date)

    return met_real

def met_stations_typecasting(met_stations):
    # typecast all categorical features to type object
    met_stations = met_stations.astype({"station_no":"object", "clutter_class":"object"})

    # typecast all numerical features to type float32
    met_stations = met_stations.astype({"height":"float32"})
    
    return met_stations

def distances_typecasting(distances):
    # typecast all numerical features to type float32
    distances = distances.astype("float32")

    return distances

if __name__ == '__main__':
    pass
