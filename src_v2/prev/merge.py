

def merge_tables(rl_kpis, rl_sites, distances, met_forecast, met_stations,
                 met_real, include_forecast, include_real):

    ###### merge rl-kpi with rl-sites ######
    rl_kpis = rl_kpis.merge(rl_sites,
                            how="inner",
                            left_on=("site_id"),
                            right_on=("site_id"),
                            suffixes=("", "_rl_site")
                            )

    ###### merge rl-kpi with distances ######
    # merge rl_kpi table with distances table using rl-site ids
    # this way all sites have closest weather station id and distance
    # handle rl-site association with weather stations
    rl_kpis = rl_kpis.merge(distances,
                            how="inner",
                            left_on=("site_id"),
                            right_on=("RL_Sites"),
                            suffixes=("", "_distance")
                            )
    #print(f"merged rl_kpis_with_labels + distance shape: {rl_kpis.shape}")

    if include_forecast:
        ###### merge met-forecast with met_stations ######
        met_forecast = met_forecast.merge(met_stations,
                                          how="inner",
                                          left_on=("station_no"),
                                          right_on=("station_no"),
                                          suffixes=("", "_met_station")
                                          )

        ###### merge rl_sites+distance with ws_sites ######
        # merge rl-kpi table with ws-forecast on closest ws
        # this way closest ws-forecast data is considered for each rl-site
        rl_kpis = rl_kpis.merge(met_forecast,
                                how="inner",
                                left_on=("closest_WS", "datetime"),
                                right_on=("station_no", "datetime"),
                                suffixes=("", "_forecast")
                                )
        #print(f"merged rl_kpis_with_labels + distance + met_forecast shape: {rl_kpis.shape}")

    if include_real:
        if include_forecast == False:
            ###### merge met-real with met_stations ######
            met_real = met_real.merge(met_stations,
                                      how="inner",
                                      left_on=("station_no"),
                                      right_on=("station_no"),
                                      suffixes=("", "_met_station")
                                      )

        ###### merge rl_sites with ws_real ######
        rl_kpis = rl_kpis.merge(met_real,
                                how="inner",
                                left_on=("closest_WS", "datetime"),
                                right_on=("station_no", "datetime"),
                                suffixes=("", "_real")
                                )

    return rl_kpis


if __name__ == '__main__':
    pass
