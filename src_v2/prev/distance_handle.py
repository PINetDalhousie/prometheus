from process_tables import process_distance_table
from utility import get_closest_ws
import pandas as pd

def generate_min_rl2ws_dist(dataset):
    if dataset == 'prev':
        data_zip_path = "../data/train/prev_data.zip"
    elif dataset == 'new':
        data_zip_path = "../data/train/RegionA.zip"
    
    distances = process_distance_table(data_zip_path)
    distances = get_closest_ws(distances)
    pd.set_option('display.max_rows', None)
    # sort values based on min_distance
    distances = distances.sort_values(by = ['min_distance'])
    print(distances)

    #figure = distances["min_distance"].hist(bins=100)
    figure = distances.plot.scatter(x='RL_Sites', y='min_distance')

    # plot closest ws figure before thresholding
    figure.get_figure().savefig(f'../report/{dataset}_closest_ws_beforeTH.png')

    # remove rows based on thresholding for prev dataset
    if dataset == 'prev':
        distances = distances[distances['min_distance'] < 200]

        # plot closest ws figure after thresholding
        figure = distances.plot.scatter(x='RL_Sites', y='min_distance')
        figure.get_figure().savefig(f'../report/{dataset}_closest_ws_afterTH.png')

    # get the maximum of the minimum distances
    print(f"{dataset} dataset min max optimal distance {distances['min_distance'].max()}")

if __name__ == '__main__':
    generate_min_rl2ws_dist(dataset="new")
    pass
