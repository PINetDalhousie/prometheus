# ...existing code...
import csv
import argparse
import math

import numpy as np
import pandas as pd

def main():

    for i in range(6):
        print(i)
        '''
        p1 = pd.read_csv(f'/home/papry/Exp_Project/MS_thesis-main/logs/20250524_004740/20251025_023538_results_time{i}.csv')
        p0 = pd.read_csv(f'/home/papry/Exp_Project/MS_thesis-main/logs/20250525_000127/20251025_011446_results_time{i}.csv')
        p4 = pd.read_csv(f'/home/papry/Exp_Project/MS_thesis-main/logs/20250526_011406/20251025_032524_results_time{i}.csv')
        p3= pd.read_csv(f'/home/papry/Exp_Project/MS_thesis-main/logs/20250527_012340/20251025_145446_results_time{i}.csv')
        p2= pd.read_csv(f'/home/papry/Exp_Project/MS_thesis-main/logs/20250527_233848/20251025_125426_results_time{i}.csv')
        # first column as strings (will capture quoted "" as empty string)
        '''
        p1 = pd.read_csv(f'/home/papry/Exp_Project/MS_thesis-main/logs/20250524_004740/20251026_015940_deletion_results_time{i}.csv')
        p0 = pd.read_csv(f'/home/papry/Exp_Project/MS_thesis-main/logs/20250525_000127/20251026_083710_deletion_results_time{i}.csv')
        p4 = pd.read_csv(f'/home/papry/Exp_Project/MS_thesis-main/logs/20250526_011406/20251026_093630_deletion_results_time{i}.csv')
        p3= pd.read_csv(f'/home/papry/Exp_Project/MS_thesis-main/logs/20250527_012340/20251026_131547_deletion_results_time{i}.csv')
        p2= pd.read_csv(f'/home/papry/Exp_Project/MS_thesis-main/logs/20250527_233848/20251026_225814_deletion_results_time{i}.csv')
        # first column as strings (will capture quoted "" as empty string)
        first_col = p1.iloc[:, 0].astype(str).tolist()

        # numeric part: columns after the first
        a1 = p1.iloc[:, 6].to_numpy(dtype=float)
        a0 = p0.iloc[:, 6].to_numpy(dtype=float)
        a4 = p4.iloc[:, 6].to_numpy(dtype=float)
        a3 = p3.iloc[:, 6].to_numpy(dtype=float)
        a2 = p2.iloc[:, 6].to_numpy(dtype=float)
        print(a2)
        print(a3)
        print(a4)
        print(a0)
        print(a1)   
        nrows = min(len(a1), len(a0), len(a4), len(a3), len(a2))
        print("nrows:", nrows)
        means = []
        for i in range(nrows):
            mean_val = (a1[i] + a0[i] + a4[i] + a3[i] + a2[i]) / 5.0
            means.append(mean_val)
        # print all means in a single row (space-separated), formatted to 2 decimals
        print(" ,".join(f"{v:.2f}" for v in means))

if __name__ == '__main__':
    main()