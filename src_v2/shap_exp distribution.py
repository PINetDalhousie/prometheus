
# %%
# %%
import os
import sklearn
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import pandas as pd
uva=[]
lst_data = []
lst_shap = []
for i in range(24):
    shap=np.loadtxt(f'/home/papry/Exp_Project/MS_thesis-main/logs/20250909_132157/input{i}_shap_data.txt')

    data = np.loadtxt(f'/home/papry/Exp_Project/MS_thesis-main/logs/20250909_132157/input{i}_new_data.txt')
    #shap2=np.loadtxt(f'/users/grad/papry/MS_thesis-main/logs/20240831_070447/shap{i}_new_data.txt')
    #data2 = np.loadtxt(f'/users/grad/papry/MS_thesis-main/logs/20240831_070447/input{i}_new_data.txt')

    file_1 = open("/home/papry/Exp_Project/MS_thesis-main/column_name.txt", "r")
    labels= file_1.read()
    file_1.close()
    label_list = labels.split(",")
   # print(label_list)
 
    colors = ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8"]
    my_map = ListedColormap(colors, name="my_map")

    data_values = data.reshape(3,4,78)
    shap_values = shap.reshape(3,4,78)
   # data2_values = data2.reshape(3,4,84)
   # shap2_values = shap2.reshape(3,4,84)
    #data_values = np.concatenate((data_values, data2_values), axis=0)
    #shap_values = np.concatenate((shap_values, shap2_values), axis=0)
    #print(data.shape)
   ##print(shap_values.shape)
    #mpl.colormaps.register(cmap=my_map)
   
    #axs[0, 0].plot(shap_values[0, :, 0], label='shap_values')
    
    feature_wise_importance = shap_values.sum(axis=(0, 1))
    feature_wise_importance.shape
    feature_wise_importance_abs = np.abs(shap_values).sum(axis=(0, 1))
    k=0
    ws_wise_importance_abs = np.abs(shap_values).sum(axis=(2))
    time_wise_input = data_values.sum(axis=(0))
    #print("input shape",time_wise_input.shape)
   # print(time_wise_input[:, 3])
    time_wise_importance = shap_values.sum(axis=(0))
#fig, axs = plt.subplots(1, 1, figsize=(8, 15), sharex=True)
    index=5
    c_index=0
    for j in range(4):
        if(time_wise_input[j, index]>0  ):
            lst_data.append(time_wise_input[j, index])
            lst_shap.append(time_wise_importance[j, index])
            c_index+=1
    print(f"count for {i} is {c_index}")
feature_values, shap_values_feature = np.array(lst_data), np.array(lst_shap)
plt.figure(figsize=(7, 5))
plt.scatter(feature_values, shap_values_feature, alpha=0.6, color='teal', edgecolor='k')
plt.xlabel(f'Feature values ({label_list[index-1]})', fontsize=14, fontweight='bold')
plt.ylabel(f'SHAP values for {label_list[index-1]}', fontsize=14, fontweight='bold')
#plt.title(f'Distribution: SHAP vs Feature {2} ({label_list[2]})', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('distribution_new_bbe_urban_all.pdf', bbox_inches='tight', pad_inches=0.3)
#plt.show()
plt.figure(figsize=(7, 5))
sns.kdeplot(
    x=feature_values, 
    y=shap_values_feature, 
    fill=True, 
    cmap="Blues", 
    thresh=0.05, 
    levels=10
)
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel(f'Feature values ({label_list[index-1]})', fontsize=20, fontweight='bold')
plt.ylabel(f'SHAP values ({label_list[index-1]})', fontsize=20, fontweight='bold')
#plt.title('2D KDE: SHAP vs Feature Value', fontsize=14, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('distribution_kde_prec_urban_all.pdf', bbox_inches='tight', pad_inches=0.3)


plt.figure(figsize=(7, 5))
# Use hist2d for a basic heatmap
plt.hist2d(feature_values, shap_values_feature, bins=30, cmap='Blues')
plt.colorbar(label='Counts')
plt.xlabel(f'Feature values ({label_list[index-1]})', fontsize=20, fontweight='bold')
plt.ylabel(f'SHAP values', fontsize=20, fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('distribution_heatmap_bbe_urban_all.pdf', bbox_inches='tight', pad_inches=0.3)
#plt.show()
plt.show()
