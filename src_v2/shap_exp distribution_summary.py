
# %%
# %%
import os
import shap
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
lst_data = [[] for _ in range(15)]
lst_shap = [[] for _ in range(15)]
tp=np.loadtxt(f'/home/papry/Exp_Project/MS_thesis-main/logs/20251011_005606/input_new_data.txt') #70
data_1=tp.reshape(-1,3,4,80)
shap_new_data = np.loadtxt(f'/home/papry/Exp_Project/MS_thesis-main/logs/20251011_005606/shap_new_data.txt')
shap_1=shap_new_data.reshape(-1,3,4,80)
print(data_1.shape)
data_4= np.loadtxt(f'/home/papry/Exp_Project/MS_thesis-main/logs/20250909_132157/input_new_data.txt')#33
shap_4= np.loadtxt(f'/home/papry/Exp_Project/MS_thesis-main/logs/20250909_132157/shap_new_data.txt')
data_4=data_4.reshape(-1,3,4,78)
shap_4=shap_4.reshape(-1,3,4,78)
print(data_4.shape)
for i in range(68):
    if i<23:
        shap3=np.loadtxt(f'/home/papry/Exp_Project/MS_thesis-main/logs/20251008_010449/shap{i}_new_data.txt')

        data3 = np.loadtxt(f'/home/papry/Exp_Project/MS_thesis-main/logs/20251008_010449/input{i}_new_data.txt')
    #shap2=np.loadtxt(f'/users/grad/papry/MS_thesis-main/logs/20240831_070447/shap{i}_new_data.txt')
    #data2 = np.loadtxt(f'/users/grad/papry/MS_thesis-main/logs/20240831_070447/input{i}_new_data.txt')

        
    shap2=np.loadtxt(f'/home/papry/Exp_Project/MS_thesis-main/logs/20251009_005125/shap{i}_new_data.txt')

    data2 = np.loadtxt(f'/home/papry/Exp_Project/MS_thesis-main/logs/20251009_005125/input{i}_new_data.txt')
       # shap4=np.loadtxt(f'/home/papry/Exp_Project/MS_thesis-main/logs/20240831_070447/shap{i}_new_data.txt')
        #data4 = np.loadtxt(f'/home/papry/Exp_Project/MS_thesis-main/logs/20240831_070447/input{i}_new_data.txt')

    
   # shap3=np.loadtxt(f'/users/grad/papry/MS_thesis-main/logs/20241001_010049/shap{i}_new_data.txt')
    shap1=shap_1[i]
    data1=data_1[i]
    if i<33:
        data4=data_4[i]
        shap4=shap_4[i]
    file_1 = open("/home/papry/Exp_Project/MS_thesis-main/column_name.txt", "r")
    labels= file_1.read()
    file_1.close()
    label_list = labels.split(",")
    #print(label_list)
 
    colors = ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8"]
    my_map = ListedColormap(colors, name="my_map")
    if i<23:
        data1_values = data1.reshape(3,4,80)
        shap1_values = shap1.reshape(3,4,80)
        data2_values = data2.reshape(3,4,79)
        shap2_values = shap2.reshape(3,4,79)
        data3_values = data3.reshape(3,4,79)
        shap3_values = shap3.reshape(3,4,79)
        shap4_values = shap4.reshape(3,4,78)
        data4_values = data4.reshape(3,4,78)
        shap_values_list = [shap1_values, shap2_values, shap3_values, shap4_values]
        data_values_list = [data1_values, data2_values, data3_values, data4_values]
       #data_values = np.concatenate((data_values, data4_values), axis=0)
        #shap_values = np.concatenate((shap_values, shap4_values), axis=0)
       # data_values = np.concatenate((data_values, data2_values, data3_values), axis=0)
       # shap_values = np.concatenate((shap_values, shap2_values, shap3_values), axis=0)
    elif i<33:
        data1_values = data1.reshape(3,4,80)
        shap1_values = shap1.reshape(3,4,80)
        data2_values = data2.reshape(3,4,79)
        shap2_values = shap2.reshape(3,4,79)
        shap4_values = shap4.reshape(3,4,78)
        data4_values = data4.reshape(3,4,78)
        shap_values_list = [shap1_values, shap2_values, shap4_values]
        data_values_list = [data1_values, data2_values, data4_values]
    else:
        data1_values = data1.reshape(3,4,80)
        shap1_values = shap1.reshape(3,4,80)
        data2_values = data2.reshape(3,4,79)
        shap2_values = shap2.reshape(3,4,79)
        shap_values_list = [shap1_values, shap2_values]
        data_values_list = [data1_values, data2_values]
    #print(data.shape)
   ##print(shap_values.shape)
    #mpl.colormaps.register(cmap=my_map)
   
    #axs[0, 0].plot(shap_values[0, :, 0], label='shap_values')
   # print("data shape",data_values.shape)
    #print("shap shape",shap_values.shape)
    size=shap_values_list.__len__()

    for k in range(size):
            feature_wise_importance1 = shap_values_list[k].sum(axis=(0, 1))
            feature_wise_importance1.shape
            feature_wise_importance1_abs = np.abs(shap_values_list[k]).sum(axis=(0, 1))
            k=0
            #ws_wise_importance_abs = np.abs(shap_values).sum(axis=(2))
            time_wise_input = data_values_list[k].sum(axis=(0))
            #print("input shape",time_wise_input.shape)
        # print(time_wise_input[:, 3])
            time_wise_importance = shap_values_list[k].sum(axis=(0))
        #fig, axs = plt.subplots(1, 1, figsize=(8, 15), sharex=True)
            index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
            #index=[3]
            m=0
            for idx in index:
                sum1=0
                sum2=0
                for j in range(3,4):
                    #if time_wise_input[j, idx]<2:
                        #print("bbe",time_wise_input[j, idx])
                    sum1=time_wise_input[j, idx]
                    sum2=time_wise_importance[j, idx]
                    lst_data[m].append(sum1)
                    lst_shap[m].append(sum2)

                m=m+1
index=14
feature_values, shap_values_feature = np.array(lst_data[index]), np.array(lst_shap[index])
plt.figure(figsize=(7, 5))

num_time_steps=1
feature_idx=3  # Index of the feature to plot (0-based index)
fc=feature_values.reshape(-1,1)
sc=shap_values_feature.reshape(-1,1)
data1_values=fc
shap1_values=sc
plt.figure(figsize=(12, 8))
for t in range(num_time_steps):
    plt.subplot(1, num_time_steps, t+1)
    plt.scatter(
        data1_values[:, t], 
        shap1_values[:, t], 
        alpha=0.6, label=f'Time step {t+1}'
    )
    plt.xlabel('Feature Value', fontsize=12)
    plt.ylabel('SHAP Value', fontsize=12)
    plt.title(f'Time Step {t+1}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('uvatimestep.pdf', bbox_inches='tight', pad_inches=0.3)
'''
plt.scatter(feature_values, shap_values_feature, alpha=0.6, color='teal', edgecolor='k')
plt.xlabel(f'Feature values ({label_list[index-1]})', fontsize=12, fontweight='bold')
plt.ylabel(f'SHAP values for {label_list[index-1]}', fontsize=12, fontweight='bold')
#plt.title(f'Distribution: SHAP vs Feature {2} ({label_list[2]})', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('distribution_new_bbe_rural_all.pdf', bbox_inches='tight', pad_inches=0.3)
#plt.show()
'''

x = np.asarray(feature_values).ravel()
y = np.asarray(shap_values_feature).ravel()

# keep only positive x values (log undefined for <=0)
mask = x > 0
x_pos = x[mask]
y_pos = y[mask]

# log-transform x (base 10)
x_log = np.log10(x_pos)
plt.figure(figsize=(7, 5))
plt.figure(figsize=(7, 5))
sns.kdeplot(
    x=x_log,
    y=y_pos,
    fill=True,
    cmap="Reds",
    thresh=0.01,
    #cbar=True,
    levels=20
)
ax = plt.gca()
xticks_log = np.linspace(x_log.min(), x_log.max(), 4)
ax.set_xticks(xticks_log)
ax.set_xticklabels([f"{10**t:.3f}" for t in xticks_log])
'''
ax = plt.gca()
formatter = ticker.ScalarFormatter(useMathText=True)
#ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
#ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
formatter.set_powerlimits((-2, 2))  # Use scientific notation for small/large values
formatter.set_useOffset(False)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
'''
#plt.ticklabel_format(axis='both', style='sci', scilimits=(-2,2))
plt.xlabel(f'Feature values', fontsize=24, fontweight='bold')
plt.ylabel(f'SHAP values', fontsize=24, fontweight='bold')

plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
#plt.xlim(0, None)

'''


x_data = np.array(x_log)  # or whatever your x-axis data is
xticks = np.linspace(0, x_data.max(), num=5)  # 6 ticks including first and last

# Ensure first and last are included
xticks[0] = 0
xticks[-1] = x_data.max()
xticks = xticks[xticks != 0]
ax.set_xticks(xticks)
y_data = np.array(y_pos)  # or whatever your x-axis data is
yticks = np.linspace(0, y_data.max(), num=5)  # 6 ticks including first and last

# Ensure first and last are included
yticks[0] = 0
yticks[-1] = y_data.max()

ax.set_yticks(yticks)
#ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
'''
#ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
#ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
plt.tight_layout()
#plt.title('2D KDE: SHAP vs Feature Value', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('Final_kde_new_rlmax_urban_Reds.pdf', bbox_inches='tight', pad_inches=0.3)
plt.savefig('distribution_kde_prec_urban_all_inf.pdf', bbox_inches='tight', pad_inches=0.3)
'''
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
plt.savefig('distribution_heatmap_uva_urban_all.pdf', bbox_inches='tight', pad_inches=0.3)

#plt.show()
print(np.array(lst_data).shape)
print(np.array(lst_shap).shape)
shap_val = np.array(lst_shap).T  # shape: (num_samples, 3)
feature_val = np.array(lst_data).T  # shape: (num_samples, 3)

# Provide feature names (replace indices with your actual feature names if available)
feature_names = [label_list[idx-1] for idx in [2,6,7,3,5,11,13,14]]

# Draw summary plot
shap.summary_plot(
    shap_val, 
    feature_val, 
    plot_type="dot", 
    feature_names=feature_names, 
    show=True
)

#plt.xlabel('Feature Value', fontsize=20, fontweight='bold')
plt.xlabel('SHAP value (impact on model output)', fontsize=18, fontweight='bold')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig('summary_plot_8features.pdf', bbox_inches='tight', pad_inches=0.3)


plt.figure(figsize=(7, 5))
for i, fname in enumerate(feature_names):
    y = np.full_like(feature_val[:, i], fill_value=i, dtype=float)
    color = np.where(feature_val[:, i] > 0, 'red', 'blue')
    plt.scatter(shap_val[:, i], y, c=color, alpha=0.7, label=fname if i==0 else "")
plt.yticks(range(len(feature_names)), feature_names, fontsize=14)
plt.xlabel('SHAP value (impact on model output)', fontsize=16, fontweight='bold')
plt.xticks(fontsize=14)
plt.title('Custom SHAP Summary Plot (Red=Positive, Blue=Negative)', fontsize=14)
plt.tight_layout()
plt.savefig('summary_plot_new_color_features.pdf', bbox_inches='tight', pad_inches=0.3)
plt.figure(figsize=(7, 5))

'''
feature_values, shap_values_feature = np.array(lst_data[5]), np.array(lst_shap[5])
plt.figure(figsize=(7, 5))
index=11
plt.scatter(feature_values, shap_values_feature, alpha=0.6, color='teal', edgecolor='k')
plt.xscale('symlog')  # Use 'symlog' for both positive and negative values, or 'log' if all positive
plt.xlabel(f'Feature values ({label_list[index-1]})', fontsize=16, fontweight='bold')
plt.ylabel(f'SHAP values for {label_list[index-1]}', fontsize=16, fontweight='bold')
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('scatter_logscale_temp.pdf', bbox_inches='tight', pad_inches=0.3)
plt.show()