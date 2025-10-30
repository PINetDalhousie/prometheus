
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
import matplotlib.ticker as mticker
uva=[]
sum=np.zeros(84)
sum_abs=np.zeros(84)
count_all=0
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
for i in range(23):
    
            shap3=np.loadtxt(f'/home/papry/Exp_Project/MS_thesis-main/logs/20251008_010449/shap{i}_new_data.txt')

            data3 = np.loadtxt(f'/home/papry/Exp_Project/MS_thesis-main/logs/20251008_010449/input{i}_new_data.txt')
            data3_values = data3.reshape(3,4,79)
            shap3_values = shap3.reshape(3,4,79)
            feature_wise_importance1 = shap3_values.sum(axis=(0, 1))
            feature_wise_importance1.shape
            feature_wise_importance1_abs = np.abs(shap3_values).sum(axis=(0, 1))
            count_all+=1
            #ws_wise_importance_abs = np.abs(shap_values).sum(axis=(2))
            time_wise_input = data3_values.sum(axis=(0))
            #print("input shape",time_wise_input.shape)
        # print(time_wise_input[:, 3])
            time_wise_importance = shap3_values.sum(axis=(0))
        #fig, axs = plt.subplots(1, 1, figsize=(8, 15), sharex=True)
            #index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
            #index=[3]
            for j in range(78):
                if feature_wise_importance1[j]:
                   # print(j, "is", feature_wise_importance1[j])
                    sum[j]+=feature_wise_importance1[j]
                    sum_abs[j]+=feature_wise_importance1_abs[j]
    #shap2=np.loadtxt(f'/users/grad/papry/MS_thesis-main/logs/20240831_070447/shap{i}_new_data.txt')
    #data2 = np.loadtxt(f'/users/grad/papry/MS_thesis-main/logs/20240831_070447/input{i}_new_data.txt')
shap2 = np.loadtxt(f'/home/papry/Exp_Project/MS_thesis-main/logs/20251021_012848/shap_all_new_data.txt')
data2 = np.loadtxt(f'/home/papry/Exp_Project/MS_thesis-main/logs/20251021_012848/input_all_new_data.txt')
shap2=shap2.reshape(-1,3,4,79)
data2=data2.reshape(-1,3,4,79)
for i in range(shap2.shape[0]):   
            shap2_values = shap2[i]
            data2_values = data2[i]
            feature_wise_importance1 = shap2_values.sum(axis=(0, 1))
            feature_wise_importance1.shape
            feature_wise_importance1_abs = np.abs(shap2_values).sum(axis=(0, 1))
            count_all+=1
            #ws_wise_importance_abs = np.abs(shap_values).sum(axis=(2))
            time_wise_input = data2_values.sum(axis=(0))
            #print("input shape",time_wise_input.shape)
        # print(time_wise_input[:, 3])
            time_wise_importance = shap2_values.sum(axis=(0))
        #fig, axs = plt.subplots(1, 1, figsize=(8, 15), sharex=True)
            #index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
            #index=[3]
            for j in range(78):
                if feature_wise_importance1[j]:
                   # print(j, "is", feature_wise_importance1[j])
                    sum[j]+=feature_wise_importance1[j]
                    sum_abs[j]+=feature_wise_importance1_abs[j]
'''
for i in range(69):    
            shap2=np.loadtxt(f'/home/papry/Exp_Project/MS_thesis-main/logs/20251009_005125/shap{i}_new_data.txt')

            data2 = np.loadtxt(f'/home/papry/Exp_Project/MS_thesis-main/logs/20251009_005125/input{i}_new_data.txt')
            # shap4=np.loadtxt(f'/home/papry/Exp_Project/MS_thesis-main/logs/20240831_070447/shap{i}_new_data.txt')
                #data4 = np.loadtxt(f'/home/papry/Exp_Project/MS_thesis-main/logs/20240831_070447/input{i}_new_data.txt')
            data2_values = data2.reshape(3,4,79)
            shap2_values = shap2.reshape(3,4,79)
            d2 = np.asarray(data2_values)
            s2= np.asarray(shap2_values)
            has_nan1 = np.isnan(d2).any()
            has_inf1 = np.isinf(d2).any()
            has_nan = np.isnan(s2).any()
            has_inf = np.isinf(s2).any()
            if has_nan1 or has_inf1:
                continue
            if has_nan or has_inf:
                continue
            feature_wise_importance1 = shap2_values.sum(axis=(0, 1))
            feature_wise_importance1.shape
            feature_wise_importance1_abs = np.abs(shap2_values).sum(axis=(0, 1))
            count_all+=1
            #ws_wise_importance_abs = np.abs(shap_values).sum(axis=(2))
            time_wise_input = data2_values.sum(axis=(0))
            #print("input shape",time_wise_input.shape)
        # print(time_wise_input[:, 3])
            time_wise_importance = shap2_values.sum(axis=(0))
        #fig, axs = plt.subplots(1, 1, figsize=(8, 15), sharex=True)
            #index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
            #index=[3]
            for j in range(78):
                if feature_wise_importance1[j]:
                   # print(j, "is", feature_wise_importance1[j])
                    sum[j]+=feature_wise_importance1[j]
                    sum_abs[j]+=feature_wise_importance1_abs[j]
    #shap2=np.loadtxt(f'/users/grad/papry/MS_thesis-main/logs/202408
   # shap3=np.loadtxt(f'/users/grad/papry/MS_thesis-main/logs/20241001_010049/shap{i}_new_data.txt')
    
'''  
for i in range(33):
            data4=data_4[i]
            shap4=shap_4[i]
            data2_values = data4.reshape(3,4,78)
            shap2_values = shap4.reshape(3,4,78)
            feature_wise_importance1 = shap2_values.sum(axis=(0, 1))
            feature_wise_importance1.shape
            feature_wise_importance1_abs = np.abs(shap2_values).sum(axis=(0, 1))
            count_all+=1
            #ws_wise_importance_abs = np.abs(shap_values).sum(axis=(2))
            time_wise_input = data2_values.sum(axis=(0))
            #print("input shape",time_wise_input.shape)
        # print(time_wise_input[:, 3])
            time_wise_importance = shap2_values.sum(axis=(0))
        #fig, axs = plt.subplots(1, 1, figsize=(8, 15), sharex=True)
            #index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
            #index=[3]
            for j in range(78):
                if feature_wise_importance1[j]:
                   # print(j, "is", feature_wise_importance1[j])
                    sum[j]+=feature_wise_importance1[j]
                    sum_abs[j]+=feature_wise_importance1_abs[j]
       
for i in range(68):
            shap1=shap_1[i]
            data1=data_1[i]
            data2_values = data1.reshape(3,4,80)
            shap2_values = shap1.reshape(3,4,80)
            feature_wise_importance1 = shap2_values.sum(axis=(0, 1))
            feature_wise_importance1.shape
            feature_wise_importance1_abs = np.abs(shap2_values).sum(axis=(0, 1))
            count_all+=1
            #ws_wise_importance_abs = np.abs(shap_values).sum(axis=(2))
            time_wise_input = data2_values.sum(axis=(0))
            #print("input shape",time_wise_input.shape)
        # print(time_wise_input[:, 3])
            time_wise_importance = shap2_values.sum(axis=(0))
        #fig, axs = plt.subplots(1, 1, figsize=(8, 15), sharex=True)
            #index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
            #index=[3]
            for j in range(78):
                if feature_wise_importance1[j]:
                   # print(j, "is", feature_wise_importance1[j])
                    sum[j]+=feature_wise_importance1[j]
                    sum_abs[j]+=feature_wise_importance1_abs[j]
            
              
k=0
other=0
index=[]
index_abs=[]
sum_failure=[]
sum_abs_failure=[]
print("Total samples:", count_all)
counter=0
for i in range(19):
    print(sum[i]/count_all)
    if abs(sum[i])/count_all>=.005 :
      #print(i, "is", sum[i])
      counter+=1
      index.append(i)
      sum_failure.append(sum[i]/count_all)
      k=k+1
    else:
        other+=sum[i]/count_all
#print("other:", other) 
#index.append(84)
#sum_failure.append(other)   
k=0  
other=0    
for i in range(19):
    if sum_abs[i]/count_all>.01:  
        
        sum_abs_failure.append(sum_abs[i]/count_all)
        index_abs.append(i)
        k=k+1
    else:
        other+=sum_abs[i]/count_all
#sum_abs_failure.append(other) 
#index_abs.append(84) 
print("other_abs:", sum_failure)  
print ("total",sum_abs_failure)
print("index", index)
# Assuming x_coords and y_coords are defined elsewhere in your code
file_1 = open("/home/papry/Exp_Project/MS_thesis-main/column_name.txt", "r")
labels= file_1.read()
file_1.close()
label_list = labels.split(",")
print(label_list)
#feature_index=[2,3,5,6,7,9,36,47,59,68]
#Bar plot with sorted SHAP feautre values
'''
f_index=index
feature_names = [ ]
print(f_index)
for j in (f_index):
    if j==84:
        feature_names.append('Others')
    else:
        feature_names.append(label_list[j-1])

feature_names.append('Others')
shap_values = [0.0079138287888041, 0.5816807891006924,0.2552412025437722, 0.014813232937887475,0.028584113814317802,
               0.009402638816766362,-0.010145174995362253,0.01675079673543499,-0.009134130322183742, 0.010885140441632472,0.06815069868197007]
shap_values_f=[0.0038689432253040325, 0.003625090187148847, 0.00024021859478459063, 0.00036747787896292665,0.0011202926770185102]
'''
# Create bar plot
f_index=index
feature_names = [ ]
print(f_index)
for j in (f_index):
     feature_names.append(label_list[j-1])
shap_values_f=sum_failure
plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(10, 6))
sorted_indices = np.argsort(shap_values_f)[::-1]
sorted_shap_values = np.array(shap_values_f)[sorted_indices]
sorted_feature_names = np.array(feature_names)[sorted_indices]

# Create the horizontal bar plot

shap_values_f=sorted_shap_values 
feature_names=sorted_feature_names
plt.rcParams.update({'font.size': 20, 'lines.linewidth': 2, 'axes.linewidth': 2})
fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.6)))  # Dynamic height for clarity

bars = ax.barh(
    feature_names,
    shap_values_f,
    color=['#d62728' if value > 0 else '#1f77b4' for value in shap_values_f],
    edgecolor='black',
    height=0.6
)

# Set y-tick labels (feature names) to bold
yticklabels = ax.get_yticklabels()
#for label in yticklabels:
    #label.set_fontweight('bold')
ax.set_yticklabels(yticklabels)

# Calculate padding for text to ensure all values are visible
x_min = min(0, min(shap_values_f) - 0.01)
x_max = max(shap_values_f) + 0.01
x_range = x_max - x_min
pad = x_range * 0.04  # 4% of the range as padding

# Add value labels outside the bars, with 3 digits
for bar, value in zip(bars, shap_values_f):
    width = bar.get_width()
    if width > 0:
        xpos = width + pad
        ha = 'left'
        color = 'black'
    else:
        xpos = width - pad
        ha = 'right'
        color = 'black'
    ax.text(
        xpos,
        bar.get_y() + bar.get_height() / 2,
        f'{value:+.3f}',
        va='center',
        ha=ha,
        fontsize=20,
        #fontweight='bold',
        color=color,
        clip_on=False  # Allow text to go outside axes if needed
    )

ax.set_xlabel('Mean of SHAP value', fontsize=24, fontweight='bold')
#for label in ax.get_xticklabels():
   #label.set_fontweight('bold')
ax.set_ylabel('Features', fontsize=24, fontweight='bold')
#ax.set_title('Most Important Features (SHAP Values)', fontsize=18, fontweight='bold', pad=15)
ax.grid(axis='x', linestyle='--', alpha=0.5)
ax.xaxis.set_major_locator(mticker.MaxNLocator(7))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout(pad=2.0)

# Adjust xlim to fit all bars and labels, with extra padding for visibility
ax.set_xlim(x_min - pad, x_max + pad)
plt.savefig('SHAP_gentrap_Urban.pdf', format='pdf', bbox_inches='tight', dpi=600)
#plt.savefig('new_failure_Transformer.pdf', format='pdf', bbox_inches='tight', dpi=600)
plt.show()


