
# %%
import os
from matplotlib import ticker
import sklearn

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import seaborn as sns

shap=np.loadtxt(f'/home/papry/Exp_Project/MS_thesis-main/logs/20251009_005125/shap5_new_data.txt')

data = np.loadtxt(f'/home/papry/Exp_Project/MS_thesis-main/logs/20251009_005125/input5_new_data.txt')


file_1 = open("/home/papry/Exp_Project/MS_thesis-main/column_name.txt", "r")
labels= file_1.read()
file_1.close()
label_list = labels.split(",")
print(label_list)

colors = ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8"]
my_map = ListedColormap(colors, name="my_map")
data = data.reshape(3,4,79)
shap_values = shap.reshape(3,4,79)
print(data.shape)
#print(label_list[8], data[0,0,0,9], data[0,1,0,9], data[0,2,0,9])
print(shap_values.shape)
#mpl.colormaps.register(cmap=my_map)
norm = mcolors.CenteredNorm()
norm(shap_values)
cmap = plt.get_cmap('coolwarm')
#fig, axs = plt.subplots(15, 4, figsize=(10, 15), squeeze= False, sharey= True, width_ratios=[1, 1, 1, 0.5])

fig, axs = plt.subplots(3,2, figsize=(10,8))
#axs[0, 0].plot(shap_values[0, :, 0], label='shap_values')

feature_wise_importance = shap_values.sum(axis=(0, 1))
feature_wise_importance.shape
feature_wise_importance_abs = np.abs(shap_values).sum(axis=(0, 1))
k=0
ws_wise_importance_abs = np.abs(shap_values).sum(axis=(2))
time_wise_input = data.sum(axis=(0))
print("input shape",time_wise_input.shape)
print(time_wise_input[:, 3])
time_wise_importance = shap_values.sum(axis=(0))
#fig, axs = plt.subplots(1, 1, figsize=(8, 15), sharex=True)
for idx in range(len(label_list)):
    name = label_list[idx]
    if len(name) > 10:
        name = name[:4] + "..." + name[-4:]
        if idx in {2}:
            name="uva"
        label_list[idx] = name


k=0
x_list=["RL Time series features", "WS Time series features", "Static features"]
for i in range(78):
    if i==3 or i==10 or i==17:
        ik=i
        list=[]
        if i==3:
            list=[3,5]
        if i==10:
            list=[13,14]
        if i==17:
            list=[17,34]
        j=0
        for n in (list):      
            my_xticks = ("t-4","t-3","t-2","t-1")
            x = np.arange(len(my_xticks))
            axs[k,j].set_xticks(x, my_xticks, fontsize=22)
            axs[k,j].plot(
                time_wise_input[:, n],
                #label='shap_values',
                alpha=1,  # fully opaque
                linewidth=1.5,  # make line bolder
            color='#1a1a1a'  # dark gray/black for strong contrast
            )
        # Example scatter points (optional, comment out if not needed)
            line_colors = cmap(norm(time_wise_importance[:, n]))
            axs[k,j].scatter(
                np.arange(0, 4),
                time_wise_input[:, n],
                color=line_colors,
                s=120,  # large circles
                edgecolor='#1a1a1a',
                linewidth=1.5
            )
            axs[k,j].set_ylabel(label_list[n-1], fontsize=22, fontweight='bold', labelpad=10)
           # axs[k,j].legend(loc='upper right', fontsize=10)
            #axs[k,j].legend(loc='upper right', fontsize=10)
            # Add vertical space inside plot
            ymin, ymax = axs[k,j].get_ylim()
            yrange = ymax - ymin
            axs[k,j].set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
            if i==3:
                yticks = np.linspace(0, ymax, num=3)
            else:
                yticks = np.linspace(ymin, ymax, num=3)
            axs[k,j].set_yticks(yticks)
            axs[k,j].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
            
            axs[k,j].tick_params(axis='y', labelsize=22)
           # axs[k,j].set_xlabel("Time steps", fontsize=22, fontweight='bold', labelpad=15, loc='right')
            j=j+1
            
        
        #for k in range(3):
       # for ax, title in zip(axs.flat, titles):
        #axs[k,0].set_title(x_list[k], loc='center')

# Adjust layout to prevent overlap

        
        k=k+1
        
#axs[k,1].set_xlabel('Time Step', fontsize=13, fontweight='bold', labelpad=20)
'''
axs[0].set_title("RL Feature", fontsize=16, pad=12)
axs[1].set_title("WS Feature", fontsize=16, pad=12)
# or for below:
axs[0].text(0.5, -0.18, "RL Feature", ha='center', va='center', transform=axs[0].transAxes, fontsize=14)
axs[1].text(0.5, -0.18, "WS Feature", ha='center', va='center', transform=axs[1].transAxes, fontsize=14)
'''
plt.tight_layout()
plt.subplots_adjust(left=0.22)  # Increase left margin for y-label and scatter points
plt.savefig('input20_boxplots_new2.pdf', bbox_inches='tight', pad_inches=0.3)
plt.show()