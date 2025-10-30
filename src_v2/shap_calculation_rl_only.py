import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import statistics
import matplotlib.ticker as mticker
sum=np.zeros(84)
sum_abs=np.zeros(84)
ws_sum=np.zeros(3)
ws_f=np.zeros(6)
path0="/home/papry/Exp_Project/MS_thesis-main/logs/20251020_005159/" # time0: 7,8,12,14 are FN cases
uavail=[]
bbe=[]
count=0
count_all=0
ws_feature=[]
ws_data=[]
temp=[]
ws_shap=[]
ws=[]
#For time 0
fs=78
'''
#print(len(ws_data))
for i in range(5):
    if i not in {7,8,12,14}:
        data0 = np.loadtxt(path0+f'input{i}_new_data.txt')
        
        data0 = data0.reshape(3,4,84)
        for j in range(3):
            ws_data.append(data0[j][0][9])
        #print(data0[j][1][9])
print("Total ws_data len",len(ws_data))
'''
data_4= np.loadtxt(f'/home/papry/Exp_Project/MS_thesis-main/logs/20251019_030830/input_new_data.txt')#33
shap_4= np.loadtxt(f'/home/papry/Exp_Project/MS_thesis-main/logs/20251019_030830/shap_new_data.txt')
data_2= np.loadtxt(f'/home/papry/Exp_Project/MS_thesis-main/logs/20251021_012848/input_all_new_data.txt')#33
shap_2= np.loadtxt(f'/home/papry/Exp_Project/MS_thesis-main/logs/20251021_012848/shap_all_new_data.txt')
fs4=78
fs2=79
data_4=data_4.reshape(-1,3,4,fs4)
shap_4=shap_4.reshape(-1,3,4,fs4)
data_2=data_2.reshape(-1,3,4,fs2)
shap_2=shap_2.reshape(-1,3,4,fs2)
print(data_4.shape)
for i in range(40):
        count_all+=1
        data4=data_4[i]
        shap4=shap_4[i]
        
        #print(label_list)


        data_values = data4.reshape(3,4,fs4)
        shap_values = shap4.reshape(3,4,fs4)
       # data = data.reshape(3,4,84)
        #shap_values = shap.reshape(3,4,84)
    # print(data.shape)
    # print(shap_values.shape)
        feature_wise_importance = shap_values.sum(axis=(0, 1))
        feature_wise_importance_abs = np.abs(shap_values).sum(axis=(0,1)) #for each ws
        ws_wise_importance = np.abs(shap_values).sum(axis=(1, 2))
       
        if feature_wise_importance[3]>0:
            count+=1
        uavail.append(feature_wise_importance[3])
        bbe.append(feature_wise_importance[5])
       # ws.append(feature_wise_importance[9])
       # temp.append(feature_wise_importance[10])
       
        for j in range(8):
          sum[j]+=feature_wise_importance[j]
          sum_abs[j]+=feature_wise_importance_abs[j]
        for j in range(3):
          ws_sum[j]+=ws_wise_importance[j]
          
for i in range(87):
        count_all+=1
        data2=data_2[i]
        shap2=shap_2[i]

        #print(label_list)


        data_values = data2.reshape(3,4,fs2)
        shap_values = shap2.reshape(3,4,fs2)
       # data = data.reshape(3,4,84)
        #shap_values = shap.reshape(3,4,84)
    # print(data.shape)
    # print(shap_values.shape)
        feature_wise_importance = shap_values.sum(axis=(0, 1))
        feature_wise_importance_abs = np.abs(shap_values).sum(axis=(0,1)) #for each ws
        ws_wise_importance = np.abs(shap_values).sum(axis=(1, 2))
       
        if feature_wise_importance[3]>0:
            count+=1
        uavail.append(feature_wise_importance[3])
        bbe.append(feature_wise_importance[5])
       # ws.append(feature_wise_importance[9])
       # temp.append(feature_wise_importance[10])
       
        for j in range(8):
          sum[j]+=feature_wise_importance[j]
          sum_abs[j]+=feature_wise_importance_abs[j]
        for j in range(3):
          ws_sum[j]+=ws_wise_importance[j]
          
               
k=0
other=0
index=[]
index_abs=[]
sum_failure=[]
sum_abs_failure=[]
print("Total samples:", count_all)
for i in range(9):

    if abs(sum[i])/count_all>=.005:
      #print(i, "is", sum[i])
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
for i in range(9):
    if sum_abs[i]/count_all>.001:  
        
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
plt.savefig('SHAP_RL_Urban.pdf', format='pdf', bbox_inches='tight', dpi=600)
#plt.savefig('new_failure_Transformer.pdf', format='pdf', bbox_inches='tight', dpi=600)
plt.show()

