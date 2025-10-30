import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import statistics
sum=np.zeros(107)
sum_abs=np.zeros(107)
ws_sum=np.zeros(4)
ws_f=np.zeros(6)
path0="/home/papry/Exp_Project/MS_thesis-main/logs/20250722_050939/" # time0: 7,8,12,14 are FN cases
column_path="/home/papry/Exp_Project/MS_thesis-main/data/20250610_143541/train_prev_rural_time4_None.csv"
with open(column_path, 'r') as f:
    first_row = f.readline().strip().split(',')
print(len(first_row))
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
data=np.loadtxt(path0+f'tp_new_data.txt')
data = data.reshape(81,4,107)
for i in range(80):
    #if i not in {7,8,12,14}:
        #data=np.loadtxt(path2+f'input{i}_new_data.txt')
        count_all+=1
        shap = np.loadtxt(path0+f'fs{i}_shap_data.txt')
        data[i] = data[i].reshape(4,107)
       # data = data.reshape(3,4,84)
        shap_values = shap.reshape(4,107)
    # print(data.shape)
    # print(shap_values.shape)
    
        relative_change = np.zeros_like(data[i])
        relative_change[1:] = np.abs((data[i][1:] - data[i][:-1]) )/ (np.abs(data[i][:-1]) + 1e-3)  # Slightly larger epsilon
        norm = np.sum(np.abs(relative_change), axis=0, keepdims=True) + 1e-6  # Avoid division by zero
        relative_change_normalized = relative_change / norm +1

# Clip relative changes to avoid very large/small values
        #relative_change = np.clip(relative_change, 1, 2)
# Multiply relative change with SHAP importance
        #relative_importance =relative_change_normalized * shap_values

# Example: sum over time to get a single importance per feature
        #relative_feature_importance = relative_importance.sum(axis=0)

# Now you can use relative_feature_importance as you wish, e.g.:
     #   print("Relative feature importance shape:", relative_feature_importance.shape)
       # time_weights = np.array([1, 1,1,1])  # Adjust as needed; last is most recent
       # feature_wise_importance = (relative_importance * time_weights[:, None]).sum(axis=0)
        feature_wise_importance=shap_values.sum(axis=(0))
        feature_wise_importance_abs = np.abs(shap_values).sum(axis=(0)) #for each ws
        ts_wise_importance = np.abs(shap_values).sum(axis=(1))
       
        if feature_wise_importance[3]>0:
            count+=1
        uavail.append(feature_wise_importance[3])
        bbe.append(feature_wise_importance[5])
       # ws.append(feature_wise_importance[9])
       # temp.append(feature_wise_importance[10])
       
        for j in range(107):
          sum[j]+=feature_wise_importance[j]
          sum_abs[j]+=feature_wise_importance_abs[j]
        for j in range(4):
          ws_sum[j]+=ts_wise_importance[j]
          

print(feature_wise_importance.shape, sum.shape,feature_wise_importance[2], sum[2])              
k=0
other=0
index=[]
index_abs=[]
sum_failure=[]
sum_abs_failure=[]

for i in range(107):

    if (sum[i])/count_all>=.012:
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
others=0    
for i in range(107):
    if sum_abs[i]/count_all>=.05:

        sum_abs_failure.append(sum_abs[i]/count_all)
        index_abs.append(i)
        k=k+1
    else:
        others+=sum_abs[i]/count_all
#sum_abs_failure.append(other) 
#index_abs.append(84) 
print("sum failure:", sum_failure)  
print ("total",sum_abs_failure)
print("index", index)
for i in index:
    print(i, "is", first_row[i])
# Assuming x_coords and y_coords are defined elsewhere in your code

file_1 = open("/home/papry/Exp_Project/MS_thesis-main/column_name.txt", "r")
labels= file_1.read()
file_1.close()
label_list = labels.split(",")
label_list=first_row[0:84]
#feature_index=[2,3,5,6,7,9,36,47,59,68]
#Bar plot with sorted SHAP feautre values


# Prepare feature names and SHAP values, ensuring 'Others' is last
import matplotlib.ticker as mticker

# Prepare feature names and SHAP values, ensuring 'Others' is first and others are sorted by weight
f_index = index_abs
feature_names = []
shap_values_f = []

# Find the index of 'Others' (assumed to be 84)
others_idx = None
for idx, j in enumerate(f_index):
    if j ==107:
        others_idx = idx
        break

# Add 'Others' first if present
if others_idx is not None:
    feature_names.append('Others')
    shap_values_f.append(sum_abs_failure[others_idx])

# Collect the rest of the features and their values
other_features = []
other_values = []
for idx, j in enumerate(f_index):
    if j == 107:
        continue
    name = label_list[j]
    if len(name) > 15:
        name = name[:4] + "..." + name[-4:]
    other_features.append(name)
    other_values.append(sum_abs_failure[idx])

# Sort features (except 'Others') by their SHAP value (weight), descending
sorted_indices = sorted(range(len(other_values)), key=lambda i: abs(other_values[i]), reverse=True)
sorted_features = [other_features[i] for i in sorted_indices]
sorted_values = [other_values[i] for i in sorted_indices]

# Combine 'Others' first, then sorted features
if 'unavail_second' in label_list:
    unavail_idx = label_list.index('unavail_second')
    if unavail_idx + 1 in f_index and 'unavail_second' not in sorted_features:
        # Find its value in sum_failure
        idx_in_findex = f_index.index(unavail_idx + 1)
        unavail_value = sum_failure[idx_in_findex]
        # Insert at the top after 'Others'
        feature_names.insert(1, 'unavail_second')
        shap_values_f.insert(1, unavail_value)

# Combine 'Others' first, then sorted features (skip unavail_second if already inserted)
for feat, val in zip(sorted_features, sorted_values):
    if feat == 'unavail_second' and 'unavail_second' in feature_names:
        continue
    feature_names.append(feat)
    shap_values_f.append(val)


plt.rcParams.update({'font.size': 16, 'lines.linewidth': 2, 'axes.linewidth': 2})
fig, ax = plt.subplots(figsize=(13, max(6, len(feature_names) * 0.6)))  # Dynamic height for clarity

bars = ax.barh(
    feature_names,
    shap_values_f,
    color=['#d62728' if value > 0 else '#1f77b4' for value in shap_values_f],
    edgecolor='black',
    height=0.6
)

# Set y-tick labels (feature names) to bold
yticklabels = ax.get_yticklabels()
for label in yticklabels:
    label.set_fontweight('bold')
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
        fontsize=13,
        fontweight='bold',
        color=color,
        clip_on=False  # Allow text to go outside axes if needed
    )

ax.set_xlabel('Mean of SHAP value', fontsize=18, fontweight='bold')
for label in ax.get_xticklabels():
    label.set_fontweight('bold')
ax.set_ylabel('Features', fontsize=18, fontweight='bold')
ax.set_title('Most Important Features (SHAP Values)', fontsize=18, fontweight='bold', pad=15)
ax.grid(axis='x', linestyle='--', alpha=0.5)
ax.xaxis.set_major_locator(mticker.MaxNLocator(7))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout(pad=2.0)

# Adjust xlim to fit all bars and labels, with extra padding for visibility
ax.set_xlim(x_min - pad, x_max + pad)
plt.savefig('figure_abs_lstm.pdf', format='pdf', bbox_inches='tight', dpi=600)
#plt.savefig('new_failure_Transformer.pdf', format='pdf', bbox_inches='tight', dpi=600)
plt.show()