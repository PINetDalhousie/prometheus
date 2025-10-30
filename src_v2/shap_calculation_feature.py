import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import statistics
import seaborn as sns
sum=np.zeros(84)
sum_abs=np.zeros(84)
ws_sum=np.zeros(3)
ws_f=np.zeros(6)
uavail=[]
bbe=[]
count=0
count_all=0
ws_feature=[]
cor_data=[]
temp=[]
ws_shap=[]
ws=[]
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
file_1 = open("/home/papry/Exp_Project/MS_thesis-main/column_name.txt", "r")
labels= file_1.read()
file_1.close()
label_list = labels.split(",")
print(label_list)
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
    
 
    colors = ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8"]
    my_map = mcolors.ListedColormap(colors, name="my_map")
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
    #print("size", size)
    for k in range(size):
            feature_wise_importance1 = shap_values_list[k].sum(axis=(0, 1))
            #print("fis",feature_wise_importance1.shape)
            feature_wise_importance1_abs = np.abs(shap_values_list[k]).sum(axis=(0, 1))
            k=0
            #ws_wise_importance_abs = np.abs(shap_values).sum(axis=(2))
            time_wise_input = data_values_list[k].sum(axis=(0))
            #print("input shape",time_wise_input.shape)
        # print(time_wise_input[:, 3])
            time_wise_importance = shap_values_list[k].sum(axis=(0))
        #fig, axs = plt.subplots(1, 1, figsize=(8, 15), sharex=True)
         #  index=[2,6,7,3,5,11,13,14]
            index=78
            
            m=0
            if np.isnan(feature_wise_importance1).any():
                print("NaN values found")

            else:
                input_feature=data_values_list[k].reshape(-1,data_values_list[k].shape[-1])
                print("input feature", input_feature.shape)
                cor_data.append(input_feature[:, 1:16])
                for idx in range(index):

                   # print("feature", feature_wise_importance1[idx])
                    sum[idx]+=feature_wise_importance1[idx]
                    #sum2=sum2+time_wise_importance[j, idx]
                count_all+=1
  

#For time 0
print("total samples:", count_all)
#print("Feature wise importance", sum)
print(" All Feature input", len(cor_data))
cor_data_array = np.array(cor_data)  # This will try to convert to shape (178, 20) if possible
print(cor_data_array.shape)
cor_data_array = cor_data_array.reshape(-1, cor_data_array.shape[-1])  # shape: (n_samples, n_features)
print(cor_data_array.shape)
cor_matrix = np.corrcoef(cor_data_array, rowvar=False)  # shape: (n_features, n_features)
#print("correlation matrix shape:", cor_matrix.shape)
#print("correlation matrix:", cor_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(cor_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Among Selected Features")
plt.savefig("feature_correlation_matrix.png", dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()
other=0
index=[]
index_abs=[]
sum_failure=[]
sum_abs_failure=[]
k=0
for i in range(78):

    if abs(sum[i])/count_all>.01:
      #print(i, "is", sum[i])
      index.append(i)
      sum_failure.append(sum[i]/count_all)
      k=k+1
    else:
        other+=sum[i]/count_all
#print("other:", other) 

other=0    
'''
for i in range(84):
    if sum_abs[i]/count_all>.04:  
        
        sum_abs_failure.append(sum_abs[i]/count_all)
        index_abs.append(i)
        k=k+1
    else:
        other+=sum_abs[i]/count_all
sum_abs_failure.append(other) 
index_abs.append(84) 
'''
print("other_abs:", sum_failure)  
print ("total",index)
print("count", count, count_all)
'''
#print(ws_shap)
print("ws_len ", len(ws_feature), len(ws_sum)) 


# Sample data points (replace with your actual data)
#import matplotlib.pyplot as plt

# Assuming x_coords and y_coords are defined elsewhere in your code
file_1 = open("/users/grad/papry/column_name.txt", "r")
labels= file_1.read()
file_1.close()
label_list = labels.split(",")

#feature_index=[2,3,5,6,7,9,36,47,59,68]
#Bar plot with sorted SHAP feautre values
'''
label_list=label_list[0:70]
#feature_index=[2,3,5,6,7,9,36,47,59,68]
#Bar plot with sorted SHAP feautre values


# Prepare feature names and SHAP values, ensuring 'Others' is last
import matplotlib.ticker as mticker

# Prepare feature names and SHAP values, ensuring 'Others' is first and others are sorted by weight
f_index = index
feature_names = []
shap_values_f = []
sum_failure= sum_failure
# Find the index of 'Others' (assumed to be 84)
others_idx = None


# Add 'Others' first if present


# Collect the rest of the features and their values
other_features = []
other_values = []
for idx, j in enumerate(f_index):
    
    name = label_list[j-1]
    if len(name) > 15:
        name = name[:4] + "..." + name[-4:]
    other_features.append(name)
    other_values.append(sum_failure[idx])

# Sort features (except 'Others') by their SHAP value (weight), descending
sorted_indices = sorted(range(len(other_values)), key=lambda i: abs(other_values[i]), reverse=True)
sorted_features = [other_features[i] for i in sorted_indices]
sorted_values = [other_values[i] for i in sorted_indices]

# Combine 'Others' first, then sorted features


# Combine 'Others' first, then sorted features (skip unavail_second if already inserted)
for feat, val in zip(sorted_features, sorted_values):
    if feat == 'unavail_second' and 'unavail_second' in feature_names:
        continue
    feature_names.append(feat)
    shap_values_f.append(val)


plt.rcParams.update({'font.size': 16, 'lines.linewidth': 2, 'axes.linewidth': 2})
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
        fontsize=16,
        fontweight='bold',
        color=color,
        clip_on=False  # Allow text to go outside axes if needed
    )

ax.set_xlabel('Mean of SHAP value', fontsize=18, fontweight='bold')
for label in ax.get_xticklabels():
    label.set_fontweight('bold')
ax.set_ylabel('Features', fontsize=18, fontweight='bold')
#ax.set_title('Most Important Features (SHAP Values)', fontsize=18, fontweight='bold', pad=15)
ax.grid(axis='x', linestyle='--', alpha=0.5)
ax.xaxis.set_major_locator(mticker.MaxNLocator(7))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout(pad=2.0)

# Adjust xlim to fit all bars and labels, with extra padding for visibility
ax.set_xlim(x_min - pad, x_max + pad)
plt.savefig('figure_transformer_urban.pdf', format='pdf', bbox_inches='tight', dpi=600)
#plt.savefig('new_failure_Transformer.pdf', format='pdf', bbox_inches='tight', dpi=600)
plt.show()
