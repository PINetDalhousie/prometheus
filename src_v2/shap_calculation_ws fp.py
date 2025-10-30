import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import statistics
sum=np.zeros(84)
sum_abs=np.zeros(84)
ws_sum=np.zeros(3)
ws_f=np.zeros(6)
path0="/users/grad/papry/MS_thesis-main/logs/20240831_070447/" # time0: 7,8,12,14 are FN cases
path4="/users/grad/papry/MS_thesis-main/logs/20240828_213216/" #time4: 18, 19, 42, 46, 50, 51, 52, 90
path1="/users/grad/papry/MS_thesis-main/logs/20241001_010049/" # time 1: 15,22,34
path2="/users/grad/papry/MS_thesis-main/logs/20241003_105828/" # for test_time2 FN cases: 53,54,156,157,158,175,176
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
for i in range(25):
    if i not in {7,8,12,14}:
        data0 = np.loadtxt(path0+f'input{i}_new_data.txt')
        
        data0 = data0.reshape(3,4,84)
        for j in range(3):
            ws_data.append(data0[j][0][9])
            
all_data2=np.loadtxt(path2+f'ws_new_data.txt')
print(all_data2.shape)
all_data2 = all_data2.reshape(177,3,4,84)
for i in range(52):
    if i not in {18, 19, 42, 46, 50, 51, 52}:
        for j in range(3):
         ws_data.append(all_data2[i][j][0][9])
        #print(all_data[i][j][1][9])
print(len(ws_data))
all_data4=np.loadtxt(path4+f'ws_new_data.txt')
print("4", all_data4.shape)
all_data4 = all_data4.reshape(91,3,4,84)
for i in range(52):
    for j in range(3):
        ws_data.append(all_data4[i][j][0][9])
        #print(all_data[i][j][1][9])
#print(len(ws_data))

        #print(data0[j][1][9])
print("Total ws_data len",len(ws_data))

for i in range(25):
    if i not in {7,8,12,14}:
        #data=np.loadtxt(path2+f'input{i}_new_data.txt')
        count_all+=1
        shap = np.loadtxt(path0+f'shap{i}_new_data.txt')
    
       # data = data.reshape(3,4,84)
        shap_values = shap.reshape(3,4,84)
    # print(data.shape)
    # print(shap_values.shape)
        feature_wise_importance = shap_values.sum(axis=(0, 1))
        feature_wise_importance_abs = np.abs(shap_values).sum(axis=(1)) #for each ws
        feature_wise_importance_abs = shap_values.sum(axis=(1)) #for each ws
        ws_wise_importance = np.abs(shap_values).sum(axis=(1, 2))
       
        if feature_wise_importance[3]>0:
            count+=1
        uavail.append(feature_wise_importance[3])
        bbe.append(feature_wise_importance[5])
        ws.append(feature_wise_importance[9])
        temp.append(feature_wise_importance[10])
        for i in range(3):
            ws_all_f=0
            
            for j in range(9): #for ws_feature: 10-16
                ws_all_f+=feature_wise_importance_abs[i][j] #add j+10 for ws
            for j in range(35): #for ws_feature: 52-83 , range 32
                ws_all_f+=feature_wise_importance_abs[i][j+17] #add j+52 for ws
            '''  
            for j in range(6): #for ws_feature: 10-16
                ws_all_f+=feature_wise_importance_abs[i][j+10] #add j+10 for ws
            for j in range(32): #for ws_feature: 52-83 , range 32
                ws_all_f+=feature_wise_importance_abs[i][j+52] #add j+52 for ws
                '''
            ws_feature.append(ws_all_f)  
        for j in range(84):
          sum[j]+=feature_wise_importance[j]
          #sum_abs[j]+=feature_wise_importance_abs[j]
        for j in range(3):
          ws_sum[j]+=ws_wise_importance[j]
          
#For time 1
'''
for i in range(51):
    if i not in {15, 22, 34}:
        #data=np.loadtxt(path2+f'input{i}_new_data.txt')
        count_all+=1
        shap = np.loadtxt(path1+f'shap{i}_new_data.txt')
    
       # data = data.reshape(3,4,84)
        shap_values = shap.reshape(3,4,84)
    # print(data.shape)
    # print(shap_values.shape)
        feature_wise_importance = shap_values.sum(axis=(0, 1))
        feature_wise_importance_abs = np.abs(shap_values).sum(axis=( 1))
        ws_wise_importance = np.abs(shap_values).sum(axis=(1, 2))
        if feature_wise_importance[3]>0:
            count+=1
        uavail.append(feature_wise_importance[3])
        bbe.append(feature_wise_importance[5])
        ws.append(feature_wise_importance[9])
        temp.append(feature_wise_importance[10])
        for i in range(3):
            ws_all_f=0
            for j in range(6):
             ws_all_f+=feature_wise_importance_abs[i][j+10]
            for j in range(32):
             ws_all_f+=feature_wise_importance_abs[i][j+52]
            ws_feature.append(ws_all_f)  
        for j in range(84):
          sum[j]+=feature_wise_importance[j]
          #sum_abs[j]+=feature_wise_importance_abs[j]
        for j in range(3):
          ws_sum[j]+=ws_wise_importance[j]
        '''  
  #For time 2
#count=0      
for i in range(52):
        #data=np.loadtxt(path2+f'input{i}_new_data.txt')
    if i not in {53,54, 156,157,158,175,176} : 
            shap = np.loadtxt(path2+f'shap{i}_new_data.txt')
       # if i<25:
         #   shap2= np.loadtxt(path2+f'shap{i+51}_nonfailure_data.txt')
         #   size=i+51
    # file_1 = open("/content/sample_data/column_name.txt", "r")

        #labels= file_1.read()
    # file_1.close()
    # label_list = labels.split(",")
       
        #for fold 4    53,54,156,157,158,175,176
        #if i not in {15, 22, 34} : #for fold 1
            count_all+=1
           # print("i", i)
       # data = data.reshape(3,4,84)
            shap_values = shap.reshape(3,4,84)
            
        # print(data.shape)
        # print(shap_values.shape)
            feature_wise_importance = shap_values.sum(axis=(0, 1))
            feature_wise_importance_abs = np.abs(shap_values).sum(axis=( 1))
            feature_wise_importance_abs = shap_values.sum(axis=(1)) #for each ws
            ws_wise_ws_feature=np.abs(shap_values).sum(axis=(1, 2))
            ws_wise_importance = np.abs(shap_values).sum(axis=(1, 2))
            ws_all_f=0
           # print(ws_wise_ws_feature.shape)
            for j in range(3):
               # for k in range(84):
                ws_all_f+=ws_wise_ws_feature[j]
                ws_shap.append(ws_all_f)
            if feature_wise_importance[3]>0:
                count+=1
            uavail.append(feature_wise_importance[3])
            bbe.append(feature_wise_importance[5])
            ws.append(feature_wise_importance[9])
            temp.append(feature_wise_importance[10])
            for i in range(3):
                ws_all_f=0
                
                for j in range(9): #for ws_feature: 10-16
                    ws_all_f+=feature_wise_importance_abs[i][j] #add j+10 for ws
                for j in range(35): #for ws_feature: 52-83 , range 32
                    ws_all_f+=feature_wise_importance_abs[i][j+17] #add j+52 for ws
                '''
                for j in range(6): #for ws_feature: 10-16
                    ws_all_f+=feature_wise_importance_abs[i][j+10] #add j+10 for ws
                for j in range(32): #for ws_feature: 52-83 , range 32
                    ws_all_f+=feature_wise_importance_abs[i][j+52] #add j+52 for ws
                    '''
                ws_feature.append(ws_all_f)  
            for j in range(84):
                sum[j]+=feature_wise_importance[j]
               # sum_abs[j]+=feature_wise_importance_abs[j]
            for j in range(3):
                ws_sum[j]+=ws_wise_importance[j]
               
#for time 4 
#print(len(ws_shap))
#count=0      
for i in range(52):
        #data=np.loadtxt(path2+f'input{i}_new_data.txt')
    if i not in {18, 19, 42, 46, 50, 51, 52, 90} : 
           # shap = np.loadtxt(path4+f'fp{i}_shap_data.txt')
            shap = np.loadtxt(path4+f'shap{i}_new_data.txt')
       # if i<25:
          #  shap2= np.loadtxt(path4+f'shap{i+51}_nonfailure_data.txt')
           # size=i+51
    # file_1 = open("/content/sample_data/column_name.txt", "r")

        #labels= file_1.read()
    # file_1.close()
    # label_list = labels.split(",")
       
        #for fold 4
        #if i not in {15, 22, 34} : #for fold 1
            count_all+=1
           # print("i", i)
       # data = data.reshape(3,4,84)
            shap_values = shap.reshape(3,4,84)
            
        # print(data.shape)
        # print(shap_values.shape)
            feature_wise_importance = shap_values.sum(axis=(0, 1))
            feature_wise_importance_abs = np.abs(shap_values).sum(axis=(1))
            feature_wise_importance_abs = shap_values.sum(axis=(1)) #for each ws
            ws_wise_importance = np.abs(shap_values).sum(axis=(1, 2))
            if feature_wise_importance[3]>0:
                count+=1
            uavail.append(feature_wise_importance[3])
            bbe.append(feature_wise_importance[5])
            ws.append(feature_wise_importance[9])
            temp.append(feature_wise_importance[10])
            for i in range(3):
                ws_all_f=0
                
                for j in range(9): #for ws_feature: 10-16
                    ws_all_f+=feature_wise_importance_abs[i][j] #add j+10 for ws
                for j in range(35): #for ws_feature: 52-83 , range 32
                    ws_all_f+=feature_wise_importance_abs[i][j+17] #add j+52 for ws
                '''
                for j in range(6): #for ws_feature: 10-16
                     ws_all_f+=feature_wise_importance_abs[i][j+10] #add j+10 for ws
                for j in range(32): #for ws_feature: 52-83 , range 32
                     ws_all_f+=feature_wise_importance_abs[i][j+52] #add j+52 for ws
                        '''
                ws_feature.append(ws_all_f) 
            for j in range(84):
                sum[j]+=feature_wise_importance[j]
                #sum_abs[j]+=feature_wise_importance_abs[j]
                
            for j in range(3):
                ws_sum[j]+=ws_wise_importance[j]
                '''
        if i<25 and size not in {51,52,90}:
            shap_values = shap2.reshape(3,4,84)
            feature_wise_importance = shap_values.sum(axis=(0, 1))
            feature_wise_importance_abs = np.abs(shap_values).sum(axis=(0, 1))
           # if feature_wise_importance[3]>0:
               # count+=1
            uavail.append(feature_wise_importance[3])
            bbe.append(feature_wise_importance[5])
            ws.append(feature_wise_importance[9])
            temp.append(feature_wise_importance[10])
            ws_all_f=0
            for j in range(6):
                ws_all_f+=feature_wise_importance_abs[j+10]
            for j in range(32):
                ws_all_f+=feature_wise_importance_abs[j+52]
            ws_feature.append(ws_all_f)  
            for j in range(84):
                sum[j]+=feature_wise_importance[j]
                sum_abs[j]+=feature_wise_importance_abs[j]
                '''
               
k=0
other=0
index=[]
index_abs=[]
sum_failure=[]
sum_abs_failure=[]
'''
for i in range(84):

    if abs(sum[i])/count_all>.025:
      #print(i, "is", sum[i])
      index.append(i)
      sum_failure.append(sum[i]/count_all)
      k=k+1
    else:
        other+=sum[i]/count_all
#print("other:", other) 
index.append(84)
sum_failure.append(other)   
k=0  
other=0    
for i in range(84):
    if sum_abs[i]/count_all>.04:  
        
        sum_abs_failure.append(sum_abs[i]/count_all)
        index_abs.append(i)
        k=k+1
    else:
        other+=sum_abs[i]/count_all
sum_abs_failure.append(other) 
index_abs.append(84) 
print("other_abs:", sum_failure)  
print ("total",sum_abs_failure)
print("count", count, count_all)
'''
#print(ws_shap)
print("ws_len ", len(ws_feature)) 


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

# Create bar plot
shap_values_f=sum_failure
plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(10, 6))
sorted_indices = np.argsort(shap_values_f)
sorted_shap_values = np.array(shap_values_f)[sorted_indices]
sorted_feature_names = np.array(feature_names)[sorted_indices]

# Create the horizontal bar plot

bars = plt.barh(sorted_feature_names, sorted_shap_values, color=['red' if value > 0 else 'blue' for value in sorted_shap_values])

# Add values outside each bar
for bar, value in zip(bars, sorted_shap_values):
    plt.text(value +0.0005 if value > 0 else value - 0.0005,  # Adjust position to place text outside the bar
             bar.get_y() + bar.get_height() / 2,
             '{:+0.4f}'.format(value), va='center', ha='left' if value > 0 else 'right')
    '''

# Adjust x-axis limits to ensure all values fit within the plot boundaries and do not exceed 0.01


#plt.xlim(-0.2, 0.5)

'''
# Adjust x-axis limits to ensure all values fit within the plot boundaries and do not exceed 0.01
#plt.xlim(0, 1)


ws_number=['WS1','WS2','WS3']
ws_value=[]
for i in range(3):
    ws_value.append(ws_sum[i]/count_all)
uavail = np.random.rand(100)  # Example data
cmap = plt.get_cmap('Blues')

# Generate colors from the colormap
colors = [cmap(i / len(ws_number)) for i in range(len(ws_number))]

mean = np.mean(uavail)
median = np.median(uavail)
std_dev = np.std(uavail)

# Create the bar plot for the data points
plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(12, 8))

plt.bar(range(len(uavail)), uavail, color='lightblue', edgecolor='black')

# Add lines for mean, median, and standard deviation
plt.axhline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
plt.axhline(median, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}')
plt.axhline(mean + std_dev, color='blue', linestyle='dashed', linewidth=2, label=f'Std Dev: {std_dev:.2f}')
plt.axhline(mean - std_dev, color='blue', linestyle='dashed', linewidth=2)

# Add labels and title
plt.xlabel('Data Points')
plt.ylabel('Value')
plt.title('Bar Plot of Data Points with Mean, Median, and Standard Deviation')

# Add legend
plt.legend()

# Adjust layout to ensure all texts are clearly visible
bars = plt.bar(ws_number, ws_value, color=colors, edgecolor='black')

# Add text annotations
for bar, value in zip(bars, ws_value):
    plt.text(bar.get_x() + bar.get_width() / 2, value/203, f'{value:.2f}', ha='center', va='bottom')

# Add labels and title
plt.xlabel('WS Number')
plt.ylabel(' ABS Mean of SHAP value')
plt.title('Bar Plot of WS Values')
'''

plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(12, 8))

plt.scatter(ws_data, ws_feature, color='blue', edgecolor='black', alpha=0.6)

# Add labels and title
plt.xlabel('WS Distance')
plt.ylabel('Sum of RL SHAP Feature values')
plt.title('Scatter Plot of WS Distance vs RL SHAP Feature values')

# Show the plot
#plt.show()
# Adjust layout to ensure all texts are clearly visible
plt.tight_layout()

# Save the plot to a PDF file
plt.savefig('ws_distance_vs_rl_shap_not_abs_scatterplot.pdf', format='pdf')

# Show the plot
plt.show()



# Show the plot
 

