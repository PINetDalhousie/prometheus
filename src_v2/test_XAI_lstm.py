
# %%

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from data_preprocess_v1 import DataPreprocess
from metrics_manager import MetricsManager,TestMetricsManager
import argparse
from to_tfdata import ToTfData
import datetime
from utils import plot_distribution
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import sys
import time
from metrics_manager import OptimalThresholdCalculator
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl
from alibi.explainers import IntegratedGradients
#import numpy as np
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15000)])
  except RuntimeError as e:
    print(e)
 
def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--autoencoder', type=bool, default=False,
                        help='Use autoencoder approach')
    parser.add_argument('--drop_cat_features', type=bool, default=False,
                        help='Drop categorical features')
    parser.add_argument('--calculate_threshold', type=bool, default=False,
                        help='whether to calculate threshold for autoencoder')
    parser.add_argument('--positional_encoding', type=bool, default=False,
        help='Use positional encoding in transformer model')
    parser.add_argument('--fix_miss_cat', type=str, default=None,
                        help='Zero pad missing categorical feature values when compared with the maximum possible categorical class values. Set value to deployment type: rural or urban')
    parser.add_argument(
        '--threshold',
        type=float,
        default=3.0413200855255127,
        help='Threshold for autoencoder'
        )
    parser.add_argument(
        '--test_csv',
        type=str,
        default="/home/papry/Exp_Project/MS_thesis-main/data/20250719_180144/test_prev_urban_time2_None.csv",
        help='Path to test csv file'
        )
    parser.add_argument(
        '--model_path',
        type=str,
        default="/home/papry/Exp_Project/MS_thesis-main/logs/20251027_171145/ckpt",
        help='Path to trained model'
        )  
    parser.add_argument(
        '--log_path',
        type=str,
        default="/home/papry/Exp_Project/MS_thesis-main/logs/20251027_171145/",
        help='Path to log directory')
    parser.add_argument(
        '--approach',
        type=str,
        default="prev",
        help=f"String identifying which approach to use. 'prev' indicates"
            f"previous approach with derived features.")
    parser.add_argument('--num_weather_stations', type=int, default=3,
                        help='Closest number of weather stations to consider')
    args = parser.parse_args()
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
   # with open(args.log_path+f'{current_time}_test_args.txt', 'w') as file:
   #     for arg, value in vars(args).items():
   #         file.write(f"{arg}: {value}\n")
    
    # Preprocess data
    data_preprocess = DataPreprocess(args.approach)
    data_preprocess.read_test_csv(args.test_csv)
    if args.drop_cat_features:
        data_preprocess.drop_test_cat_features()
    data_preprocess.seperate_test_kpis_labels()
    data_preprocess.min_max_scale_test_data(args.log_path)
   # data_preprocess.normalize_test_data(args.log_path)
    data_preprocess.one_hot_encode_test_data(args.log_path,ignore_cat=False)
    #if args.fix_miss_cat:
       # data_preprocess.pad_missing_catgorical_columns_test(args.fix_miss_cat)
    test_kpis, test_labels = data_preprocess.get_test_kpis_labels()


    # Prepare data with tf.data.Dataset
    to_tfdata = ToTfData(
        args.batch_size,
        args.approach,
        args.autoencoder,
        args.positional_encoding)
    test_ds = to_tfdata.test_to_tfdata(test_kpis,test_labels)
    print("Size of test_ds", test_kpis.shape)
    # load the model
    model = tf.keras.models.load_model(args.model_path,compile=False)
    
    # initialize test metrics
    if args.autoencoder:
        test_metrics = TestMetricsManager(
            threshold=args.threshold,
            approach=args.approach
            )
    else:
        test_metrics = TestMetricsManager(approach=args.approach)

    # Calculate the number of steps required to cover all samples in the test set
    num_steps = len(test_kpis) // args.batch_size
    counter = 1
    last_batch_size = len(test_kpis) % args.batch_size

    c=2
    # perform inference on model
    data_set=[]
    failure_set=[]
    non_failure_set=[]
    shap_all=[]
    value_all=[]
    lbl=[]
    ws=[]
    sp=[]
    for kpis, labels in test_ds:
        if counter <180:
            kpis= tf.reshape(kpis,[-1,4*96])
            print("kpis shape", kpis.shape)
            kpis = DataPreprocess.pad_to_batch_size(
                kpis,
                args.batch_size,
                args.approach
                )
        else:
            break
        
        # get k nearest weather stations
        kpis = DataPreprocess.get_k_nearest_WS(
            kpis,
            args.num_weather_stations,
            args.approach
            )

        # run inference
        pc=0
        
        for i in range(len(labels)):
            if labels[i][1]>0.5:
               # print("pdata", i)
                pc=pc+1
                failure_set.append(kpis[i:i+1])
                lbl.append(i)
            else:
                non_failure_set.append(kpis[i:i+1])
                
        data_set.append(kpis)
        
        print("labels", counter, len(lbl))
       # for i in range (len(lbl)):
        #    print ("labels: ", lbl[i])
        def func(z):
            
            inputs1=tf.convert_to_tensor(z)
           # print("inside fn", z.shape)
            pred=[]
            #inputs=tf.reshape(inputs,[1,3,4,84])
            
            
            dataset = tf.data.Dataset.from_tensor_slices(inputs1)

            batched_dataset = dataset.batch(args.batch_size)

            for batch in batched_dataset:
            
                
                batch1=batch
                if len(batch)<1024:
                    
                    batch = DataPreprocess.pad_to_batch_size(
                    batch,
                    args.batch_size,
                    args.approach
                    )
                batch= tf.reshape(batch,[-1,4,96])    
                p=model(batch, training=False)
                p=p[:len(batch1)]
                #for i in range(len(batch1)):
                    #if p[i][1]>0.5:
                        #print("prediction: ", p[i][1])
                pred.append(p)
            pred=tf.concat(pred,0)
            
                    
            return pred

       # z=tf.reshape(kpis, [1024,3*4*84])
        if counter>=152:
            data_set=tf.concat(data_set,0)
            print("size of daata set", data_set.shape)
            kpis=kpis._numpy()
            data_set=data_set._numpy()
            #data_set_df = pd.DataFrame(data_set.numpy())
            #print(data_set_df.iloc[0].shape)
            mask = np.zeros(data_set[0].shape).reshape(1, -1)
    
            def custom_masker(mask, x):
    # in this simple example we just zero out the features we are masking
                return (x * mask).reshape(1, len(x))


# compute SHAP values
            #explainer = shap.Explainer(model.predict_proba, custom_masker)
           # K=256
            
            #masker =  shap.maskers(custom_masker(mask, data_set_df[0]))
           
            
            #masker=custom_masker(mask, data_set_df.iloc[0].to_numpy())
            #masker=shap.maskers.Tabular(data_set, clustering="correlation")
           # print("masker",masker.data.shape)
            #print("mask shape", explainer.masker.mask_shapes)
            explainer = shap.SamplingExplainer(func, data_set) 
            #explainer = shap.explainers.other.LimeTabular(fn,data=data_set, masker=masker)
            #explainer.feature_names =  data_set_df.columns.tolist()
                
#Extract false negative cases from test data
            
            failure_set=tf.concat(failure_set,0)
            fn=0
            print("Total failure cases:", len(failure_set))
            for i in range (len(failure_set)):
                fn=0
                indx=i
              #  print("index", indx)
                z=failure_set[indx]
                #print("shape of z", z.shape)
              #  z=tf.reshape(z, [1,1008]) #for transformer model
                z=tf.reshape(z, [1,4*96]) #for explainability lstm
                z1=z
                
              #  print("shape of z1", z1.shape)
               # z1=tf.reshape(z1, [1,3,4,84])
                z1=tf.reshape(z1, [1,4,96])
                z=z._numpy()
                z1=z1._numpy()  
                pred=func(z)
              # for i in range(len(batch1)):
                p=pred
                if p[0][1]>=0.5:
                        print("prediction in fn: ",i, p[0][0], p[0][1])     
                        fn=+1
            print("Total fn", fn)                  
#Extract false positive cases from test data
            '''
            fp_set=[]
            total_fp=0
            non_failure_set=tf.concat(non_failure_set,0)
            for i in range (len(non_failure_set)):
                indx=i
              #  print("index", indx)
                z=non_failure_set[indx]
                #print("shape of z", z.shape)
               # z=tf.reshape(z, [1,1008])
                z=tf.reshape(z, [1,4*107]) #for explainability lstm
                z1=z
                
              #  print("shape of z1", z1.shape)
                #z1=tf.reshape(z1, [1,3,4,84])
                z1=tf.reshape(z1, [1,4,107])
                z=z._numpy()
                z1=z1._numpy()  
                pred=func(z)
              # for i in range(len(batch1)):
                p=pred
                if p[0][1]>0.5:
                        #print("prediction in fp: ",i, p[0][0], p[0][1])  
                        fp_set.append(z1)
                        total_fp+=1
            '''''                                       
# Extract Shapley values from the explainer
            #print("Total fp", total_fp)
           # fp_set=tf.concat(fp_set,0)
            print("Total failure set", len(failure_set))
            for i in range (len(failure_set)):
                indx=i
              #  print("index", indx)
                z=failure_set[indx]
                #print("shape of z", z.shape)
                z=tf.reshape(z, [1,4*96]) #for explainability lstm
                z1=z
                ws.append(z1[0])
              #  print("shape of z1", z1.shape)
               # z1=tf.reshape(z1, [1,3,4,84])
                z1=tf.reshape(z1, [1,4,96])
                z=z._numpy()
                z1=z1._numpy()
                #print("z1 shape", z1.shape)
                if i==80:
                    break
                #print("z shape",z.shape)
            # print("kpis", kpis.shape)
               # z=z._numpy()
                #x15=x15._numpy()
               # print("z in numpy", z)
            
                if fn==1:
                  shap_values = explainer(z)
                #print("shap val1",shap_values, shap_values.values, shap_values.data )
                #shap.summary_plot(shap_values[:,:,1])
               # shap.plots.bar(shap_values[:,:,1])
                
                  if isinstance(shap_values,np.ndarray) :
                    sv=shap_values[:,:,1]
                  else:
               # print("if not working") 
                    sv=np.array(shap_values.values[:,:,1])
                   #s print("shap shape", sv.shape)
            
            #sv=sv._numpy()    
                 # sv=sv.reshape((1,3,4,84))
                  sp.append(sv[0])
                 # ws.append(z1[0])
                  sv=sv.reshape((1,4,96))
                #z=z.reshape((1,3,4,84))
                  shap_all=np.append(shap_all, sv[0])
                  value_all=np.append(value_all,z1[0])
                  z1=z1[0]
                  sv1=sv[0]
                  
                  #with open(args.log_path+f'fs{indx}_shap_data.txt', 'w') as outfile:
                    #for slice_2d in sv1:
                       # np.savetxt(outfile, slice_2d) 
                    
            
                              
              #  print("data", z1[0].shape, sv1[0].shape)
            
                  #with open(args.log_path+f'input{i}_new_data.txt', 'w') as outfile:
                   # for slice_2d in z1:
                        #np.savetxt(outfile, slice_2d)
                         
            
        counter += 1
        if counter==153:
            break
    '''
            #shap.image_plot(
            # shap_values=shap_values.values,
             #pixel_values=shap_values.data,
             #labels=shap_values.output_names,
           # )
            #sv=tf.convert_to_tensor(shap_values)
            
           # shap.summary_plot(sv[:1,:1,:1,:])
           # p=np.count_nonzero(sv)
            #n=np.zeros(sv)
            #mx=sv.max()
           # print("non_zero", p, mx)
            #shap.summary_plot(sv, z)
            
            
        #sv=sv._numpy()
            
       # print("shap val2",sv.shape)
       
            #shap.plots.bar(shap_values)
       # shap.plots.beeswarm(shap_values[0], max_display=14)
    
        #predictions = fn(z)
        #print(predictions)
       # X100 = shap.utils.sample(kpis, 100)
        #X1=kpis[:last_batch_size]
        
        #print( kpis_num.shape)
        #X2=shap.utils.sample(kpis_num,1)
       # print( X2.shape)
        #X2=X2.reshape(1008, order='C')
       # kpis=kpis._numpy()
        #ig  = IntegratedGradients(f)
        #explanation = ig.explain(z)
        #X500 = shap.utils.sample(test_kpis, 1000)
       # back = np.random.choice(test_kpis.shape,100, replace=False)
       #supports_model(model)
        #df = shap.kmeans(z.values, 25)
# Instantiate an explainer with the model predictions and training data summary
        
        #explainer = shap.KernelExplainer(f,z)
        #shap_values = explainer.shap_values(z)
       
        
       #ss shap.summary_plot(shap_values,X2[0])
        #shap_values = explainer(test_kpis[0])

# make a standard partial dependence plot


        #shap.plots.scatter(shap_values[:, "MedInc"])
       # sample_ind=20
       # shap.plots.waterfall(shap_values[sample_ind], max_display=14)
        # get first valid elements using prev approach
        
        if counter > num_steps:
            predictions = predictions[:last_batch_size]
            labels = labels[:last_batch_size]
            kpis = kpis[:last_batch_size]

        if args.autoencoder:
            test_metrics.update_autoencoder_test_metrics(
                labels,
                predictions,
                kpis
                )
        else:
            test_metrics.update_test_metrics(labels,predictions)
        counter += 1
    '''
    with open(args.log_path+f'input_all_new_data.txt', 'w') as outfile:
                    for slice_2d in ws:
                        np.savetxt(outfile, slice_2d) 
    with open(args.log_path+f'shap_all_new_data.txt', 'w') as outfile:
                    for slice_2d in sp:
                        np.savetxt(outfile, slice_2d) 
    
    '''
    print (value_all.shape)
    print(shap_all.shape)
    data = value_all.reshape(35,3,4,84)
    shap_values = shap_all.reshape(35,3,4,84)
    sum=np.zeros(84)
    sum_abs=np.zeros(84)
   # print(data.shape)
    print (data.shape)
    print(shap_values.shape)
    print( "ex no. ", len(shap_values))
    for i in range(len(shap_values)):
           
         sv=shap_values[i]
         #print("sv size", sv.shape)
         feature_wise_importance = sv.sum(axis=(0, 1))
         feature_wise_importance_abs = np.abs(sv).sum(axis=(0, 1))
         print (feature_wise_importance[3], feature_wise_importance_abs[3])
         for j in range(84):
            sum[j]+=feature_wise_importance[j]
            sum_abs[j]+=feature_wise_importance_abs[j]
            
    '''   
    print(counter, len(ws))
    


if __name__ == '__main__':
    main()
# %%
