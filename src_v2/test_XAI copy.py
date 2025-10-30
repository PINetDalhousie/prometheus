# %%

import os

import lime.lime_base
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import lime.explanation
import tensorflow as tf
from data_preprocess_X import DataPreprocess
from metrics_manager import MetricsManager,TestMetricsManager
import argparse
from to_tfdata_X import ToTfData
import datetime
from utils import plot_distribution
import tensorflow_probability as tfp
import numpy as np
import sys
from metrics_manager import OptimalThresholdCalculator
import shap
import lime
import alibi
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
    parser.add_argument('--positional_encoding', type=bool, default=True,
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
        default="/home/papry/Exp_Project/MS_thesis-main/data/20240310_151048/test_new_rural_time4_None.csv",
        help='Path to test csv file'
        )
    parser.add_argument(
        '--model_path',
        type=str,
        default="/home/papry/Exp_Project/MS_thesis-main/logs/20240618_100218/ckpt",
        help='Path to trained model'
        )  
    parser.add_argument(
        '--log_path',
        type=str,
        default="/home/papry/Exp_Project/MS_thesis-main/logs/20240618_100218/",
        help='Path to log directory')
    parser.add_argument(
        '--approach',
        type=str,
        default="new",
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
    sample= test_kpis[:1]
    colum_name=sample.columns
    
    print("size of the coulum:", len(colum_name))
   
            
            
    with open('/home/papry/Exp_Project/MS_thesis-main/column_name.txt', 'w') as outfile:
                    for i in range (len(colum_name)):
                        if(i<83):
                             outfile.write(colum_name[i]+ ',')
                     
        
    
    # perform inference on model
    
    for kpis, labels in test_ds:
        if counter < 15:
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
        
        def fn(z):
            
            inputs1=tf.convert_to_tensor(z)
            print("inside fn", z.shape)
            #inputs=tf.reshape(inputs,[1,3,4,84])
            inputs = DataPreprocess.pad_to_batch_size(
                inputs1,
                args.batch_size,
                args.approach
                )
            p=model(inputs, training=False)
            p=p[:len(inputs1)]
            print("prediction: ", p)
            return p
        
        def fn1(z):
            inputs1 = tf.convert_to_tensor(z)
            print("inside fn", z.shape)
            inputs = DataPreprocess.pad_to_batch_size(
                inputs1,
                args.batch_size,
                args.approach
            )
            p = model(inputs, training=False)
            p = p[:len(inputs1)]
            print("prediction for alibi: ", p)
            return p.numpy() 

       # z=tf.reshape(kpis, [1024,3*4*84])
        z=kpis[:1]
        x10=kpis[:20]
        print("z shape",z.shape)
        print("kpis", kpis.shape)
        z=z._numpy()
        x10=x10._numpy()
        print("z in numpy", z)
        kpis=kpis._numpy()
        pred=fn(kpis)
        print(pred.shape)
     
        explainer = shap.KernelExplainer(fn, z)
        exp=alibi.explainers.ALE(fn1, z)
        ale=exp.explain(z)
        print("ale", len(ale.ale_values))
        c=0
        for feature, values in zip(ale.feature_names, ale.ale_values):
            print(f"Feature: {feature}")
            print(f"ALE Values: {values}")
            c += 1
        print("c", c)
# Extract Shapley values from the explainer
        shap_values = explainer.shap_values(z)
       # shap.summary_plot(shap_values,z)
        print("shap val",shap_values[0])
        ig = IntegratedGradients(fn1)   # TensorFlow model
        result = ig.explain(z)                   # model class probability prediction to obtain attribution for 
        print("result", result)

        #plot_importance(result.data['attributions'][0], features, 0)
           
       # shap.plots.beeswarm(shap_values[0], max_display=14)
        counter += 1
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
        '''
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
    

if __name__ == '__main__':
    main()
# %%
