import re
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, Normalizer,OneHotEncoder
import joblib
import numpy as np
from imblearn.under_sampling import RandomUnderSampler


class DataPreprocess:
    Approaches = ["new","prev"]
    
    def __init__(self,approach:str) -> None:
        
        # Check if input is valid
        if approach in DataPreprocess.Approaches:
            self.approach = approach
        else:
            raise ValueError("Invalid argument")

        self.train_data = None
        self.val_data = None
        self.train_kpis = None
        self.train_labels = None
        self.val_kpis = None
        self.val_labels = None
        self.order_preserved_columns = None
        self.prev_day_data=4
        self.num_stations=3
        self.cat_feature_number = None
        self.feature_number = None
        self.test_data = None

        # Read categorical feature list and conver to current
        # list of categorical features for new approach
        

    def get_train_kpis_labels(self):
        """
        Return current train kpis and labels
        """
        return self.train_kpis,self.train_labels
    

    def get_val_kpis_labels(self):
        """
        Return current train kpis and labels
        """
        return self.val_kpis,self.val_labels


    def seperate_kpis_labels(self) -> None:
        """
        Seperate data into kpis and label
        """
        self.train_kpis = self.train_data.drop(["1-day-predict"],axis=1)
        self.train_labels = self.train_data["1-day-predict"]
        self.val_kpis = self.val_data.drop(["1-day-predict"],axis=1)
        self.val_labels = self.val_data["1-day-predict"]
        self.train_data = None
        self.val_data = None

    def __min_max_scale_train(self,current_time,log_path):

        # Get all numeric columns
        numerical_columns = self.train_kpis._get_numeric_data().columns

        if log_path:
            # Find the last index of the '/' character
            last_slash_index = log_path.rfind('/')
            # Extract the substring up to the last slash
            log_path = log_path[:last_slash_index+1]

            min_max_scaler = joblib.load(log_path+f"min_max_scaler_{self.approach}.save")
            # Scale numerical columns only
            train_df_num_scaled = pd.DataFrame(min_max_scaler.transform(self.train_kpis[numerical_columns]), columns=numerical_columns)
        else:
            # Scale numerical columns only
            min_max_scaler = MinMaxScaler()
            train_df_num_scaled = pd.DataFrame(min_max_scaler.fit_transform(self.train_kpis[numerical_columns]), columns=numerical_columns)

        # Get all categorical columns
        cat_cols = list(set(self.train_kpis.columns) - set(numerical_columns))
        
        # Combine scaled numerical columns with categorical column
        train_df_scaled = pd.concat([train_df_num_scaled, self.train_kpis[cat_cols].reset_index()], axis=1)

        # Rearrange columns to original order
        cols = list(self.train_kpis.columns)
        self.train_kpis = train_df_scaled[cols]

        # Save scaler
        joblib.dump(min_max_scaler,f"/home/papry/Exp_Project/MS_thesis-main/logs/{current_time}/min_max_scaler_{self.approach}.save")

        return min_max_scaler

    
    def __min_max_scale_val(self,min_max_scaler):

        # Get all numeric columns
        numerical_columns = self.val_kpis._get_numeric_data().columns
        # Get all categorical columns
        cat_cols = list(set(self.val_kpis.columns) - set(numerical_columns))

        # Scale numerical columns only
        val_df_num_scaled = pd.DataFrame(min_max_scaler.transform(self.val_kpis[numerical_columns]), columns=numerical_columns)

        # Combine scaled numerical columns with categorical column
        val_df_scaled = pd.concat([val_df_num_scaled, self.val_kpis[cat_cols].reset_index()], axis=1)

        # Rearrange columns to original order
        cols = list(self.val_kpis.columns)
        self.val_kpis = val_df_scaled[cols]
        

    def min_max_scale(self,current_time,log_path) -> None:
        """
        Min max scale the data and save the scalar
        """
        # Scale Train data
        min_max_scaler = self.__min_max_scale_train(current_time,log_path)
        # Scale Validation data
        self.__min_max_scale_val(min_max_scaler)
        
    def __calc_cat_feat_number(self,df_encoded:pd.DataFrame) -> int:
        """
        Calculates the number of categorical features per time step
        """
        if self.approach == "prev":
            cat_feature_number = int(df_encoded.shape[1]/self.prev_day_data)
        elif self.approach == "new":
            cat_feature_number = int((df_encoded.shape[1]/self.num_stations)/self.prev_day_data)
            print("this is cat feature", df_encoded.shape[1], cat_feature_number)
        self.cat_feature_number = cat_feature_number

    def __calc_feat_number(self,train_encoded:pd.DataFrame) -> int:
        """
        Calculates the number of features per time step
        """
        if self.approach == "prev":
            feature_number = int(train_encoded.shape[1]/self.prev_day_data)
        elif self.approach == "new":
            feature_number = int((train_encoded.shape[1]/self.num_stations)/self.prev_day_data)
        print("this is total feature",train_encoded.shape[1], feature_number)
        self.feature_number = feature_number
       

    def __normalize_train(self,current_time,log_path):

        # Get all numeric columns
        numerical_columns = self.train_kpis._get_numeric_data().columns

        if log_path:
            # Find the last index of the '/' character
            last_slash_index = log_path.rfind('/')
            # Extract the substring up to the last slash
            log_path = log_path[:last_slash_index+1]

            normalizer = joblib.load(log_path+f"normalize_{self.approach}.save")
            # Scale numerical columns only
            train_df_num_scaled = pd.DataFrame(normalizer.transform(self.train_kpis[numerical_columns]), columns=numerical_columns)
        else:
            # Scale numerical columns only
            normalizer = Normalizer()
            train_df_num_scaled = pd.DataFrame(normalizer.fit_transform(self.train_kpis[numerical_columns]), columns=numerical_columns)

        # Get all categorical columns
        cat_cols = list(set(self.train_kpis.columns) - set(numerical_columns))
        
        # Combine scaled numerical columns with categorical column
        train_df_scaled = pd.concat([train_df_num_scaled, self.train_kpis[cat_cols].reset_index()], axis=1)

        # Rearrange columns to original order
        cols = list(self.train_kpis.columns)
        self.train_kpis = train_df_scaled[cols]

        # Save scaler
        joblib.dump(normalizer,f"/home/papry/Exp_Project/MS_thesis-main/logs/{current_time}/normalize_{self.approach}.save")

        return normalizer

    
    def __normalize_val(self,normalizer):

        # Get all numeric columns
        numerical_columns = self.val_kpis._get_numeric_data().columns
        # Get all categorical columns
        cat_cols = list(set(self.val_kpis.columns) - set(numerical_columns))

        # Scale numerical columns only
        val_df_num_scaled = pd.DataFrame(normalizer.transform(self.val_kpis[numerical_columns]), columns=numerical_columns)

        # Combine scaled numerical columns with categorical column
        val_df_scaled = pd.concat([val_df_num_scaled, self.val_kpis[cat_cols].reset_index()], axis=1)

        # Rearrange columns to original order
        cols = list(self.val_kpis.columns)
        self.val_kpis = val_df_scaled[cols]
        

    def normalize(self,current_time,log_path) -> None:
        """
       normalize scale the data and save the scalar
        """
        # Scale Train data
        normalizer = self.__normalize_train(current_time,log_path)
        # Scale Validation data
        self.__normalize_val(normalizer)

    def __one_hot_encode_train(self,current_time,log_path):
        # get categorical columns
        cat_cols = self.train_kpis.select_dtypes(include=['object']).columns.tolist()

        if log_path:
            # Find the last index of the '/' character
            last_slash_index = log_path.rfind('/')

            # Extract the substring up to the last slash
            log_path = log_path[:last_slash_index+1]

            one_hot_encoder = joblib.load(log_path+f"one_hot_encoder_{self.approach}.save")
            one_hot_encoder.handle_unknown = 'ignore'
            df_encoded = one_hot_encoder.transform(self.train_kpis[cat_cols])
        else:
            # Create OneHotEncoder object
            one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
            # Fit and transform data
            df_encoded = one_hot_encoder.fit_transform(self.train_kpis[cat_cols])

        self.__calc_cat_feat_number(df_encoded)
        # Create dataframe with column names
        df_encoded = pd.DataFrame(
            df_encoded.toarray(),
            columns=one_hot_encoder.get_feature_names_out(cat_cols)
            )
        
        def normalize_column_name(column_name):
            if "WS2" in column_name:
                return re.sub(r'_WS2+', '', column_name)
            return column_name

        # Function to find columns that are the same except for the "WS" value and print mismatches
        def find_and_print_mismatches(df_1, df_2):
            columns = df_2
            normalized_columns = [normalize_column_name(col) for col in columns]
            similar_columns = df_1
            df_12 = []
            for i, col in enumerate(normalized_columns):
               # normalized_col = normalized_columns[i]
                if col not in similar_columns:
                    df_12.append(col)
                #similar_columns[normalized_col].append(col)
            print("Columns that are the same except for the 'WS' value:")
            print(df_12)
            # Filter out entries with only one column
            #mismatches = {k: v for k, v in similar_columns.items() if len(v) == 1}

            # Print mismatches
        def find_and_print_2(df_1, df_2):
            columns_1 = df_1
            columns_2 = df_2

            normalized_columns_1 = [normalize_column_name(col) for col in columns_1]
            normalized_columns_2 = [normalize_column_name(col) for col in columns_2]
            for i in range(10):
               print(normalized_columns_1[i], normalized_columns_2[i])
            
            mismatches_1 = [col for col in columns_1 if normalize_column_name(col) not in normalized_columns_2]
            mismatches_2 = [col for col in columns_2 if normalize_column_name(col) not in normalized_columns_1]

            #print("Columns in df_1 but not in df_2 (normalized):")
           # print(mismatches_1)
            print("Columns in df_2 but not in df_1 (normalized):")
            print(mismatches_2)    

        # Example usage with df_encoded
       # find_and_print_mismatches(df_encoded)
       
        ws3_count=0
        ws5_count=0
        ws4_count=0
        ws1_count=0
        ws2_count=0
        other=0
        df_1=[]
        df_2=[]
        df_34=[]
        df_13=[]
        for column in df_encoded.columns.unique():
            if "WS3" in column:
                ws3_count+=1
                #df_32.append(column)
                df_34.append(column)
                df_13.append(column)
            elif "WS2" in column:
                ws2_count+=1
                df_2.append(column)
            elif "WS4" in column:
                ws4_count+=1 
                df_34.append(column)
            elif "WS1" in column:
                ws1_count+=1
                df_13.append(column)
                            
            else:
                #print(column)
                other+=1  
                df_1.append(column)
        print(f"ws3: {ws3_count}, ws4: {ws4_count}, oth: {other},  ws1: {ws1_count}, ws2: {ws2_count}")
       
       # max_count = max(ws3_count, ws4_count, ws5_count, ws1_count, ws2_count)
        #print("for weather st 1")
       # find_and_print_mismatches(df_13)
        print("for weather st 1")
        find_and_print_2(df_1, df_2)
       # print("for weather st 2")
       ## find_and_print_mismatches(df_32)
        ##### Fix for new technique in one hot encoding ws3 met clutter
        if self.approach == "new":
            ws3_clutter_columns = []
            ws4_clutter_columns = []
            ws44_clutter_columns = []
            wslmu_clutter_columns = []
            wsgs_clutter_columns = []
            wslsu_clutter_columns = []
            wsa_clutter_columns = []


            if "clutter_class_met_stations_WS3_AIRPORT" not in df_encoded.columns.to_list():
                df_encoded['clutter_class_met_stations_WS3_AIRPORT'] = 0.0
                for i in range(1,self.prev_day_data):
                    df_encoded[f"clutter_class_met_stations_T-{i}_WS3_AIRPORT"] = 0.0
                ws3_clutter_columns = [
                    "clutter_class_met_stations_WS3",
                    "clutter_class_met_stations_T-1_WS3",
                    "clutter_class_met_stations_T-2_WS3",
                    "clutter_class_met_stations_T-3_WS3"
                   # "clutter_class_met_stations_T-4_WS3"
                   # "clutter_class_met_stations_T-5_WS3"
                   # "clutter_class_met_stations_T-6_WS3",
                   # "clutter_class_met_stations_T-7_WS3"

                ]
            if "clutter_class_met_stations_WS4_AIRPORT" not in df_encoded.columns.to_list():
                df_encoded['clutter_class_met_stations_WS4_AIRPORT'] = 0.0
                for i in range(1,self.prev_day_data):
                    df_encoded[f"clutter_class_met_stations_T-{i}_WS4_AIRPORT"] = 0.0
                ws4_clutter_columns = [
                    "clutter_class_met_stations_WS4",
                    "clutter_class_met_stations_T-1_WS4",
                    "clutter_class_met_stations_T-2_WS4",
                    "clutter_class_met_stations_T-3_WS4"
                   # "clutter_class_met_stations_T-4_WS3"
                   # "clutter_class_met_stations_T-5_WS3"
                   # "clutter_class_met_stations_T-6_WS3",
                   # "clutter_class_met_stations_T-7_WS3"

                ]
            if "clutter_class_met_stations_WS4_LOW-SPARSE URBAN" not in df_encoded.columns.to_list():
                df_encoded['clutter_class_met_stations_WS4_LOW-SPARSE URBAN'] = 0.0
                for i in range(1,self.prev_day_data):
                    df_encoded[f"clutter_class_met_stations_T-{i}_WS4_LOW-SPARSE URBAN"] = 0.0
                ws44_clutter_columns = [
                    "clutter_class_met_stations_WS4",
                    "clutter_class_met_stations_T-1_WS4",
                    "clutter_class_met_stations_T-2_WS4",
                    "clutter_class_met_stations_T-3_WS4"
                   # "clutter_class_met_stations_T-4_WS3"
                   # "clutter_class_met_stations_T-5_WS3"
                   # "clutter_class_met_stations_T-6_WS3",
                   # "clutter_class_met_stations_T-7_WS3"

                ]
             #   self.cat_feature_number += 1
            if "clutter_class_met_stations_LOW-SPARSE URBAN" not in df_encoded.columns.to_list():
                df_encoded['clutter_class_met_stations_LOW-SPARSE URBAN'] = 0.0
                for i in range(1,self.prev_day_data):
                    df_encoded[f"clutter_class_met_stations_T-{i}_LOW-SPARSE URBAN"] = 0.0
                wslsu_clutter_columns = [
                    "clutter_class_met_stations",
                    "clutter_class_met_stations_T-1",
                    "clutter_class_met_stations_T-2",
                    "clutter_class_met_stations_T-3"
                   # "clutter_class_met_stations_T-4"
                   # "clutter_class_met_stations_T-5"
                   # "clutter_class_met_stations_T-6",
                   # "clutter_class_met_stations_T-7"
                ]
                
            if "clutter_class_met_stations_AIRPORT" not in df_encoded.columns.to_list():
                    df_encoded['clutter_class_met_stations_AIRPORT'] = 0.0
                    for i in range(1,self.prev_day_data):
                        df_encoded[f"clutter_class_met_stations_T-{i}_AIRPORT"] = 0.0
                    wsa_clutter_columns = [
                        "clutter_class_met_stations",
                        "clutter_class_met_stations_T-1",
                        "clutter_class_met_stations_T-2",
                        "clutter_class_met_stations_T-3"
                       # "clutter_class_met_stations_T-4"
                      #  "clutter_class_met_stations_T-5"
                     #   "clutter_class_met_stations_T-6",
                     #   "clutter_class_met_stations_T-7"
                    ]
                   # self.cat_feature_number += 1
            if "clutter_class_met_stations_LOW-MEDIUM URBAN" not in df_encoded.columns.to_list():
                df_encoded['clutter_class_met_stations_LOW-MEDIUM URBAN'] = 0.0
                for i in range(1,self.prev_day_data):
                    df_encoded[f"clutter_class_met_stations_T-{i}_LOW-MEDIUM URBAN"] = 0.0
                wslmu_clutter_columns = [
                    "clutter_class_met_stations",
                    "clutter_class_met_stations_T-1",
                    "clutter_class_met_stations_T-2",
                    "clutter_class_met_stations_T-3"
                  #  "clutter_class_met_stations_T-4"
                  #  "clutter_class_met_stations_T-5"
                   # "clutter_class_met_stations_T-6",
                 #   "clutter_class_met_stations_T-7"
                ]
            if "clutter_class_met_stations_GREEN HOUSE" not in df_encoded.columns.to_list():
                df_encoded['clutter_class_met_stations_GREEN HOUSE'] = 0.0
                for i in range(1,self.prev_day_data):
                    df_encoded[f"clutter_class_met_stations_T-{i}_GREEN HOUSE"] = 0.0
                wsgs_clutter_columns = [
                    "clutter_class_met_stations",
                    "clutter_class_met_stations_T-1",
                    "clutter_class_met_stations_T-2",
                    "clutter_class_met_stations_T-3"
                   # "clutter_class_met_stations_T-4"
                   # "clutter_class_met_stations_T-5"
                  #  "clutter_class_met_stations_T-6",
                  #  "clutter_class_met_stations_T-7"
                ]
        self.__calc_cat_feat_number(df_encoded)
        # create new dataframe with encoded columns
        
        ws3_count=0
        ws5_count=0
        ws4_count=0
        ws1_count=0
        ws2_count=0
        other=0
        df_32=[]
        df_34=[]
        df_13=[]
        for column in df_encoded.columns.unique():
            if "WS3" in column:
                ws3_count+=1
                df_32.append(column)
                df_34.append(column)
                df_13.append(column)
            elif "WS2" in column:
                ws2_count+=1
                df_32.append(column)
            elif "WS4" in column:
                ws4_count+=1 
                df_34.append(column)
            elif "WS1" in column:
                ws1_count+=1
                df_13.append(column)
                            
            else:
                #print(column)
                other+=1  
               # df_13.append(column)
        print(f"ws3: {ws3_count}, ws4: {ws4_count}, oth: {other},  ws1: {ws1_count}, ws2: {ws2_count}")
       
       # max_count = max(ws3_count, ws4_count, ws5_count, ws1_count, ws2_count)
        #print("for weather st 1")
       # find_and_print_mismatches(df_13)
        #print("for weather st 4")
        #find_and_print_mismatches(df_34)
       # print("for weather st 2")
       # find_and_print_mismatches(df_32)
        df_encoded = pd.concat([
            self.train_kpis.drop(cat_cols, axis=1),
            df_encoded],
            axis=1
            )

        # Get column list with original ordering
        order_preserved_columns = []
        categorical_column_index = 0
        for column in self.train_kpis.columns.to_list():
            # Create new column names for each category
            if column in cat_cols:
                column_categories = one_hot_encoder.categories_[categorical_column_index]
                column_names = [f'{column}_{category}' for category in column_categories]
                ##### Fix for new technique in one hot encoding ws3 met clutter
                if self.approach == "new":
                    if column in ws3_clutter_columns:
                        column_names.insert(0,column+"_AIRPORT")
                    if column in ws4_clutter_columns:
                        column_names.insert(0,column+"_AIRPORT") 
                    if column in ws44_clutter_columns:
                        column_names.insert(8,column+"_LOW-SPARSE URBAN")   
                    if column in wslsu_clutter_columns:
                        column_names.insert(8,column+"_LOW-SPARSE URBAN")
                    if column in wsa_clutter_columns:
                        column_names.insert(0,column+"_AIRPORT")   
                    if column in wslmu_clutter_columns:
                        column_names.insert(7,column+"_LOW-MEDIUM URBAN")
                    if column in wsgs_clutter_columns:
                        column_names.insert(4,column+"_GREEN HOUSE")   
                            
                order_preserved_columns += column_names
                categorical_column_index += 1
            else:
                # Add non-categorical columns
                order_preserved_columns.append(column)

        self.order_preserved_columns = order_preserved_columns
        # Reorder columns to preserve original ordering
        self.train_kpis = df_encoded[order_preserved_columns]

        # Calculate number of features per time step
        self.__calc_feat_number(self.train_kpis)

        joblib.dump(one_hot_encoder,f"/users/grad/papry/MS_thesis-main/logs/{current_time}/one_hot_encoder_{self.approach}.save")
        return one_hot_encoder


    def __one_hot_encode_val(self,one_hot_encoder):
        # One hot encode validation data
        # get categorical columns
        cat_cols = self.val_kpis.select_dtypes(include=['object']).columns.tolist()

        # Transform data
        df_encoded = one_hot_encoder.transform(self.val_kpis[cat_cols])
        # Create dataframe with column names
        df_encoded = pd.DataFrame(df_encoded.toarray(), columns=one_hot_encoder.get_feature_names_out(cat_cols))
        # create new dataframe with encoded columns
        df_encoded = pd.concat([self.val_kpis.drop(cat_cols, axis=1), df_encoded], axis=1)

        ##### Fix for new technique
        if self.approach == "new":
            if "clutter_class_met_stations_WS3_AIRPORT" not in df_encoded.columns.to_list():
                df_encoded['clutter_class_met_stations_WS3_AIRPORT'] = 0.0
                for i in range(1,self.prev_day_data):
                    df_encoded[f"clutter_class_met_stations_T-{i}_WS3_AIRPORT"] = 0.0
            if "clutter_class_met_stations_WS4_AIRPORT" not in df_encoded.columns.to_list():
                df_encoded['clutter_class_met_stations_WS4_AIRPORT'] = 0.0
                for i in range(1,self.prev_day_data):
                    df_encoded[f"clutter_class_met_stations_T-{i}_WS4_AIRPORT"] = 0.0
            if "clutter_class_met_stations_WS4_LOW-SPARSE URBAN" not in df_encoded.columns.to_list():
                df_encoded['clutter_class_met_stations_WS4_LOW-SPARSE URBAN'] = 0.0
                for i in range(1,self.prev_day_data):
                    df_encoded[f"clutter_class_met_stations_T-{i}_WS4_LOW-SPARSE URBAN"] = 0.0
            if "clutter_class_met_stations_AIRPORT" not in df_encoded.columns.to_list():
                df_encoded['clutter_class_met_stations_AIRPORT'] = 0.0
                for i in range(1,self.prev_day_data):
                    df_encoded[f"clutter_class_met_stations_T-{i}_AIRPORT"] = 0.0
            
            if "clutter_class_met_stations_GREEN HOUSE" not in df_encoded.columns.to_list():
                df_encoded['clutter_class_met_stations_GREEN HOUSE'] = 0.0
                for i in range(1,self.prev_day_data):
                    df_encoded[f"clutter_class_met_stations_T-{i}_GREEN HOUSE"] = 0.0

        # Reorder columns to preserve original ordering
        self.val_kpis = df_encoded[self.order_preserved_columns]



    def one_hot_encode(self,current_time,log_path):
        """
        Performs one-hot encoding on the categorical columns of the data.
        """
        # One hot encode train data
        one_hot_encoder = self.__one_hot_encode_train(current_time,log_path)
        self.__one_hot_encode_val(one_hot_encoder)
    
        
    def one_hot_encode_test_data(self,log_path,ignore_cat):
        # get categorical columns
        cat_cols = self.test_kpis.select_dtypes(include=['object']).columns.tolist()
        one_hot_encoder = joblib.load(log_path+f"one_hot_encoder_{self.approach}.save")
        one_hot_encoder.handle_unknown = 'ignore'

        # Transform data
        if ignore_cat == False:
            df_encoded = one_hot_encoder.transform(self.test_kpis[cat_cols])
            # Create dataframe with column names
            df_encoded = pd.DataFrame(df_encoded.toarray(), columns=one_hot_encoder.get_feature_names_out(cat_cols))

        if ignore_cat == True:
            # This way we ignore categorical
            num_rows = len(self.test_kpis)
            df_encoded = pd.DataFrame(0, index=range(num_rows), columns=one_hot_encoder.get_feature_names_out(cat_cols))

        ##### Fix for new technique in one hot encoding ws3 met clutter
        if self.approach == "new":
            ws3_clutter_columns = []
            ws4_clutter_columns = []
            ws44_clutter_columns = []
            wslmu_clutter_columns = []
            wsgs_clutter_columns = []
            wslsu_clutter_columns = []
            wsa_clutter_columns = []


            if "clutter_class_met_stations_WS3_AIRPORT" not in df_encoded.columns.to_list():
                df_encoded['clutter_class_met_stations_WS3_AIRPORT'] = 0.0
                for i in range(1,self.prev_day_data):
                    df_encoded[f"clutter_class_met_stations_T-{i}_WS3_AIRPORT"] = 0.0
                ws3_clutter_columns = [
                    "clutter_class_met_stations_WS3",
                    "clutter_class_met_stations_T-1_WS3",
                    "clutter_class_met_stations_T-2_WS3",
                    "clutter_class_met_stations_T-3_WS3"
                   # "clutter_class_met_stations_T-4_WS3"
                   # "clutter_class_met_stations_T-5_WS3"
                   # "clutter_class_met_stations_T-6_WS3",
                   # "clutter_class_met_stations_T-7_WS3"

                ]
            if "clutter_class_met_stations_WS4_AIRPORT" not in df_encoded.columns.to_list():
                df_encoded['clutter_class_met_stations_WS4_AIRPORT'] = 0.0
                for i in range(1,self.prev_day_data):
                    df_encoded[f"clutter_class_met_stations_T-{i}_WS4_AIRPORT"] = 0.0
                ws4_clutter_columns = [
                    "clutter_class_met_stations_WS4",
                    "clutter_class_met_stations_T-1_WS4",
                    "clutter_class_met_stations_T-2_WS4",
                    "clutter_class_met_stations_T-3_WS4"
                   # "clutter_class_met_stations_T-4_WS3"
                   # "clutter_class_met_stations_T-5_WS3"
                   # "clutter_class_met_stations_T-6_WS3",
                   # "clutter_class_met_stations_T-7_WS3"

                ]
            if "clutter_class_met_stations_WS4_LOW-SPARSE URBAN" not in df_encoded.columns.to_list():
                df_encoded['clutter_class_met_stations_WS4_LOW-SPARSE URBAN'] = 0.0
                for i in range(1,self.prev_day_data):
                    df_encoded[f"clutter_class_met_stations_T-{i}_WS4_LOW-SPARSE URBAN"] = 0.0
                ws44_clutter_columns = [
                    "clutter_class_met_stations_WS4",
                    "clutter_class_met_stations_T-1_WS4",
                    "clutter_class_met_stations_T-2_WS4",
                    "clutter_class_met_stations_T-3_WS4"
                   # "clutter_class_met_stations_T-4_WS3"
                   # "clutter_class_met_stations_T-5_WS3"
                   # "clutter_class_met_stations_T-6_WS3",
                   # "clutter_class_met_stations_T-7_WS3"

                ]
             #   self.cat_feature_number += 1
            if "clutter_class_met_stations_LOW-SPARSE URBAN" not in df_encoded.columns.to_list():
                df_encoded['clutter_class_met_stations_LOW-SPARSE URBAN'] = 0.0
                for i in range(1,self.prev_day_data):
                    df_encoded[f"clutter_class_met_stations_T-{i}_LOW-SPARSE URBAN"] = 0.0
                wslsu_clutter_columns = [
                    "clutter_class_met_stations",
                    "clutter_class_met_stations_T-1",
                    "clutter_class_met_stations_T-2",
                    "clutter_class_met_stations_T-3"
                   # "clutter_class_met_stations_T-4"
                   # "clutter_class_met_stations_T-5"
                   # "clutter_class_met_stations_T-6",
                   # "clutter_class_met_stations_T-7"
                ]
                
            if "clutter_class_met_stations_AIRPORT" not in df_encoded.columns.to_list():
                    df_encoded['clutter_class_met_stations_AIRPORT'] = 0.0
                    for i in range(1,self.prev_day_data):
                        df_encoded[f"clutter_class_met_stations_T-{i}_AIRPORT"] = 0.0
                    wsa_clutter_columns = [
                        "clutter_class_met_stations",
                        "clutter_class_met_stations_T-1",
                        "clutter_class_met_stations_T-2",
                        "clutter_class_met_stations_T-3"
                       # "clutter_class_met_stations_T-4"
                      #  "clutter_class_met_stations_T-5"
                     #   "clutter_class_met_stations_T-6",
                     #   "clutter_class_met_stations_T-7"
                    ]
                   # self.cat_feature_number += 1
            if "clutter_class_met_stations_LOW-MEDIUM URBAN" not in df_encoded.columns.to_list():
                df_encoded['clutter_class_met_stations_LOW-MEDIUM URBAN'] = 0.0
                for i in range(1,self.prev_day_data):
                    df_encoded[f"clutter_class_met_stations_T-{i}_LOW-MEDIUM URBAN"] = 0.0
                wslmu_clutter_columns = [
                    "clutter_class_met_stations",
                    "clutter_class_met_stations_T-1",
                    "clutter_class_met_stations_T-2",
                    "clutter_class_met_stations_T-3"
                  #  "clutter_class_met_stations_T-4"
                  #  "clutter_class_met_stations_T-5"
                   # "clutter_class_met_stations_T-6",
                 #   "clutter_class_met_stations_T-7"
                ]
            if "clutter_class_met_stations_GREEN HOUSE" not in df_encoded.columns.to_list():
                df_encoded['clutter_class_met_stations_GREEN HOUSE'] = 0.0
                for i in range(1,self.prev_day_data):
                    df_encoded[f"clutter_class_met_stations_T-{i}_GREEN HOUSE"] = 0.0
                wsgs_clutter_columns = [
                    "clutter_class_met_stations",
                    "clutter_class_met_stations_T-1",
                    "clutter_class_met_stations_T-2",
                    "clutter_class_met_stations_T-3"
                   # "clutter_class_met_stations_T-4"
                   # "clutter_class_met_stations_T-5"
                  #  "clutter_class_met_stations_T-6",
                  #  "clutter_class_met_stations_T-7"
                ]
            

        # create new dataframe with encoded columns
        df_encoded = pd.concat([self.test_kpis.drop(cat_cols, axis=1), df_encoded], axis=1)

        # Get column list with original ordering
        order_preserved_columns = []
        categorical_column_index = 0
        for column in self.test_kpis.columns.to_list():
            # Create new column names for each category
            if column in cat_cols:
                column_categories = one_hot_encoder.categories_[categorical_column_index]
                column_names = [f'{column}_{category}' for category in column_categories]
                ##### Fix for new technique in one hot encoding ws3 met clutter
                if self.approach == "new":
                    if column in ws3_clutter_columns:
                        column_names.insert(0,column+"_AIRPORT")
                    if column in ws4_clutter_columns:
                        column_names.insert(0,column+"_AIRPORT") 
                    if column in ws44_clutter_columns:
                        column_names.insert(8,column+"_LOW-SPARSE URBAN")   
                    if column in wslsu_clutter_columns:
                        column_names.insert(8,column+"_LOW-SPARSE URBAN")
                    if column in wsa_clutter_columns:
                        column_names.insert(0,column+"_AIRPORT")   
                    if column in wslmu_clutter_columns:
                        column_names.insert(7,column+"_LOW-MEDIUM URBAN")
                    if column in wsgs_clutter_columns:
                        column_names.insert(4,column+"_GREEN HOUSE")   
                order_preserved_columns += column_names
                categorical_column_index += 1
            else:
                # Add non-categorical columns
                order_preserved_columns.append(column)

        # Reorder columns to preserve original ordering
        self.test_kpis = df_encoded[order_preserved_columns]

    
    def read_train_val_csv(self,train_csv,val_csv) -> None:
        """
        Reads train and validation data from csv files
        """
        self.train_data = pd.read_csv(train_csv)
        self.val_data = pd.read_csv(val_csv)
    
    def read_test_csv(self,test_csv) -> None:
        """
        Reads test data from csv file
        """
        self.test_data = pd.read_csv(test_csv)

    def seperate_test_kpis_labels(self):
        """
        Seperate test data into kpis and label
        """
        self.test_kpis = self.test_data.drop(["1-day-predict"],axis=1)
        self.test_labels = self.test_data["1-day-predict"]


    def scale_test_data(self) -> None:
        """
        Load min max scaler from path and scale test data when there is no categorical features
        """
        scaler = joblib.load(f"/home/papry/Exp_Project/MS_thesis-main/scalers/min_max_scaler_{self.approach}.save")
        self.test_kpis = scaler.transform(self.test_kpis)

    def min_max_scale_test_data(self,log_path):
        
        # Get all numeric columns
        numerical_columns = self.test_kpis._get_numeric_data().columns
        # Get all categorical columns
        cat_cols = list(set(self.test_kpis.columns) - set(numerical_columns))

        min_max_scaler = joblib.load(log_path+f"min_max_scaler_{self.approach}.save")
        # Scale numerical columns only
        test_df_num_scaled = pd.DataFrame(min_max_scaler.transform(self.test_kpis[numerical_columns]), columns=numerical_columns)
        # if True:
        #     # Get all numeric columns
        #     numerical_columns = self.test_kpis._get_numeric_data().columns

        #     # Get all categorical columns
        #     cat_cols = list(set(self.test_kpis.columns) - set(numerical_columns))

        #     # Scale numerical columns only
        #     min_max_scaler = MinMaxScaler()
        #     test_df_num_scaled = pd.DataFrame(min_max_scaler.fit_transform(self.test_kpis[numerical_columns]), columns=numerical_columns)

        # Combine scaled numerical columns with categorical column
        test_df_scaled = pd.concat([test_df_num_scaled, self.test_kpis[cat_cols].reset_index()], axis=1)

        # Rearrange columns to original order
        cols = list(self.test_kpis.columns)
        self.test_kpis = test_df_scaled[cols]

    def normalize_test_data(self,log_path):
        
        # Get all numeric columns
        numerical_columns = self.test_kpis._get_numeric_data().columns
        # Get all categorical columns
        cat_cols = list(set(self.test_kpis.columns) - set(numerical_columns))

        normalizer = joblib.load(log_path+f"normalize_{self.approach}.save")
        # Scale numerical columns only
        test_df_num_scaled = pd.DataFrame(normalizer.transform(self.test_kpis[numerical_columns]), columns=numerical_columns)
        # if True:
        #     # Get all numeric columns
        #     numerical_columns = self.test_kpis._get_numeric_data().columns

        #     # Get all categorical columns
        #     cat_cols = list(set(self.test_kpis.columns) - set(numerical_columns))

        #     # Scale numerical columns only
        #     min_max_scaler = MinMaxScaler()
        #     test_df_num_scaled = pd.DataFrame(min_max_scaler.fit_transform(self.test_kpis[numerical_columns]), columns=numerical_columns)

        # Combine scaled numerical columns with categorical column
        test_df_scaled = pd.concat([test_df_num_scaled, self.test_kpis[cat_cols].reset_index()], axis=1)

        # Rearrange columns to original order
        cols = list(self.test_kpis.columns)
        self.test_kpis = test_df_scaled[cols]

    def get_test_kpis_labels(self):
        """
        Return current test kpis and labels
        """
        return self.test_kpis,self.test_labels
    
    def remove_clutter_met(self):
        # remove clutter class met stations columns using regex
        train_cols_to_remove = self.train_kpis.filter(
            regex='^(clutter_class_met_stations)'
            ).columns
        self.train_kpis = self.train_kpis.drop(train_cols_to_remove, axis=1)
        val_cols_to_remove = self.val_kpis.filter(
            regex='^(clutter_class_met_stations)'
            ).columns
        self.val_kpis = self.val_kpis.drop(val_cols_to_remove, axis=1)

    

    def fix_encoded_cluttermet(self):
        print(self.train_kpis.info(verbose=True))


       # print(asd)
        
        # remove clutter class met stations columns using regex
        train_cols_to_remove = self.train_kpis.filter(
            regex='^(clutter_class_met_stations)'
            ).columns
        self.train_kpis = self.train_kpis.drop(train_cols_to_remove, axis=1)
        val_cols_to_remove = self.val_kpis.filter(
            regex='^(clutter_class_met_stations)'
            ).columns
        self.val_kpis = self.val_kpis.drop(val_cols_to_remove, axis=1)

    def remove_test_clutter_met(self):
        test_cols_to_remove = self.test_kpis.filter(
            regex='^(clutter_class_met_stations)'
            ).columns
        self.test_kpis = self.test_kpis.drop(test_cols_to_remove, axis=1)
        
    def remove_failures(self):
        # remove failures from train and validation data
        self.train_data = self.train_data[self.train_data['1-day-predict'] != 1]
        self.val_data = self.val_data[self.val_data['1-day-predict'] != 1]


    def remove_test_failures(self):
        # remove failures from test data
        self.test_data = self.test_data[self.test_data['1-day-predict'] == 0]
        

    def undersample(self,fraction):
        # perform majority undersampling
        train_kpis = self.train_data.drop(["1-day-predict"],axis=1)
        train_labels = self.train_data["1-day-predict"]

        random_undersampler = RandomUnderSampler(sampling_strategy=fraction, random_state=42)

        train_kpis, train_labels = random_undersampler.fit_resample(train_kpis, train_labels)

        # replace train data with undersampled data
        self.train_data = pd.concat([train_kpis,train_labels],axis=1)

    def pad_missing_catgorical_columns(self,approach):
        # Read feature txt file 
        with open(f'/home/papry/Exp_Project/MS_thesis-main/data/{approach}.txt', 'r') as f:
            lines = f.readlines()
            column_names_list = [line.strip() for line in lines]

        # Find the columns missing in the DataFrame
        train_missing_columns = [col for col in column_names_list if col not in self.train_kpis.columns]
        val_missing_columns = [col for col in column_names_list if col not in self.val_kpis.columns]

        # Add missing columns to the DataFrame with all zero values
        for col in train_missing_columns:
            self.train_kpis[col] = 0
        for col in val_missing_columns:
            self.val_kpis[col] = 0

        # Sort the columns in a specific order if needed
        self.train_kpis = self.train_kpis[column_names_list]
        self.val_kpis = self.val_kpis[column_names_list]

        if approach == "rural":
            self.cat_feature_number = 67
            self.feature_number = 83

    def pad_missing_catgorical_columns_test(self,approach):
        # Read feature txt file 
        with open(f'/home/papry/Exp_Project/MS_thesis-main/data/{approach}.txt', 'r') as f:
            lines = f.readlines()
            column_names_list = [line.strip() for line in lines]

        # Find the columns missing in the DataFrame
        test_missing_columns = [col for col in column_names_list if col not in self.test_kpis.columns]

        # Add missing columns to the DataFrame with all zero values
        for col in test_missing_columns:
            self.test_kpis[col] = 0

        # Sort the columns in a specific order if needed
        self.test_kpis = self.test_kpis[column_names_list]

        if approach == "rural":
            self.cat_feature_number = 67
            self.feature_number = 83
    


    @staticmethod
    def reshape_for_time_steps(kpis,labels):
        # Calculate number of previous day data used
        prev_day_data=4
        kpis = tf.reshape(kpis,(prev_day_data,-1))
        # cast to float 32
        kpis = tf.cast(kpis, dtype=tf.float32)
        labels = tf.cast(labels, dtype=tf.int32)
        labels = tf.one_hot(labels, depth=2)

        return kpis,labels
    
    def drop_cat_features(self):
        categorical_cols = self.train_data.select_dtypes(include='object').columns.tolist()
        self.train_data.drop(categorical_cols, axis=1, inplace=True)
        categorical_cols = self.val_data.select_dtypes(include='object').columns.tolist()
        self.val_data.drop(categorical_cols, axis=1, inplace=True)

    def drop_test_cat_features(self):
        categorical_cols = self.test_data.select_dtypes(include='object').columns.tolist()
        self.test_data.drop(categorical_cols, axis=1, inplace=True)
    
    @staticmethod
    def positional_encoding(kpis, labels):
        # Define a tensor of shape (5, 1)
        print("I am in positonal encoding",kpis.shape)
        positional_tensor = tf.constant(
            [
                [
                   # [-1.0],
                    #[-0.50],
                    #[-0.33],
                    [0.0],
                    #[0.2],
                   # [0.4],
                    #[0.6],
                   # [0.8],
                   # [0.5],
                    
                    [0.33],
                    [0.67],
                    [1.0]
                    
                   
                ]
                    
                
            ]
            
        )

        # Repeat the tensor along the first dimension to get a tensor of shape (3, 5, 1)
        positional_tensor = tf.tile(positional_tensor, [3, 1, 1])
#        p_kpis = tf.tile(kpis, [3, 1, 1])
        # Concatenate the expanded vector with the original tensor along dimension 2
        kpis = tf.concat([positional_tensor,kpis], axis=2)
        #kpis=tf.tile(kpis, [3, 1, 1])
        #this below line is for explainability model to fit
        kpis=tf.reshape(kpis, [3*4,78])
        return kpis, labels
    
    @staticmethod
    def reshape_weather_kpis(kpis):
        prev_day_data=4
        num_stations=3
        print("I am here ", kpis.shape)
        kpis = tf.reshape(kpis,(num_stations,prev_day_data,-1))
        print("done")
        # cast to float 32
        kpis = tf.cast(kpis, dtype=tf.float32)
        return kpis

    @staticmethod
    def reshape_weather_label(labels):
        labels = tf.cast(labels, dtype=tf.int32)
        labels = tf.one_hot(labels, depth=2)
        return labels
    
    @staticmethod
    def reshape_for_weather_stations(kpis,labels):
        kpis = DataPreprocess.reshape_weather_kpis(kpis)
        labels = DataPreprocess.reshape_weather_label(labels)
        return kpis,labels

    @staticmethod
    def reshape_for_autoencoder(kpis,labels):
        return kpis,kpis
    
    @staticmethod
    def reshape_for_autoencoder_testing(kpis,labels):
        return kpis[:,24:25],kpis[:,24:25]


    @staticmethod
    def test_reshape_for_autoencoder(kpis,labels):
        return kpis,labels


    @staticmethod
    def pad_to_batch_size(
        kpis:tf.Tensor,
        batch_size:int,
        approach:str
    ) -> tf.Tensor:
        """
        Zero Pads input tensor kpis with zeros to match first dimension to be same as batch_size.

        This function is recommended to be using during test time. Because the trained model uses
        batch size as one if it's attributes, this function pads the input kpis to match it's first
        batch
        size
        """
        if approach == "new":
            paddings = tf.constant([[0, int(batch_size-len(kpis))], [0, 0]]) #for explainabiity
            #paddings = tf.constant([[0, int(batch_size-len(kpis))], [0, 0], [0, 0], [0, 0]])
            padded_kpis = tf.pad(kpis, paddings, "CONSTANT")
        elif approach == "prev":
           # paddings = tf.constant([[0, int((batch_size-len(kpis)))], [0, 0], [0, 0]])
            paddings = tf.constant([[0, int(batch_size-len(kpis))], [0, 0]])
            padded_kpis = tf.pad(kpis, paddings, "CONSTANT")
        return padded_kpis
    
    @staticmethod
    def get_first_element(input_tensor,approach):
        """
        Get first element of input tensor
        """
        if approach == "new":
            first_element =  input_tensor[0:1,...]
        elif approach == "prev":
            first_element = input_tensor
        return first_element
    
    
    @staticmethod
    def get_k_nearest_WS(kpis:tf.Tensor,num_weather_stations:int,approach:str) -> tf.Tensor:
        """
        Get the first num_weather_stations weather stations from kpis tensor

        Arguments:
            kpis: a tensor of shape (batch_size,number of weather stations,time steps,number of features).
            num_weather_stations: an integer indicating the first k weather stations to consider from 
                the number of weather stations in kpis.
        """
        if approach == "new":
            kpis=tf.reshape(kpis, [1024,3,4,78])
            transformed_kpis = kpis[:,0:num_weather_stations,:,:]
            transformed_kpis=tf.reshape(transformed_kpis, [1024,3*4*78])
        elif approach == "prev":
            kpis=tf.reshape(kpis, [1024,4*96])
            transformed_kpis = kpis

        return transformed_kpis
    




if __name__ == '__main__':
    
    pass