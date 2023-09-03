from exception import exception
import os,sys
import shutil
from logger import logging
import pandas as pd
from predict_data_validation import Pre_Validation
from sklearn.impute import KNNImputer
import numpy as np
import dill
import matplotlib.pyplot as plt
import pickle
class predict_preprocessing:
    def read_data_frame(self):
        """
        Description: This function take self.input_file as path and return a pandas datafram
        output:dataframe
        failure:Raise exception
        """
        try:
            
            data_frame=pd.read_csv(self.input_file)
            
            return data_frame
        
        except Exception as e:
            
            raise exception(e,sys) from e
            
            logging.info(exception(e,sys))
    def deletion_of_a_columns(self,columns,dataframe):
        """
        Description:Takes columns and dataframe as input and removes columns from dataframe and returns the output dataframe
        output:Dataframe
        failure:Raise Exception
        """
        try:
            
            data=dataframe.drop(columns,axis=1)
            
            return data
        
        except Exception as e:
            
            raise exception(e,sys) from e
            
            logging.info(exception(e,sys)) 
    def null_check(self,dataframe)->bool:
        """
        Description:Checks if there are any na values in the dataframe and return 1 if there are
        output:0 or 1
        failure:Raise exception
        """         
        try:
            null_check=0
            
            list1=[]
            
            list2=[]
            
            for i in dataframe:
                
                list1.append(i)
                
                list2.append(dataframe[i].isna().sum())
                
                if dataframe[i].isna().sum()>0:
                    
                    null_check=1
            
            null_checker=pd.DataFrame({
                'column_names':list1,
                'no_of_null_values':list2
            })
            
            null_checker.to_csv('data_preprocessing/null_check.csv')
            
            return null_check      
                    
        except Exception as e:
            raise exception(e,sys) from e
            logging.info(exception(e,sys)) 
    
    def fill_null_values(self,dataframe):
        """
        Description:This function uses the imputer object used on train data to transform the prediction dataframe
        output:dataframe
        failure:Raise Exception
        """
        try:
            #This imputer will fill the na values with avg of its 3 nearest neighbours
            columns=dataframe.columns
            
            path1="C:\Users\rajes\wafer_detection\data_preprocessing\imputer.pkl"
            
            imputer_model=pickle.load(open(path1,'rb'))
            
            nd_array1=imputer_model.transform(dataframe)
            
            dataframe2=pd.Dataframe(nd_array1,columns=columns)
            
            return dataframe2
        except Exception as e:
            
            raise exception(e,sys) from e
            
            logging.info(exception(e,sys))
            
    def columns_with_zero_std_deviation(self,dataframe)->list:
        """
        Description:Checking if any column in the dataframe have zero standard deviation and returning column names in the form of list
        output:list
        failure:Raise exception
        """
        try:
            
            list1=[]
            
            std_values=dataframe.describe().iloc[3,:]
            
            std_values=dict(std_values)
            
            for i in std_values:
                
                if std_values[i]==0.0:
                    
                    list1.append(i)
            
            return list1
        
        except Exception as e:
            
            raise exception(e,sys) from e
            
            logging.info(exception(e,sys))    
    def cluster_fitting(self,dataframe)->list:
        """
        Description:predicting the cluster number for each datapoint in the dataframe and returning that data in the form of a list
        output:list
        failure:Raise Exception
        """
        try:
            path1="C:\Users\rajes\wafer_detection\data_preprocessing\cluster.pkl"
            
            cluster_model=pickle.load(path1)
            
            list1=cluster_model.predict(dataframe)
            
            return list1
        
        except Exception as e:
            
            raise exception(e,sys) from e
            
            logging.info(exception(e,sys))        
            
    def value_prediction(self,dataframe)->None:
        """
        Description:This function will take the dataframe containing prediction input or test dataframe containing cluster number as column according to the the cluster number fetches the model respective cluster model and predicts the value and returns a file containing wafer number and prediction output
        output:.txt file
        failure:Raise Exception
        """                
        try:
            cluster_numbers=dataframe['clusters'].unique()
            
            for i in cluster_numbers:
                
                dataframe1=dataframe[dataframe['clusters']==i]
                
                wafer_names=dataframe1['Wafer']
                
                dataframe1.drop(['Wafer','clusters'],axis=1,inplace=True)
                
                model_name=pickle.load(open(f'C:\Users\rajes\wafer_detection\data_training\xgboost_{i+1}.pkl','rb'))
                
                predicted_values=model_name.predict(dataframe1)
                
                list1=[]
                
                for i in wafer_names:
                
                    list1.append([i])
                
                for i in range(len(predicted_values)):
                
                    list1[i].append(predict_values[i])    
                
                with open('predictions.txt','a') as file:
                
                    file.write(str(list1))
        
        except Exception as e:
            
            raise exception(e,sys) from e
            
            logging.info(exception(e,sys))         
    def __init__(self):
        try:
            logging.info("predict processing and input value prediction started")
            
            self.input_file="predict_data_ingestion/input.csv"
            
            logging.info("inputfile to dataframe conversion started")
            
            self.data_frame=self.read_data_frame()
            
            logging.info("input file to dataframe conversion completed")
            
            logging.info("removing Wafer column started")
            
            self.wafer=data_frame['Wafer']
            
            self.Modified_dataframe=self. deletion_of_a_columns(['Wafer','Unnamed: 0'],self.data_frame)
            
            logging.info("removing Wafer column completed")
            
            logging.info("sepearating dataframe into into input and output features")
            
            self.input_features=self.Modified_dataframe
            
            logging.info("sepearating input and output features completed")
            
            null_checker1=self.null_check(self.input_features)
            if null_checker1==1 :
                logging.info("na value filling started in input_features")
                self.input_features=self.fill_null_values(self.input_features)
                logging.info("na value filling completed  in input_features")
            logging.info("checking if any columns have zero std  ")    
            
            columns_with_zero_deviation=self.columns_with_zero_std_deviation(self.input_features)    
            
            logging.info("checking std deviation is zero completed")
            
            logging.info("deleting columns with zero deviation")
            
            self.input_features=self. deletion_of_a_columns( columns_with_zero_deviation,self.input_features)
            
            logging.info("deleting columns with zero deviation Completed")
            
            logging.info("checking the cluster number of each datapoint in the dataset")
            
            self.cluster_output=self.cluster_fitting(self.input_features)
            
            logging.info("cluster number for each datapoint is found")
            
            self.input_features['clusters']=self.cluster_output
            
            self.input_features['Wafer']=self.wafer
            
            logging.info("input value prediction started")
            
            self.value_prediction(self.input_features)
            
            logging.info("input value prediction completed")
            
            logging.info("predict processing and input value prediction  completed")
            
            logging.info("prediction.txt file created")
        except Exception as e:
            raise exception(e,sys) from e
            logging.info(exception(e,sys)) 
        
        
        
        
        
        