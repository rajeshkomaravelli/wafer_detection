from exception import exception
import os,sys
import shutil
from logger import logging
import pandas as pd
from data_validation import Validation
from sklearn.impute import KNNImputer
import numpy as np
import dill
from kneed import KneeLocator
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#This library uses a hueristc approch to find the elbow point it finds a point from where elbow plot decreasing slowly
class preprocessing:
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
    def sep_input_and_output_features(self,dataframe):
        """
        Description:Seperating giving dataframe into input and output features
        output:input and output dataframe
        failure:Raise Exception
        """
        try:
            X=dataframe.iloc[:,:-1]
            Y=dataframe.iloc[:,-1]
            return X,Y 
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
        Description:Using KNN imputer we are filling null values in the dataframe parameters in the KNN imputer are n_neighbors=3 weights='uniform' missing_values=nan
        output:dataframe
        failure:Raise Exception
        """
        try:
            #This imputer will fill the na values with avg of its 3 nearest neighbours
            columns=dataframe.columns
            knn_imputer=KNNImputer(n_neighbors=3,weights='uniform',missing_values=np.nan)
            nd_arrray=knn_imputer.fit_transform(dataframe)
            dataframe1=pd.DataFrame(nd_arrray,columns=columns) 
            #creating a pickle file for the imputer
            with open('data_preprocessing/imputer.pkl','wb') as file:
                dill.dump(knn_imputer,file)
            return dataframe1    
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
            
    def finding_kvalue(self,dataframe):
        """
        Description:This function  finds  the k value for kmean clusters and saves the elbow plot the dataframe produce should have input features
        output:store elbow plot and return k value for kmean
        failure:Raise exception
        """
        try:
            wcss=[]
            #We are collecting wcss value for a range of point to find elbow point this range is adjusted until a elbow point is obtained
            for i in range(1,15):
                kmean=KMeans(n_clusters=i,init='k-means++',random_state=30)
                #Creating kmean object
                kmean.fit(dataframe)
                wcss.append(kmean.inertia_)
            plt.plot(range(1,15),wcss)
            plt.xlabel("no.of centriods")
            plt.ylabel("wcss")
            plt.savefig('data_preprocessing/Elbowplot.png')
            kn = KneeLocator(range(1, 15), wcss, curve='convex', direction='decreasing')
            #Finding k value
            return kn.knee
            #kn.knee returns the k value for kmean
        except Exception as e:
            raise exception(e,sys) from e
            logging.info(exception(e,sys))  
            
    def create_cluster(self,no_of_clusters,dataframe):
        """
        Description:This function takes no_of_clusters and dataframe to be created as input and gives a list which has info about which row belongs to which cluster
        output:list
        failure: Raise Exception
        """
        try:
            kmean=KMeans(n_clusters=no_of_clusters,init='k-means++',random_state=30)      
            cluster_list=kmean.fit_predict(dataframe)
            with open("data_preprocessing/cluster.pkl",'wb') as file:
                dill.dump(kmean,file)
            return cluster_list
        except Exception as e:
            raise exception(e,sys) from e
            logging.info(exception(e,sys))    
             
            
                           
                             
    def __init__(self):
        try:
            self.input_file="data_ingestion/input.csv"
            
            logging.info("inputfile to dataframe conversion started")
            
            self.data_frame=self.read_data_frame()
            
            logging.info("input file to dataframe conversion completed")
            
            logging.info("removing Wafer column started")
            
            self.Modified_dataframe=self. deletion_of_a_columns(['Wafer','Unnamed: 0'],self.data_frame)
            
            logging.info("removing Wafer column completed")
            
            logging.info("sepearating dataframe into into input and output features")
            
            self.input_features,self.output_features=self.sep_input_and_output_features(self.Modified_dataframe)
            
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
            
            logging.info("Intiating clustering")
            logging.info("Finding k values process started")
            self.no_of_clusters=self.finding_kvalue(self.input_features)
            logging.info("Finding K value process completed")
             
            logging.info("Creating clusters") 
            self.cluster_output=self.create_cluster(self.no_of_clusters,self.input_features)
            logging.info("Creating clusters completed")
            
            self.input_features['clusters']=self.cluster_output
            
            self.input_features["output_feature"]=self.output_features
            
            self.input_features.to_csv('data_ingestion/input.csv')
            
        except Exception as e:
            raise exception(e,sys) from e
            logging.info(exception(e,sys)) 
        
        
        
        
        
        