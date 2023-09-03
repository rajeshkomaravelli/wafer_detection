from exception import exception
import os,sys
import yaml
import shutil
import re
from logger import logging
import pandas as pd
import json
class Validation:
    def read_yaml_file(self,file_path:str)->dict:
        """
        Reads a YAML file and returns the contents as a dictionary.
        file_path: str
        """
        try:
            with open(file_path, 'rb') as yaml_file:
                return yaml.safe_load(yaml_file)
        except Exception as e:
                raise exception(e,sys) from e    
            
    def format_check(self,file_name)->None:
        """
        This function checks if a csv in a directory is in the format 'wafer-8numbers-6numbers' or not and 
        segregate them into good files or bad files the directory of name goodfile and badfile should be preprared before using this function 
        output :segregate the files in good and bad files
        error:Raise exception
        """
        try:
            files=[file_name for file_name in os.listdir(self.data_file_path)]
            list1=[]
            for file_name in files:
                file_name=file_name.split('.csv')[0]
                file_name=file_name.split('_')
                list1.append(file_name)
            for i in range(len(list1)):
                if len(list1[i])!=3:
                    shutil.copy(self.data_file_path+'/'+files[i],self.bad_file_path+'/'+files[i])
                    continue
                    
                test1=re.fullmatch('wafer',list1[i][0]) 
                test2=re.fullmatch(r'\d+',list1[i][1])
                test3=re.fullmatch(r'\d+',list1[i][2])
                
                if test1:
                    pass
                else:
                    shutil.copy(self.data_file_path+'/'+files[i],self.bad_file_path+'/'+files[i])
                    continue
                if test2:
                    pass
                else:
                    shutil.copy(self.data_file_path+'/'+files[i],self.bad_file_path+'/'+files[i]) 
                    continue 
                if test3:
                    pass
                else:
                    shutil.copy(self.data_file_path+'/'+files[i],self.bad_file_path+'/'+files[i])
                    continue
                shutil.copy(self.data_file_path+'/'+files[i],self.good_file_path+'/'+files[i])   
                files1=[file_name for file_name in os.listdir(self.good_file_path) ]
                list2=[]
                for file_name in files1:
                        file_name=file_name.split('.csv')[0]
                        file_name=file_name.split('_')
                        list2.append(file_name)
                for i in range(len(list2)):
                    if len(list2[i][1])==self.LengthOfDateStampInFile and len(list2[i][2])==self.LengthOfTimeStampInFile:
                        pass
                    else:
                        shutil.move(self.good_file_path+'/'+files1[i],self.bad_file_path+'/'+files1[i])
        except Exception as e:
            raise  exception(e,sys) from e                
    def check_and_convert_na_into_null(self):
        """
        THIS function will check if a csv file has a na value in one of the columns and fill it with NULL and if a whole column has na values this function will drop that column and crete a new 
        directory good_file_path1 folder where it save modified csv files 
        output:Creates a folder which have .csv files wirh no na values
        error:Raise exception
        """
        try:
            good_file_path=self.good_file_path
            self.good_file_path=self.root_dir+'/good_file_path1'
            os.makedirs(self.good_file_path,exist_ok=True)
            files2=[file for file in os.listdir(good_file_path)]
            for path in files2:
                df1=pd.read_csv(good_file_path+'/'+path)
                shape=df1.shape
                for i in df1:
                    if df1[i].isna().sum()==shape[1]:
                        df1.drop([i],axis=1,inplace=True)
                    else:
                        df1[i].fillna('NULL',inplace=True) 
                df1.to_csv(f'{self.good_file_path}/{path}', index=False,encoding='utf-8')            
            shutil.rmtree(good_file_path)          
        except Exception as e:
            raise  exception(e,sys) from e                 

    def __init__(self):
        try:
            logging.info("validaion started")
            logging.info("schema file reading started")
            json_file=None
            with open("C:/Users/rajes/wafer_detection/schema_training.json",'r') as file4:
                json_file=json.load(file4)
            self.schema_file=json_file
            logging.info("schema file reading completed")
            self.LengthOfDateStampInFile=self.schema_file['LengthOfDateStampInFile']
            self.LengthOfTimeStampInFile=self.schema_file['LengthOfTimeStampInFile']
            self.column_names=self.schema_file['ColName']
            self.number_of_columns=self.schema_file['NumberofColumns']
            self.root_dir=os.getcwd()
            self.good_file_path=self.root_dir+'/goodfiles'
            self.bad_file_path=self.root_dir+'/badfiles'
            os.makedirs(self.good_file_path,exist_ok=True)
            os.makedirs(self.bad_file_path,exist_ok=True)
            self.data_file_path=r"C:\Users\rajes\wafer_detection\Training_Batch_Files"
            logging.info("file format checking started")
            self.format_check(self.data_file_path)
            logging.info("file format checking completed")
            logging.info("na value check started")
            
            self.check_and_convert_na_into_null()
            logging.info("na value check completed")
            logging.info("validation completed successfully")
        except Exception as e:
            raise  exception(e,sys) from e
        
        
    
    