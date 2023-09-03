from exception import exception
import os,sys
import shutil
from logger import logging
import pandas as pd
from predict_data_validation import Pre_Validation
import sqlite3
import subprocess 
#This library is used to run the commands which manually should be done on command prompt by we are running them using this library
class predict_Dbinsertion:
              
    def create_and_dataiinsertion_into_table (self):
        """
        Description:This method will read all the csv files in the good_file_path and store the data in a .sql file
        output:.sql file
        error:Raise exception
        """
       
       
        try:
            files = [filename for filename in os.listdir(self.good_file_path)]
            for i in files:
                df1 = pd.read_csv(os.path.join(self.good_file_path, i))
                df1.to_sql(self.tablename, self.connector, if_exists='append', index=False)
           # Command 1: sqlite3 your_database_file.db
            process = subprocess.Popen(['sqlite3', f'{self.db_file}'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            #This command popen opening s1llite3 shell and using this shell we are connecting to db file 
            #  stdin store the input given in the sqlite shell 
            # stdout is the variable store the output of the command entered
            # stderr stores the error if occured after entering the command
            # process.communicate()runs the command in the shell and we can store stdin,stdout,stderr
            
            
            # Command 2: .output output_file.sql
            process = subprocess.Popen(['sqlite3', f'{self.db_file}', '.output',f"{self.output_sql_file}"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            #Here the command is  we are directing output we get from command prompt to a output file
            #What ever output we get stored in this file
            
            # Command 3: .schema your_table_name
            process = subprocess.Popen(['sqlite3',f'{self.db_file}', '.schema',  f" {self.tablename}"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            #This command retrive the schema of a particular table
            
            
            # Command 4: .dump your_table_name
            dump_command = f".dump {self.tablename}"
            output_file = f"{self.output_sql_file}"

            with open(output_file, 'w') as file:
                process = subprocess.Popen(['sqlite3', f'{self.db_file}'], 
                                        stdin=subprocess.PIPE, 
                                        stdout=file, 
                                        stderr=subprocess.PIPE, 
                                        text=True)
                process.communicate(input=dump_command)
                
                #We are opening  the file in write mode and executing the dump command by which output from the command is stored in the .sql file
                
                
            # Command 5: .exit
            process = subprocess.Popen(['sqlite3',  f'{self.db_file}', '.exit'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()   
            
            
            #exiting the sql shell

        except Exception as e:
            raise exception(e,sys) from e
                 
                
    def creating_input_file(self):
        """
        Description:Read a .sql file and return a .csv file
        output:.csv file
        error :Raise exception 
        """
        try:
            sql_file=self.output_sql_file
            with open(sql_file,'r') as file:
                query=file.read()
                self.connector.executescript(query)
            query1=f"select * from {self.tablename}"    
            final_data_frame=pd.read_sql(query1,self.connector)
            final_data_frame.rename(columns={'Unnamed: 0':'Wafer'},inplace=True)
            final_data_frame.to_csv("predict_data_ingestion/input.csv")
        except Exception as e:
            raise exception(e,sys) from e                    

           
            
    def __init__(self):
        try:
            self.db_file="predict_data_ingestion/training.db"
            self.connector=sqlite3.connect(self.db_file)
            validation_object=Pre_Validation()
            self.schema_file=validation_object.schema_file
            self.good_file_path=validation_object.good_file_path
            self.connector.execute("drop table if exists predict_wafertable")
            self.tablename='predict_wafertable'
            self.output_sql_file='predict_data_ingestion/'+f'{self.tablename}'+'.sql' 
            logging.info("table creation and data ingestion started")
            self.cursor=self.connector.cursor()
            self.create_and_dataiinsertion_into_table()
            logging.info("table creation and data ingestion completed")
            logging.info("creating input csv started") 
            self.creating_input_file()
            logging.info("creating input csv completed")
            logging.info("db insertion and input file creation completed")
               
        except Exception as e:
            raise exception(e,sys) from e 
                
            
        
        
        