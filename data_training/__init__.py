from exception import exception
import os,sys
import shutil
from logger import logging
import pandas as pd 
import numpy as np
import dill
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import  roc_auc_score
from sklearn.metrics import accuracy_score
class Data_training:
    def accuracy_score1(self):
            """
            Description:This function will take model object,xtest,ytest and return accuracy_score
            output:integer
            error:raise exception
            """
            try:
                predicted_score=self.logistic_model.predict(self.xtest)
                accuracy_score_of_model=accuracy_score(self.ytest,predicted_score)
                return accuracy_score_of_model
            except Exception as e:
                raise exception(e,sys) from e
    def accuracy_score2(self):
            """
            Description:This function will take model object,xtest,ytest and return accuracy_score
            output:integer
            error:raise exception
            """
            try:
                predicted_score=self.svc_model.predict(self.xtest)
                accuracy_score_of_model=accuracy_score(self.ytest,predicted_score)
                return accuracy_score_of_model
            except Exception as e:
                raise exception(e,sys) from e  
    def accuracy_score3(self):
            """
            Description:This function will take model object,xtest,ytest and return accuracy_score
            output:integer
            error:raise exception
            """
            try:
                predicted_score=self.random_forest_model.predict(self.xtest)
                accuracy_score_of_model=accuracy_score(self.ytest,predicted_score)
                return accuracy_score_of_model
            except Exception as e:
                raise exception(e,sys) from e         
    
    def accuracy_score4(self):
            """
            Description:This function will take model object,xtest,ytest and return accuracy_score
            output:integer
            error:raise exception
            """
            try:
                predicted_score=self.xgboost_model.predict(self.xtest)
                accuracy_score_of_model=accuracy_score(self.ytest,predicted_score)
                return accuracy_score_of_model
            except Exception as e:
                raise exception(e,sys) from e         
            

    def roc_score1(self):
            """
            Description:This function will take model object,xtest,ytest and return roc_auc score
            output:integer
            error:raise exception
            """
            try:
                predicted_score= self.logistic_model.predict(self.xtest)
                roc_score=roc_auc_score(self.ytest,predicted_score)
                return roc_score
            except Exception as e:
                raise exception(e,sys) from e 
    def roc_score2(self):
            """
            Description:This function will take model object,xtest,ytest and return roc_auc score
            output:integer
            error:raise exception
            """
            try:
                predicted_score= self.svc_model.predict(self.xtest)
                roc_score=roc_auc_score(self.ytest,predicted_score)
                return roc_score
            except Exception as e:
                raise exception(e,sys) from e     
    def roc_score3(self):
            """
            Description:This function will take model object,xtest,ytest and return roc_auc score
            output:integer
            error:raise exception
            """
            try:
                predicted_score= self.random_forest_model.predict(self.xtest)
                roc_score=roc_auc_score(self.ytest,predicted_score)
                return roc_score
            except Exception as e:
                raise exception(e,sys) from e  
    def roc_score4(self):
            """
            Description:This function will take model object,xtest,ytest and return roc_auc score
            output:integer
            error:raise exception
            """
            try:
                predicted_score= self.xgboost_model.predict(self.xtest)
                roc_score=roc_auc_score(self.ytest,predicted_score)
                return roc_score
            except Exception as e:
                raise exception(e,sys) from e      
    
    def xgboost(self):
            """
            Description:This model will take x_train,x_test,y_train,y_test and return a xgboost model with best parameters
            output:xgboost model
            error:raise exception
            """
            try:
                parameter={'learning_rate':[1e-3,1e-2,1e-1,1,5],'max_depth':[1,3,5,10]}
                model=GridSearchCV(XGBClassifier(random_state=42,max_iter=100),param_grid=parameter,cv=5)
                model.fit(self.xtrain,self.ytrain_mapped)
                best_estimator=model.best_params_
                learning_rate=best_estimator['learning_rate']
                max_depth=best_estimator['max_depth']
                model1=XGBClassifier(random_state=42,max_iter=100,max_depth=max_depth,learning_rate=learning_rate)
                model1.fit(self.xtrain,self.ytrain_mapped)
                return model1
            except Exception as e:
                raise exception(e,sys) from e 
            
            
    def random_forest(self):
            """
            Description:This model will take x_train,x_test,y_train,y_test and return a random forest model with best parameters
            output:randomforest object
            error:raise exception
            """
            try:
                parameter={'max_depth':[5,10,15,20]}
                model=GridSearchCV(RandomForestClassifier(random_state=42,n_estimators=100),param_grid=parameter,cv=5)
                model.fit(self.xtrain,self.ytrain)
                best_estimator=model.best_params_
                max_depth=best_estimator['max_depth']
                model1=RandomForestClassifier(random_state=42,n_estimators=100,max_depth=max_depth)
                model1.fit(self.xtrain,self.ytrain)
                return model1
            except Exception as e:
                raise exception(e,sys) from e 
                
    def SVC_fun(self):
            """
            Description:This model will take x_train,x_test,y_train,y_test and return a SVC model with best parameters
            output:SVC object
            error:raise exception
            """
            try:
                parameter={'C':[1e-3,1e-2,1e-1,1,5,10,100]}
                model=GridSearchCV(SVC(random_state=42,max_iter=100),param_grid=parameter,cv=5)
                model.fit(self.xtrain,self.ytrain)
                best_estimator=model.best_params_
                C=best_estimator['C']
                model1=SVC(random_state=42,max_iter=100,C=C)
                model1.fit(self.xtrain,self.ytrain)
                return model1
            except Exception as e:
                raise exception(e,sys) from e  
            
            
            
            
    def logistic_regression(self):
            """
            Description:This model will take x_train,x_test,y_train,y_test and return a logistic regression model with best parameters
            output:logistic regression object
            error:raise exception
            """
            try:
                parameter={'solver':['newton-cg'],'C':[1e-3,1e-2,1e-1,1,3,10]}
                model=GridSearchCV(LogisticRegression(random_state=42,max_iter=100),param_grid=parameter,cv=5)
                model.fit(self.xtrain,self.ytrain)
                best_estimator=model.best_params_
                solver=best_estimator['solver']
                C=best_estimator['C']
                model1=LogisticRegression(random_state=42,max_iter=100,solver=solver,C=C)
                model1.fit(self.xtrain,self.ytrain)
                return model1
            except Exception as e:
                raise exception(e,sys) from e    
                                    
            
    def model_selection(self):
            """    
            Description:This function runs every cluster into 4 classification algorithms namely (Random forest,XGboost,logistic regression,SVC) and return the best model and its pickle file
            output:modelname and modelpickle file
            error:raise exception
            """
            try:
                no_of_cluster=self.dataframe['clusters'].unique()
                for i in no_of_cluster:
                    dataframe1=self.dataframe[self.dataframe['clusters']==i]
                    X=self.dataframe.iloc[:,:-2]
                    Y=self.dataframe.iloc[:,-1]
                    self.xtrain,self.xtest,self.ytrain,self.ytest=train_test_split(X,Y,test_size=.3,random_state=42)
                    self. logistic_model=self.logistic_regression()
                    self.svc_model=self.SVC_fun()
                    self.random_forest_model=self.random_forest()
                    self.ytrain_mapped = [1 if label == 1 else 0 for label in self.ytrain]
                    self.ytest_mapped = [1 if label == 1 else 0 for label in self.ytest]      
                    #we are converting -1 to 0 in both test and train set because xgboost classifier we are using only allows 0,1 as output for binary classification
                    self.xgboost_model=self.xgboost()
                    logistic_model_score=None
                    svc_model_score=None
                    random_forest_score=None
                    xgboost_model_score=None
                    dict1={1:"logistic",2:"SVC",3:"random_forest",4:"xgboost"}
                    max=0
                    y=0
                    model=None
                    if len(self.ytest.unique())==1:
                        logistic_model_score=self.accuracy_score1()
                        svc_model_score=self.accuracy_score2()
                        random_forest_score=self.accuracy_score3()
                        xgboost_model_score=self.accuracy_score4()
                    else:    
                        logistic_model_score=self.roc_score1()
                        svc_model_score=self.roc_score2()
                        random_forest_score=self.roc_score3()
                        xgboost_model_score=self.roc_score4()
                    
                    if max<logistic_model_score:
                        max=logistic_model_score
                        model=self.logistic_model
                        y=1
                    if max<svc_model_score:
                        max=svc_model_score
                        model=self.svc_model
                        y=2
                    if max<random_forest_score:
                        max=random_forest_score
                        model=self.random_forest_model
                        y=3
                    if max<xgboost_model_score:
                        max=xgboost_model_score
                        model=self.xgboost_model
                        y=4    
                    with open(f"data_training/{dict1[y]}_{i+1}.pkl",'wb') as file:
                        dill.dump(model,file)
            except Exception as e:
                raise exception(e,sys) from e             
    def __init__(self):
            try:
                self.dataframe=pd.read_csv('data_ingestion/input.csv')
                self.dataframe.drop(['Unnamed: 0'],axis=1,inplace=True)
                self.model_selection()
                logging.info("model selection started")
            except Exception as e:
                raise exception(e,sys) from e     
                    
