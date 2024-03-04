#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection  import train_test_split
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")
class CSVDataCleansing():
    def __init__(self,data):       
        self.data = data
#         self.DataCleansing()
    # placing null columns data to mean for numaric data and mode for charater data.
    def DataCleansing(self): 
        column_name_list = self.data.columns
        for column_name in column_name_list: 
            print(self.data [column_name].dtypes)
            if self.data [column_name].dtypes =='object':
                self.data [column_name] = self.data [column_name].fillna(self.data [column_name].mode()[0])
            else:
                self.data [column_name] = self.data [column_name].fillna(self.data [column_name].median())
    
    def getNumericColumnList(self):
        return self.data.corr().reset_index()
    
    def displayOutlierForAllcolumns(self):
        column_list = self.getNumericColumnList()
        # self.data.corr().reset_index()
        for col_name in column_list["index"]:
            plt.boxplot(self.data[col_name])
            plt.xlabel(col_name)
            plt.ylabel('count')
            plt.show()
            
    def remove_outlier(self,excludeColumns): #data, column_list):
        column_list = self.getNumericColumnList()
        for column_name in column_list["index"]: #column_list:
            Q1 = self.data[column_name].quantile(0.25)
            Q3 = self.data[column_name].quantile(0.75)
            IQR = Q3 - Q1
            LowerOuterlier = Q1 - (IQR * 1.5) # also called lower_whisker
            UpperOuterlier = Q3 + (IQR * 1.5)  # also called upper_whisker

            median = (self.data[column_name].median())
            print(column_name,median,LowerOuterlier,UpperOuterlier)
#             self.data[column_name] = np.where(self.data[column_name]>UpperOuterlier, UpperOuterlier,np.where(self.data[column_name]<LowerOuterlier,LowerOuterlier,self.data[column_name]))
            if column_name not in excludeColumns: #("capital-gain","capital-loss"):
                self.data[column_name] = np.where(self.data[column_name]>UpperOuterlier, UpperOuterlier,np.where(self.data[column_name]<LowerOuterlier,LowerOuterlier,self.data[column_name]))
            
    def getObjectColumns(self,dataType):
        col_list=[]
        if dataType =="All":
            return self.data.columns
        else:
            for col in self.data.columns:
                if(self.data[col].dtype ==dataType):
                    col_list.append(col)
            return col_list
    def getColumns(self,exclude_cols):
        col_list=[]
        for col in self.data.columns:
            if(col != exclude_cols):
                col_list.append(col)
        return col_list
    
    def getDuplicateRecordCount(self):
        return self.data.duplicated().sum()
    
    def ReplaceDataWithNaN(self,SpecialCharacter):
        self.data.replace(SpecialCharacter, np.nan, inplace=True)
    def ApplyLabelEncoder(self,data):        
        le = LabelEncoder()
        for i in self.getObjectColumns('object'):
            data[i]=le.fit_transform(data[i])
        return data
    def getWeeklyVsUnmploymentGraph(self,data,store):
        store_data = data[data["Store"]==store]
        sns.barplot(x= 'Unemployment',y ='Weekly_Sales',data=store_data, palette='mako')
        plt.title('Unemployment vs Weekly_Sales')
        plt.show()
    def getPredictionFromLogisticRegression(self,y_pred,y_train):        
        lo_model=LogisticRegression()
        # training model
        lo_model.fit(x_train,y_train)
        y_pred=lo_model.predict(x_test)
        return y_pred
    def accuracy_score(self,y_pred,y_test):
        return accuracy_score(y_pred,y_test)
    def getConfusion_Matrix(self,y_pred,y_test):
        cm=confusion_matrix(y_pred,y_test)
        return cm
    def DisplayHeadMap(self,cm):
        sns.heatmap(cm,annot=True,fmt="G")
    def getProbability(self,x_test):
        probability=lo_model.predict_proba(x_test)
        return probability
    def Specificity(self,cm):
        # Specificity=true negatives/(true negative + false positives)
        specificity = cm[0][0]/(cm[0][0] + cm[0][1])    
        return specificity
    def getPrecision(self,cm):
        precision = cm[1][1]/(cm[1][1] + cm[0][1])    
        return precision
    def getRcall(self,cm):
        recall= cm[1][1]/(cm[1][1] + cm[1][0]) 
        return recall
    def getRcall(self,cm):
        recall= cm[1][1]/(cm[1][1] + cm[1][0]) 
        return recall
    def getF1_Score(self,cm): 
#         f1_Score= 2 * (Precision * Recall) / (Precision + Recall)
        f1_Score = 2 * (getPrecision(cm) * getRecall(cm)) / (getPrecision(cm) + getRecall(cm))
        #######python function stated below ####
#         from sklearn.metrics import f1_score
#         f1_score(y_test, y_pred)
        return f1_Score
    def getClassification_Report(self,y_test, y_pred):
        return classification_report(y_test, y_pred)
    
    def getTrain_test_split(self,x,y,test_siz,random_state):
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_siz,random_state=random_state)
        return x_train,x_test,y_train,y_test
    
    def SplitTrainTest(self,data, percent):
        split = (len(data)*percent)
        split = int(split) 
        train = data.iloc[:-split,:]
        test = data.iloc[-split:, :]    
        return train, test
    
    
    def getStorewiseForcast(self,data,store,forcastColumn):
        dropColumnList = []
        for column in data.columns:
            if column != forcastColumn:
                dropColumnList.append(column)

        store_data = data[data["Store"]==store]
        train, test = self.SplitTrainTest(data=store_data,percent=0.2)
        train = train.drop(columns=dropColumnList, axis=1)
        train = train.reset_index()
        train.columns=['ds','y']
        train['ds']=pd.to_datetime(train['ds'])

        model=Prophet(interval_width=0.95)
        model.fit(train)
        future=model.make_future_dataframe(periods=12, freq="W", include_history=True)
        forecast = model.predict(future)

        forecast_plot = model.plot(fcst=forecast,xlabel='Date', ylabel='Weekly Sales of Store:'+str(store), include_legend=True)

        forecast_components_plot = model.plot_components(forecast)
        return train,test, future, forecast

class ReadCSVFile(CSVDataCleansing):
    def __init__(self,filename):
        self.filename=filename
        self.data = pd.read_csv(self.filename)
        super().__init__(self.data)
    def getdata(self):
        # self.data = DataCleansing()
        return self.data
    
 



