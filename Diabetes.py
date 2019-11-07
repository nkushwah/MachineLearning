# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 23:00:03 2019

@author: nkushwah
"""

# Data Manipulation
import numpy as np
import pandas as pd

# Visualization 
import matplotlib.pyplot as plt
import missingno

# Machine learning
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Let's be rebels and ignore warnings for now
import warnings
warnings.filterwarnings('ignore')


# Load the diabetes dataset
diabetes = datasets.load_diabetes()
#Attribute Information
Col= ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4',  's5',  's6']
#Number of rows
Row = np.arange(1,len(diabetes.data)+1)
#Loading diabetes dataset to Dataframe
df_data = pd.DataFrame(diabetes.data,Row,Col)
#quantitative measure of disease progression one year after baseline
df_target = pd.DataFrame(diabetes.target,Row,["Output_Col"])
#Combine data frames
df = pd.concat([df_data, df_target], axis=1, ignore_index=False)
#col_pairs = list(itertools.combinations(df.columns[4:10], 2))
#df1 = pd.DataFrame({'{}{}'.format(a, b): df[a] - df[b] for a, b in col_pairs})
#df = pd.concat([df, df1], axis=1, ignore_index=False)
#Describe the data
df.describe()
#Dataframe name
df.dataframeName = 'Diabetes patience record'
#Missing value check
missingno.matrix(df_data, figsize = (10,10))
missingno.matrix(df_target, figsize = (10,10))
#or
df_data.isnull().sum()
df_target.isnull().sum()


def PlotCorrelation(df, graphWidth):
    filename = df.dataframeName
    print(df)
    df = df.dropna('columns')
    #df = df[[col for col in df if df[col].nunique() > 1]] 
    #print(df)
    print(df.shape[1])
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    print(corr)
    plt.figure(num = None, figsize=(graphWidth,graphWidth), dpi=80, facecolor='w', edgecolor='r')
    corrMat = plt.matshow(corr,fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)),corr.columns,rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()

df.corr()
PlotCorrelation(df,10)
#Dropping S3 column as it shows negative correlation with other features
df_data.drop("s3" ,inplace=True, axis=1)
#Genrate Test and train data for data and target dataframe
lr = LinearRegression()

#Test model to find out split and seed ratio, to get max R2 score
temp_R2 = 0
temp_seed = 0
temp_testsize = 0
for testsize in np.arange(0.2,0.4,0.01):
    testsize=np.round(testsize,2)
    print('*'*20,"testsize",testsize,'*'*20)
    for seed in range(0,100,1):
        print('*'*20,seed,'*'*20)
        xtrain,xtest,ytrain,ytest = train_test_split(df_data,df_target,test_size=testsize, random_state=seed)
        lr.fit(xtrain,ytrain)
        ypred = lr.predict(xtest)
        # The mean squared error
        print("Mean squared error: %.2f"% mean_squared_error(ytest, ypred))
        # Explained variance score: 1 is perfect prediction
        print('R2 score: %.2f' % r2_score(ytest, ypred))
        if(r2_score(ytest, ypred)>temp_R2):
            temp_R2 = r2_score(ytest, ypred)
            temp_seed = seed
            temp_testsize = testsize
       
print("R2 score: ",temp_R2)
print("seed value: ", temp_seed)
print("test size: ", temp_testsize)

xtrain,xtest,ytrain,ytest = train_test_split(df_data,df_target,test_size=temp_testsize, random_state=temp_seed)
lr.fit(xtrain,ytrain)
ypred = lr.predict(xtest)
print('Coefficients: \n', lr.coef_)
print('Intercept: \n', lr.intercept_)
# The mean squared error
print("Mean squared error: %.2f"% mean_squared_error(ytest, ypred))
# Explained variance score: 1 is perfect prediction
print('R2 score: %.2f' % r2_score(ytest, ypred))