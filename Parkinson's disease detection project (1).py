#!/usr/bin/env python
# coding: utf-8

#importing libraries
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#loading the dataset
df=pd.read_csv("C:\\Users\\soumi\\.ipython\\archive (4).zip")

print(df)

#selecting the top 5 columns
df.head()

df.describe()

df.info()

df.shape

#data cleaning
df.isnull().sum()

df['status'].value_counts()

df.groupby('status').mean()

X=df.drop(columns=['name','status'],axis=1)
Y=df['status']

print(Y)
print(X)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)

s=StandardScaler()
s.fit(X_train)

X_train=s.transform(X_train)
X_test=s.transform(X_test)

print(X_train)

print(X_test)

model=svm.SVC(kernel='linear')

#modelling the training dataset
model.fit(X_train,Y_train)

#printing the result
X_train_pred=model.predict(X_train)
train_data_accurac=accuracy_score(Y_train,X_train_pred)

print(train_data_accurac)

#modeling the test dataset and result
X_test_pred=model.predict(X_test)
test_data_accurac=accuracy_score(Y_test,X_test_pred)
print(test_data_accurac)

#building the model 
input_data = (138.19,203.522,83.34,0.00704,0.00005,0.00406,0.00398,0.01218,0.04479,0.441,0.02587,0.02567,0.0322,0.07761,0.01968,18.305,0.538016,0.74148,-5.418787,0.160267,2.090438,0.229892)
input_data_np = np.asarray(input_data)
input_data_re = input_data_np.reshape(1,-1)
s_data = s.transform(input_data_re)
pred = model.predict(s_data)
print(pred)

if(pred[0]==0):
    print("the result is neagtive, no parkinsons disease is found")
else:
    print("the result is positive,parkinson disease is detected")


