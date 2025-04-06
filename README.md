# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
Developed by: priyanka R
RegisterNumber: 2122232200814
```
import pandas as pd
import numpy as np
```
dataset=pd.read_csv('Placement_Data.csv')
dataset

## output
![image](https://github.com/user-attachments/assets/14fe8da7-f734-4010-b6db-7df0f6fd9667)

dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')    
dataset["status"]=dataset["status"].astype('category') 
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

## output
![image](https://github.com/user-attachments/assets/bc099560-2d52-4d7e-86c5-29d37a900c0c)

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes   
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset

## output
![image](https://github.com/user-attachments/assets/9f04c437-48a8-4f28-a458-b6bc55d35a0b)

X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values
Y

## output
![image](https://github.com/user-attachments/assets/7cbf7698-24b7-4afa-b2aa-a7779ac76cf0)

theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y)*np.log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y) / m
        theta -= alpha * gradient
    return theta
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred = np.where(h>= 0.5,1,0)
    return y_pred
y_pred = predict(theta,X)
accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)


## output
![image](https://github.com/user-attachments/assets/1d06f632-8e8b-4c1e-985b-dc3e41ae57ff)


print(y_pred)

## output
![image](https://github.com/user-attachments/assets/52af4254-f097-4608-bed2-2aff5820279a)









## Output:
![logistic regression using gradient descent](sam.png)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

