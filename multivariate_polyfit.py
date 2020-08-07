#Import libs
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

%matplotlib inline

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from numpy import linalg, zeros, ones, hstack, asarray
import itertools

#Read data from csv file
df = pd.read_csv("/Users/waldo/AI HUB/Project/FinallProject/Dataset/us_daily.csv")

#Extract the features we need and concatenate features into one dataframe
df_x1 = df['negativeIncrease']
df_x2 = df['hospitalizedIncrease']
df_y = df['positiveIncrease']
df_x = pd.concat([df_x1,df_x2],axis=1)

df_y = pd.DataFrame(data=df_y, columns=['positiveIncrease'])
print(df_x.shape)

#split the dataset
i_train, i_test = train_test_split(range((df_x.values).shape[0]),train_size=0.8)

X_train1 = df_x.negativeIncrease[i_train]
X_train2 = df_x.hospitalizedIncrease[i_train]
X_train = pd.concat([X_train1,X_train2],axis=1).values
print(X_train.shape)

y_train = df_y.positiveIncrease[i_train]
print(y_train.shape)

X_test1 = df_x.negativeIncrease[i_test]
X_test2 = df_x.hospitalizedIncrease[i_test]
X_test = pd.concat([X_test1,X_test2],axis=1).values
print(X_test.shape)

y_test = df_y.positiveIncrease[i_test]
print(y_test.shape)

#Initialize a list for R-square score
score_all = []

for d in range(1,11):
#Generate polynomial features
    poly = PolynomialFeatures(degree=d)

#Transform the x data for proper fitting (for single variable type it returns,[1,x,x**2])
    X_train_poly = poly.fit_transform(X_train)

#Transform the prediction to fit the model type
    X_test_poly = poly.fit_transform(X_test)

#Here we can remove polynomial orders we don't want. For instance I'm removing the `x` component
    X_train_poly = np.delete(X_train_poly,(1),axis=1)
    X_test_poly = np.delete(X_test_poly,(1),axis=1)

#Generate the regression object
    clf = LinearRegression()
#Preform the actual regression
    clf.fit(X_train_poly, y_train)
    
    scores = cross_val_score(clf, X_train_poly, y_train, cv=5)
    #print(clf.predict(X_train_poly).shape)
    #print(y_train.shape)
    score_all.append(scores.mean())

    print("Prediction in degree = {} ".format(d),clf.predict(X_test_poly))
    print("====================================")
print("Score = ",score_all)
