import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

import numpy as np 
import scipy as sp



df = pd.read_csv('us_daily_enhanced.csv')
df = df.loc[0:127]
#['','']
df_ = {}

def time_windows(window_length, data_namex, data_namey, df, alpha):
    finallx = []
    finaly = []
    # this loop is for the entire dataframe:
    # the last index for the window shall be the length of the entire dataframe minus the size of the windows plus 1
    for index in range(len(df) - window_length):
        # containing  each row for the finall result
        temp = []
        # this loop is to calculate the result for the entire column_name
        for index_dn in data_namex:
            sum = 0
            # the result of each column shall be a number which we call sum
            # and now we slide the window
            for windex in range(window_length):
                sum += (df[index_dn].values)[index + windex] * (alpha ** windex)
            temp.append(sum)
        finallx.append(temp)

    for indexy in range(window_length, len(df)):
        finaly.append((df[data_namey].values)[indexy])
    return np.array(finallx), np.array(finaly)
            
print(time_windows(3,['positiveIncrease','hospitalizedIncrease'],['positiveIncrease'],df,0.8))


