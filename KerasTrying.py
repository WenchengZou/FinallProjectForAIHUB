import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers
from keras import models
from keras.layers import Dense
from keras.callbacks import Callback

tf.keras.backend.clear_session()  # For easy reset of notebook state.

print(tf.__version__)  # You should see a >2.1.0 here!
print(tf.keras.__version__)

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

def model_structure(df_x,df_y):
    model=models.Sequential()
    model.add(Dense(3,activation='relu',input_shape=(len(df_x),)))
    model.add(Dense(4,activation='relu'))
    model.add(Dense(2,activation='relu'))
    #model.add(Dense(2,activation='relu'))
    model.add(Dense(len(df_y)))
    return model


#def noramlization(data):
#    minVals = data.min(0)
#    maxVals = data.max(0)
#    ranges = maxVals - minVals
#    normData = (data - minVals)/ranges
#    return normData


df = pd.read_csv('../Dataset/us_daily_enhanced.csv')
df=df.loc[35:96]
usedcolumns=['datenum','positiveIncrease', 'hospitalizedIncrease','deathIncrease']
predicts=['positiveIncrease']
resultx, resulty = time_windows(3, usedcolumns, predicts, df, 1.2)
xtrain,xtest,ytrain,ytest=train_test_split(resultx,resulty,train_size=0.9,random_state=22)
#ytrain=noramlization(ytrain)
#input_x=resultx[0]
#input_x=np.array([input_x])

model=model_structure(usedcolumns,predicts)
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(xtrain,ytrain,epochs=1000,batch_size=13)

y_pred=model.predict(xtest)
dict={}
for i,j in enumerate(ytest):
    dict[y_pred[i][0]]=[j[0],df.loc[df['positiveIncrease']==j[0]]['date'].values]

print("预测值\t实际值\t日期")
for k,v in dict.items():
    print("{} {} {}".format(round(k,2),v[0],v[1][0]))
"""
Fit the model setting `batch_size` to 32 and set `epochs` to something reasonable to start with (we can always change it later). 
Save the results of `fit()` in a variable called `history`. 
"""





