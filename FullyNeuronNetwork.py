import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers
from keras import models
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import Callback
from tensorflow.keras import regularizers


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
    model = models.Sequential()
    myl2_reg = regularizers.l2(0.01)
    model.add(Dense(6,activation='relu',input_shape=(len(df_x),),kernel_regularizer=myl2_reg))
    model.add(layers.Dropout(0.2))
    #model.add(layers.Dropout(0.2))
    model.add(Dense(7,activation='relu',kernel_regularizer=myl2_reg))
    model.add(layers.Dropout(0.3))
    #model.add(layers.Dropout(0.2))
    #model.add(Dense(3,activation='relu'))
    #model.add(Dense(2,activation='relu'))
    model.add(Dense(len(df_y),activation='linear',kernel_regularizer=myl2_reg))
    return model


def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = (data - minVals)/ranges
    return normData


df = pd.read_csv('../Dataset/us_daily_v2.csv')
df=df.loc[0:146]
usedcolumns=['datenum','positiveIncrease', 'hospitalizedIncrease','deathIncrease']
predicts=['positiveIncrease']
resultx, resulty = time_windows(3, usedcolumns, predicts, df, 2.3)
xtrain,xtest,ytrain,ytest=train_test_split(resultx,resulty,train_size=0.9,random_state=10000)
xtrain=noramlization(xtrain)
xtest=noramlization(xtest)
#input_x=resultx[0]
#input_x=np.array([input_x])
model=model_structure(usedcolumns,predicts)
model.summary()

#sgd = optimizers.SGD(lr=0.025, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.01))
history_t=model.fit(xtrain,ytrain,validation_data=(xtest,ytest),epochs=620,batch_size=29)
plt.title("L2 Regulized model")
plt.semilogy(history_t.history['loss'], label='Train Loss')
plt.semilogy(history_t.history['val_loss'], label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.savefig('nnmse.jpg')
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