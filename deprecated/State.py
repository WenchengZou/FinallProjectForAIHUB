import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
import datetime
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

#we use this algorithm to compute every combination set
#for example, we want to get every results for randomly selecting 2 numbers in a number set
#This function will give you the exact result
def combination(df_j,num,generalnum,base,finall):
    if(num<=0):
        finall.append(base)
        return base
    for i,ix in enumerate(df_j):
        temp2 = []
        for x in base:
            temp2.append(x)
        temp2.append(ix)
        df_t=df_j[i+1:]
        combination(df_t,num-1,generalnum,temp2,finall)
    return finall


df = pd.read_csv('../Dataset/us_states_daily_v2.csv')
df_p=pd.read_csv('../Dataset/us_daily_v2.csv')
#using unique to find the states num
statesn = np.unique(df['state'].values)
base=[]
finall=[]
finall=combination(statesn,2,2,base,finall)
States=[]
mses=[]
models=[]
for i in range(len(finall)):
    #because some state's information are omited in the table, so I just directly compute from the date 20200320
    df_t1=df.loc[(df['state']==finall[i][0]) & (df['date']>=20200320)]['positiveIncrease'].values
    df_t2=df.loc[(df['state']==finall[i][1]) & (df['date']>=20200320)]['positiveIncrease'].values
    df_t1l=df_t1.reshape(-1,1)
    df_t12=df_t2.reshape(-1,1)
    df_x=np.concatenate((df_t1l,df_t12),axis=1)
    df_y=(df_p['positiveIncrease'].values)[0:141].reshape(-1,1)
    #we do multi-Linear Regression to select the model who contains the least MSE value
    xtrain,xtest,ytrain,ytest=train_test_split(df_x,df_y,train_size=0.9,random_state=22)
    model=LinearRegression()
    model.fit(xtrain,ytrain)
    y_pred=model.predict(xtest)
    mse_results=mse(y_pred,ytest)
    mses.append(mse_results)
    States.append([finall[i][0],finall[i][1]])
    models.append(model)

#Here we get the model
smallindex=np.argmin(mses)
BestStates=States[int(smallindex)]
Bestmodel=models[int(smallindex)]
dates=df_p.loc[df_p['date']>=20200320]['date'].values
states_1=df.loc[(df['state']==BestStates[0])&(df['date']>=20200320)]['positiveIncrease'].values
states_2=df.loc[(df['state']==BestStates[1])&(df['date']>=20200320)]['positiveIncrease'].values
nations=(df_p['positiveIncrease'].values)[0:141]

#Let's plot the graph about this result
#Because we want the axis-X to be the time value
#So we need to change the original data taken from column['data']
#to be the datetime
register_matplotlib_converters()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
times=[]
for i in dates:
   times.append(pd.to_datetime(str(datetime.datetime(int(str(i)[:4]), int(str(i)[4:6]), int(str(i)[6:])))))
coef=Bestmodel.coef_.ravel()
fig,ax=plt.subplots(1,2,figsize=(15,8))
ax[0].set_title("The state's positiveIncrease information")
ax[0].plot(times,states_1,color='red',label="{}'s situation".format(BestStates[0]))
ax[0].plot(times,states_2,color='green',label="{}'s situation".format(BestStates[1]))
ax[0].set_xlabel('date')
ax[0].set_ylabel('positiveIncrease')
ax[0].legend()
ax[1].set_title("The nation's positiveIncrease information")
#Here we add the coefficient of this model in the label
ax[1].plot(times,nations,color='blue',label="The model is y={}{}+{}{}".format(round(coef[0],2),BestStates[0],round(coef[1],2),BestStates[1]))
ax[1].set_xlabel('date')
ax[1].set_ylabel('positiveIncrease')
ax[1].legend()
fig.savefig("State_situation.jpg")

print("The best states is {} and {}".format(BestStates[0],BestStates[1]))
