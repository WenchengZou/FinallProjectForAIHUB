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
from sklearn.linear_model import Ridge
#pending,hospitalizedCurrently,inIcuCurrently,onVentilatorCurrently


def plot_coefficients(est, alpha):
    coef = est.coef_.ravel()
    plt.semilogy(np.abs(coef), marker='o', label="alpha = %s" % str(alpha))
    plt.ylim(((1e-1), 1e15))
    plt.ylabel('abs(coefficient)')
    plt.xlabel('coefficients')
    plt.legend(loc='upper left')
    plt.savefig('LassoResult.jpg')


df=pd.read_csv('../Dataset/us_daily_enhanced.csv')
#修改时间轴
df=df.loc[(df['date']>=20200315) & (df['date']<=20200420)]
usedcolumns=['date','hospitalizedIncrease','onVentilatorIncrease','negativeIncrease']
amount=1




alphas=[1e-10,1e-7,1e-5,1e-3,1e-1,1]
x=df[usedcolumns].values
y=df[['positiveIncrease']].values
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.9,random_state=20)
xtrain=np.reshape(xtrain,[-1,len(usedcolumns)])
xtest=np.reshape(xtest,[-1,len(usedcolumns)])
general_error=[]
for i,alpha in enumerate(alphas):
    mses=[]
    #do cross validation to find the best alpha
    for trains,valids in KFold(4,shuffle=True).split(range(xtrain.shape[0])):
        lreg=Ridge(alpha=alpha,normalize=True)
        lreg.fit(xtrain[trains],ytrain[trains])
        y_pred=lreg.predict(xtrain[valids])
        mses.append(mse(y_pred,ytrain[valids]))
        #??
    indexs=np.argmin(mses)
    general_error.append(mses[int(indexs)])
    #using the entire training dataset to fit the Lasso model with alpha x
indexs2=np.argmin(general_error)
#mset.append(general_error[int(indexs2)])
best_alpha=alphas[int(indexs2)]
lreg2=Ridge(alpha=best_alpha,normalize=True)
lreg2.fit(xtrain,ytrain)
y_pred2=lreg2.predict(xtest)

dict={}
for i,j in enumerate(ytest):
    dict[y_pred2[i][0]]=j[0]

print("预测值\t实际值")
for k,v in dict.items():
    print("{} {}".format(k,v))

mset=mse(y_pred2,ytest)
model=lreg2
alphasvalue=best_alpha

plot_coefficients(model,alphasvalue)
print("The column:[{}]'s mse is {}".format(usedcolumns,mset))