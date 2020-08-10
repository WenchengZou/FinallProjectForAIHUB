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
from sklearn.metrics import r2_score
#pending,hospitalizedCurrently,inIcuCurrently,onVentilatorCurrently


def plot_coefficients(est, alpha):
    coef = est.coef_.ravel()
    plt.semilogy(np.abs(coef), marker='o', label="alpha = %s" % str(alpha))
    plt.ylim(((1e-20), 1e15))
    plt.ylabel('abs(coefficient)')
    plt.xlabel('coefficients')
    plt.legend(loc='upper left')
    plt.savefig('RidgeResult.jpg')


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


df = pd.read_csv('../Dataset/us_daily_enhanced.csv')
df = df.loc[0:127]
usedcolumns=['datenum','positiveIncrease', 'hospitalizedIncrease','deathIncrease']
predicts=['positiveIncrease']
R2=[]
mseg=[]
models=[]
test_set=[]
bests_alpha=[]
for x in np.linspace(1,3,20):
    for y in range(1,10):
        resultx, resulty = time_windows(y, usedcolumns, predicts, df, x)
        alphas=[1e-10,1e-7,1e-5,1e-3,1e-1,1]
        xtrain,xtest,ytrain,ytest=train_test_split(resultx,resulty,train_size=0.9,random_state=22)
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
            general_error.append(np.mean(mses))
            #using the entire training dataset to fit the Lasso model with alpha x
        indexs2=np.argmin(general_error)
        #mset.append(general_error[int(indexs2)])
        best_alpha=alphas[int(indexs2)]
        lreg2=Ridge(alpha=best_alpha, normalize=True)
        lreg2.fit(xtrain,ytrain)
        y_pred2=lreg2.predict(xtrain)
        #record these data

        mseg.append(mse(y_pred2,ytrain))
        R2.append(r2_score(y_pred2,ytrain))
        models.append(lreg2)
        test_set.append([xtest,ytest])
        bests_alpha.append(best_alpha)


smallindex=np.argmax(R2)
bestmodel=models[int(smallindex)]
xtest=test_set[int(smallindex)][0]
ytest=test_set[int(smallindex)][1]
y_predb=bestmodel.predict(xtest)
besta=bests_alpha[int(smallindex)]

dict={}
for i,j in enumerate(ytest):
    dict[y_predb[i][0]]=j[0]

print("预测值\t实际值")
for k,v in dict.items():
    print("{} {}".format(k,v))

mset=mse(y_predb,ytest)
model=bestmodel
alphasvalue=besta
scores=r2_score(y_predb,ytest)

plot_coefficients(model,alphasvalue)
print("The column:[{}]'s mse is {}".format(usedcolumns,mset))
print("The score of this model is {}".format(scores))
