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


def plot_coefficients(est, alpha,axt):
    coef = est.coef_.ravel()
    axt.semilogy(np.abs(coef), marker='o', label="alpha = %s" % str(alpha))
    axt.set_ylim(((1e-20), 1e15))
    axt.set_ylabel('abs(coefficient)')
    axt.set_xlabel('coefficients')
    axt.legend(loc='upper left')
    #plt.savefig('PolyResult.jpg')


def decay_list(start,end,num):
    t = len(usedcolumns)
    n = num**t
    list = []
    for i in range(n):
        temp = []
        z = (i+1)%num if((i+1)%num != 0) else num
        y = (i)//num
        x = i%num
        temp.append((z-1)*(end-start)/(num-1)+start)
        for j in range(t-1):
            z = (y%num+1)
            y = y//num
            temp.append((z-1)*(end-start)/(num-1)+start)
        temp.reverse()
        list.append(temp)
    return list
#print(decay_list(1,3,21))

def time_windows(window_length, data_namex, data_namey, df, alpha):
    finallx = []
    finaly = []
    # this loop is for the entire dataframe:
    # the last index for the window shall be the length of the entire dataframe minus the size of the windows plus 1
    for index in range(len(df) - window_length):
        # containing  each row for the finall result
        temp = []
        # this loop is to calculate the result for the entire column_name
        for i,index_dn in enumerate(data_namex):
            sum = 0
            # the result of each column shall be a number which we call sum
            # and now we slide the window
            for windex in range(window_length):
                sum += (df[index_dn].values)[index + windex] * (alpha[i] ** windex)
            temp.append(sum)
        finallx.append(temp)
    for indexy in range(window_length, len(df)):
        finaly.append((df[data_namey].values)[indexy])
    return np.array(finallx), np.array(finaly)

df = pd.read_csv('../Dataset/us_daily_v2.csv')
df = df.loc[114:145]
usedcolumns = ['datenum', 'positiveIncrease', 'hospitalizedIncrease', 'deathIncrease']
predicts = ['positiveIncrease']
R2 = []
mseg = []
models = []
test_set = []
best_d = []
k = 0
for x in decay_list(2.5,3.5,5):
    for y in range(1, 10):
        resultx, resulty = time_windows(y, usedcolumns, predicts, df, x)
        ds = [1, 2, 3]
        xtrain, xtest, ytrain, ytest = train_test_split(resultx, resulty, train_size=0.9, random_state=22)
        xtrain = np.reshape(xtrain, [-1, len(usedcolumns)])
        xtest = np.reshape(xtest, [-1, len(usedcolumns)])
        general_error = []
        for i, d in enumerate(ds):
            mses = []
            # do cross validation to find the best alpha
            for trains, valids in KFold(4, shuffle=True).split(range(xtrain.shape[0])):
                poly = PolynomialFeatures(degree=d)
                X_train_poly = poly.fit_transform(xtrain[trains])
                X_valid_poly = poly.fit_transform(xtrain[valids])
                clf = LinearRegression()
                clf.fit(X_train_poly, ytrain[trains])
                y_pred = clf.predict(X_valid_poly)
                mses.append(mse(y_pred, ytrain[valids]))
            # ??
            general_error.append(np.mean(mses))
            # using the entire training dataset to fit the Lasso model with alpha x
        indexs2 = np.argmin(general_error)
        # mset.append(general_error[int(indexs2)])
        best_dnum = ds[int(indexs2)]
        poly = PolynomialFeatures(degree=best_dnum)
        X_train_poly = poly.fit_transform(xtrain)
        X_test_poly = poly.fit_transform(xtest)
        clf2 = LinearRegression()
        clf2.fit(X_train_poly, ytrain)
        y_pred2 = clf2.predict(X_train_poly)
        # record these data

        mseg.append(mse(y_pred2, ytrain))
        R2.append(r2_score(y_pred2, ytrain))
        models.append(clf2)
        test_set.append([X_test_poly, ytest])
        best_d.append(best_dnum)
    k = k + 1
    if (k % 20 == 0):
        print('this is the {}th cal'.format(k))

smallindex = np.argmax(R2)
print(smallindex,len(R2))
bestmodel = models[int(smallindex)]
print(bestmodel.coef_)
xtest = test_set[int(smallindex)][0]
#print(xtest.shape)
# print(best_dnum)
# poly = PolynomialFeatures(degree=best_dnum)
# X_test_poly = poly.fit_transform(xtest)
ytest = test_set[int(smallindex)][1]
#print(ytest.shape)
y_predb = bestmodel.predict(xtest)
bestd = best_d[int(smallindex)]


fig,ax=plt.subplots(1,2,figsize=(18,6))
datet=[]
for i,j in enumerate(ytest):
    temp=df.loc[df['positiveIncrease']==j[0]]['date'].values
    datet.append(temp)
register_matplotlib_converters()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
times=[]
for i in datet:
   times.append(pd.to_datetime(str(datetime.datetime(int(str(i[0])[:4]), int(str(i[0])[4:6]), int(str(i[0])[6:])))))
ax[1].set_title('The Test Result')
ax[1].scatter(times,y_predb,color='red',label='prediction')
ax[1].scatter(times,ytest,color='blue',label='reality')
ax[1].set_xlabel('date')
ax[1].set_ylabel('positiveIncrease')
ax[1].legend()
#plt.savefig('PolyTestResult.jpg')

dict={}
for i,j in enumerate(ytest):
    dict[y_predb[i][0]]=[j[0],df.loc[df['positiveIncrease']==j[0]]['date'].values]

print("预测值\t实际值\t日期")
for k,v in dict.items():
    print("{} {} {}".format(k,v[0],v[1][0]))

mset = mse(y_predb, ytest)
model = bestmodel
dvalue = bestd
scores = r2_score(y_predb, ytest)

plot_coefficients(model, dvalue,ax[0])
fig.savefig('PolyResult.jpg')
print("The column:[{}]'s mse is {}".format(usedcolumns, mset))
print("The score of this model is {}".format(scores))
print("The best degree of this model is {}".format(bestd))