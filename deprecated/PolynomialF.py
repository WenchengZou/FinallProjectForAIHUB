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

df=pd.read_csv('../Dataset/us_daily.csv')
print([column for column in df])
times=[]
datetimecount=[]
for i in df.values:
    times.append(pd.to_datetime(str(datetime.datetime(int(str(i[0])[:4]), int(str(i[0])[4:6]), int(str(i[0])[6:])))))
firstime=times[len(times)-1]
for i in times:
    datetimecount.append((i-firstime).days+1)
x=datetimecount
y=df['positiveIncrease']
degrees=np.arange(2,10)
#divide the dataset into 90% training set and 10% testing set
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.9,random_state=20)
xtrain=np.reshape(xtrain,[-1,1])
x=np.reshape(x,[-1,1])
mses=[]
for i in degrees:
    Xtrain=PolynomialFeatures(degree=i).fit_transform(xtrain)
    Poly=LinearRegression()
    Poly.fit(Xtrain,ytrain)
    y_pred=Poly.predict(Xtrain)
    mses.append(mse(y_pred,ytrain))

bestdegree=degrees[int(np.argmin(mses))]

X=PolynomialFeatures(degree=bestdegree).fit_transform(x)
Ireg=LinearRegression()
Ireg.fit(X,y)
y_pre=Ireg.predict(X)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
register_matplotlib_converters()
plt.figure(figsize=(13,7))#左边的参数表示宽，右边的参数表示高
plt.scatter(times,y,color='blue',label='original data')
plt.plot(times,y_pre,color='red',label='predict line')
plt.xlabel('day')
plt.ylabel('PositiveIncrease')
plt.legend()
plt.savefig('en6.jpg')