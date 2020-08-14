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
#df=df.loc[df['state']=='AR']
times=[]
#datetimecount=[]
#for i in df.values:
 #   times.append(pd.to_datetime(str(datetime.datetime(int(str(i[0])[:4]), int(str(i[0])[4:6]), int(str(i[0])[6:])))))
#for i in times:
 #   datetimecount.append((i-firstime).days+1)
#print(datetimecount)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
register_matplotlib_converters()
plt.figure(figsize=(16,7))#左边的参数表示宽，右边的参数表示高
time=[]
for i in df[['date','deathIncrease','positiveIncrease']].values:
    if(i[1]!=0):
        time.append(int(i[1])/int(i[2]))
        times.append(pd.to_datetime(str(datetime.datetime(int(str(i[0])[:4]), int(str(i[0])[4:6]), int(str(i[0])[6:])))))

plt.scatter(times,time,color='red')
plt.xlabel('day')
plt.ylabel('rate')
plt.savefig('en14.jpg')
