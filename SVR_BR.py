from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from finam import Exporter, Market, LookupComparator,Timeframe
import datetime
import yfinance as yf
import csv
start_date = '2019-04-20'
fromdate=datetime.date(2019, 4, 20)
end_date = '2021-09-15'

#data = yf.download(ticker, interval='1mo', start=start_date, end=end_date)

#finam exporter data source
ticker = 'BR'
exporter = Exporter()
asset = exporter.lookup(name=ticker, market=Market.FUTURES)
asset_id = asset[asset['name'] == ticker].index[0]
data = exporter.download(asset_id, market=Market.FUTURES, timeframe=Timeframe.DAILY, start_date=fromdate)
data.dropna()
bars=(len(data['<DATE>']))

feature_list=[]
for i in range(len(data['<DATE>'])):
    feature_list.append([int(i)])
data['<DATE>'] = data['<DATE>'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
data['<CLOSE>'] = data['<CLOSE>'].apply(lambda x: int(x*100))
ins = pd.DataFrame(data)
ins = ins.set_index('<DATE>')
ins= ins.loc[:,[ '<CLOSE>']]#,'<OPEN>','<HIGH>','<LOW>','<VOL>']]
ins.index.names = ['Date']
ins.columns=['Close']#,'Open','High','Low','Volume']
data=ins
target_list=data['Close'].tolist()

# Create model
linear = LinearRegression().fit(feature_list, target_list)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.0001).fit(feature_list, target_list)

# Extend a number of days for forecasting the future 
last_day = len(feature_list)
for i in range(1, 366):
    feature_list.append([last_day + i])
    target_list.append(np.nan)

# Convert back to float, so, divide every element by 100
linear_pred = list(map(lambda x: float(x) / 100, linear.predict(feature_list)))
svr_rbf_pred = list(map(lambda x: float(x) / 100, svr_rbf.predict(feature_list)))
target = list(map(lambda x: float(x) / 100, target_list))

# Display the prediction at 1 day, 30 days, and 365 days after the last available data
print('==== Linear regression prediction ====')
print(' - 1 day : ', linear_pred[last_day])
print(' - 30 days : ', linear_pred[last_day + 29])
print(' - 365 days : ', linear_pred[last_day + 364])

print('==== Support vector regression  prediction ====')
print(' - 1 day : ', svr_rbf_pred[last_day])
print(' - 30 days : ', svr_rbf_pred[last_day + 29])
print(' - 365 days : ', svr_rbf_pred[last_day + 364])

# Plot data out
#plt.hold('on')
plt.figure(figsize=(8,5))
plt.plot(feature_list, target, color='black', label='Stock Price')
plt.plot(feature_list, linear_pred, color='blue', label='Linear Regressoin')
plt.plot(feature_list, svr_rbf_pred, color='red', label='Support Vector Regression RBF')
plt.xlabel('A number of days since'+start_date)
plt.ylabel( ticker+'Price (USD)')
plt.gca().set_xlim(left = 0)
plt.gca().set_xlim(right = 1000)
plt.gca().set_ylim(bottom = 0)
plt.xticks(np.arange(0, bars+356, 100))
plt.legend()
plt.show()