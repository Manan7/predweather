
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
#get_ipython().magic('matplotlib inline')
from matplotlib.pylab import rcParams
rcParams ['figure.figsize']=20,6


# In[2]:


dataset = pd.read_csv("test13.csv")
dataset['date']= pd.to_datetime(dataset['date'],infer_datetime_format=True)
indexedDataset=dataset.set_index(['date'])




# In[3]:


from datetime import datetime
indexedDataset.head(297)


# In[4]:


plt.xlabel = 'date'
plt.ylabel = 'temperature'
plt.plot(indexedDataset)


# In[5]:


rolmean = indexedDataset.rolling(window=5).mean()
rolstd = indexedDataset.rolling(window=5).std()
print(rolmean,rolstd)


# In[6]:


orig=plt.plot(indexedDataset,color = 'blue',label = 'Original ')
mean = plt.plot(rolmean,color='red',label ='Rolling Mean')
std=plt.plot(rolstd,color='black',label ='Rolling std')
plt.legend(loc='best')
plt.show(block=False)


# In[7]:


from statsmodels.tsa.stattools import adfuller

print('Results of dickey fuller test')
dftest = adfuller(indexedDataset['temperature'],autolag='AIC')

dfoutput =pd.Series(dftest[0:4],index=['Test Value','pvalue','Lags used','Number of observations used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value(%s)'%key] = value
    
print (dfoutput)    


# In[8]:


indexedDataset_logScale= np.log(indexedDataset)
plt.plot(indexedDataset_logScale)


# In[11]:


movingAverage = indexedDataset_logScale.rolling(window=5).mean()
movingstd = indexedDataset_logScale.rolling(window=5).std()
plt.plot(indexedDataset_logScale)
plt.plot(movingAverage,color='red')


# In[12]:


dataset_logScaleminusmovingAverage=indexedDataset_logScale-movingAverage
dataset_logScaleminusmovingAverage.head(5)
dataset_logScaleminusmovingAverage.dropna(inplace=True)
dataset_logScaleminusmovingAverage.head(3)


# In[13]:


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    movingAVerage = timeseries.rolling(window=5).mean()
    movingStd = timeseries.rolling(window=5).std()
    
    orig=plt.plot(timeseries,color = 'blue',label = 'Original ')
    mean = plt.plot(movingAVerage,color='red',label ='Rolling Mean')
    std=plt.plot(movingStd,color='black',label ='Rolling std')
    plt.legend(loc='best')
    plt.show(block=False)
    
    
    print('Results of dickey fuller test')
    dftest = adfuller(timeseries['temperature'],autolag='AIC')

    dfoutput =pd.Series(dftest[0:4],index=['Test Value','pvalue','Lags used','Number of observations used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value(%s)'%key] = value
    print (dfoutput)
    
    


# In[14]:


test_stationarity(dataset_logScaleminusmovingAverage)


# In[15]:


exponentialDecayedWeightAverage=indexedDataset_logScale.ewm(halflife=5,min_periods=0,adjust=True).mean()
plt.plot(indexedDataset_logScale)
plt.plot(exponentialDecayedWeightAverage,color='red')


# In[16]:


ir= indexedDataset_logScale - exponentialDecayedWeightAverage
test_stationarity(ir)


# In[17]:


i6 = indexedDataset - indexedDataset.shift()
plt.plot(i6)


# In[18]:


i6.dropna(inplace=True)
test_stationarity(i6)


# In[19]:


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition =seasonal_decompose(indexedDataset_logScale.values, freq=15)

trend = decomposition.trend
seasonal= decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(indexedDataset_logScale,label ='Og')
plt.legend(loc = 'best')
plt.subplot(412)
plt.plot(trend,label ='trend')
plt.legend(loc = 'best')
plt.subplot(413)
plt.plot(seasonal,label ='seasonal')
plt.legend(loc = 'best')
plt.subplot(414)
plt.plot(residual,label ='resid')
plt.legend(loc = 'best')
plt.tight_layout()


# In[20]:


from statsmodels.tsa.stattools import acf,pacf
lag_acf=acf(i6,nlags=20)
lag_pacf=pacf(i6,nlags=20,method = 'ols')

plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(i6)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(i6)),linestyle='--',color='gray')
plt.title('Autocorrelation function')


plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(i6)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(i6)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation function')
plt.tight_layout()



# In[65]:


from statsmodels.tsa.arima_model import ARIMA
indexedDataset = indexedDataset.astype('float32')
model = ARIMA(indexedDataset,order=(2,1,2))
result_AR = model.fit(disp=-1)
plt.plot(i6)
plt.plot(result_AR.fittedvalues,color='red')
plt.title('RSS %.4f'% sum((result_AR.fittedvalues-i6['temperature'])**2))
print('Plotting AR Model')


# In[88]:


indexedDataset = indexedDataset.astype('float32')
model = ARIMA(indexedDataset,order=(2,1,2))

result_MA = model.fit(disp=-1)
plt.plot(i6)
plt.plot(result_MA.fittedvalues,color='red')
plt.title('RSS %.4f'% sum((result_MA.fittedvalues-i6['temperature'])**2))
print('Plotting AR Model')


# In[125]:


indexedDataset = indexedDataset.astype('float32')
model = ARIMA(indexedDataset,order=(3,0,3))
result_ARIMA = model.fit(disp=-1)
plt.plot(i6)
plt.plot(result_ARIMA.fittedvalues,color='red')
plt.title('RSS %.4f'% sum((result_ARIMA.fittedvalues-i6['temperature'])**2))


# In[98]:


predictions_ARIMA_diff=pd.Series(result_ARIMA.fittedvalues,copy=True)
print(predictions_ARIMA_diff.head())


# In[99]:


predictions_ARIMA_diff_cumsum=predictions_ARIMA_diff.cumsum()

print(predictions_ARIMA_diff_cumsum.head())


# In[100]:


predictions_ARIMA_log=pd.Series(indexedDataset_logScale['temperature'].ix[0],index=indexedDataset_logScale.index)
predictions_ARIMA_log=predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head(27)


# In[101]:


predictions_ARIMA=np.exp(predictions_ARIMA_log)
plt.plot(indexedDataset)
plt.plot(predictions_ARIMA)
print(predictions_ARIMA)


# In[102]:


indexedDataset



# In[128]:



# In[39]:


def getArimaResult3() :
#result_ARIMA.plot_predict(1,39)
    a_array = result_ARIMA.forecast(steps=20)
    return np.round(a_array[0])

