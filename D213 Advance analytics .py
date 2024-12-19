#!/usr/bin/env python
# coding: utf-8

# In[132]:


import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# In[133]:


med_data = pd.read_csv('medical_time_series .csv')


# # Part III: Data Preparation

# In[134]:


#C1.


# In[135]:


med_data


# In[136]:


med_data.shape


# In[137]:


med_data.info


# In[138]:


med_data.describe()


# In[139]:


#drop any null columns
med_data =med_data.dropna()


# In[140]:


med_data.isnull().any() # checking for missing  value 


# In[141]:


med_data.isna().sum()


# In[142]:


#C1.Provide a line graph visualizing the realization of the time series.
import matplotlib.pyplot as plt
plt.figure(figsize = (10,5))
plt.plot(med_data.Revenue)
plt.title('Revenue chart')
plt.xlabel('Day')
plt.ylabel('Revenue in million dollars')
plt.grid(True)
plt.show()


# 
# By looking at the graph above, we can see that the data is not stationary.The graph fluctuates, moving up and down 
# with a general upward trend. Overall, there is no stationarity, as no consistent trend or seasonality can be detected.

# In[143]:


#C2. Describe the time step formatting of the realization

med_data['Date']=(pd.date_range(start = datetime(2019,1,1),
        periods = med_data.shape[0], freq ='24H'))
med_data.set_index('Date',inplace = True)
med_data
                              


# In[144]:


#C3.Evaluate the stationarity of the time series.
from statsmodels.tsa.stattools import adfuller

#  ADF test
result = adfuller(med_data['Revenue'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print("Critical Value:",result[4])


if result[1] < 0.05:
    print("The time series is stationary (reject the null hypothesis).")
else:
    print("The time series is non-stationary (fail to reject the null hypothesis).")


# In[145]:


# inorder to change data to stationaty i will be using deffrencing tecknique. 


first_diff = med_data.Revenue - med_data.Revenue.shift(1)


first_diff_clean = first_diff.dropna(inplace=False)

print(first_diff_clean)


# In[146]:


#first_diff_clean

#ADF test after diffrencing data
adf_result = adfuller(first_diff_clean)
print('ADF Statistic (for Differenced data):', adf_result[0])
print('p-value Differenced data):', adf_result[1]) 
print("Critical Values Differenced data):", adf_result[4])
# Check if the  differenced data is stationary 
if adf_result[1] < 0.05:
    print("The  differenced time series is stationary (reject the null hypothesis).")
else: 
    print("The  differenced time series is non-stationary.")


# In[147]:


#plotting the data after diffrencing
plt.figure(figsize=(10, 4))
plt.plot(first_diff_clean, label='Differenced Data')
plt.title(' Differenced Data')
plt.legend()
plt.show()


# The plot above shows to be stationary.the fluctuations are centered around a constant mean  with no clear upward or downward trend. This also aligns with the AtDF test results.

# In[148]:


# time formatting after the data is diffrencened. 
med_data_diff = pd.DataFrame({
    'Day': range(1, len(first_diff_clean) + 1),  
    'Revenue': first_diff_clean
})
med_data_diff 


# In[149]:


#C4. Explain the steps you used to prepare the data for analysis, including the training and test set split.
train = med_data.iloc[:-30]
test = med_data.iloc[-30:]
print(train.shape, test.shape)


# In[150]:


#C5. Provide a copy of the cleaned data set.
train.to_csv('trained data')
test.to_csv('test data')
med_data_diff.to_csv('diffrenced_data')


# # Part IV: Model Identification and Analysis

# In[151]:


#D1.Report the annotated findings with visualizations of your data analysis, including the following elements


# In[152]:


import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import numpy as np


# In[153]:


#plot seasonality 


# In[154]:


result = seasonal_decompose(med_data_diff['Revenue'], model='additive', period=12)


# In[155]:


seasonal = result.seasonal 
# Plot the seasonal component 
plt.figure(figsize=(12, 6)) 
plt.plot(seasonal.index, seasonal, label='Seasonality') 
plt.title('Seasonal Component of Differenced Revenue') 
plt.xlabel('Date') 
plt.ylabel('Seasonality') 
plt.legend() 
plt.show()


# This plot illustrates the seasonal component of a differenced time series.The recurring oscillations indicate the presence of seasonal patterns in the data. However, the relatively small amplitude of these fluctuations suggests that, while seasonality is present, its influence on the differenced revenue appears to be minimal.

# In[156]:


# Plot trend component of the data
trend=result.trend
plt.title('Trend')
trend.plot()


# looking at the plot above is nor clear trend shown.

# In[157]:


plt.figure(figsize=(10, 4))
plot_acf(med_data_diff['Revenue'], lags=30)
plt.title('Autocorrelation Function (ACF)')
plt.show()




# The ACF plot shows a very high correlation at lag 1. the values in the vlue shardid areas are not scintifically significant.This pattern suggests the data could have an AR component, meaning the values are correlated with their past values, and the time series might follow an ARIMA  model. 

# In[68]:


#partial auto corrrelation-PACF
from statsmodels.graphics.tsaplots import plot_pacf
plt.figure(figsize=(10, 4))
plot_pacf(med_data_diff['Revenue'], lags=30)
plt.title('partial Autocorrelation Function (PACF)')
plt.show()


# The ACF plot shows a gradual decline in correlation after the first lag.
# The PACF plot displays a significant spike at the first lag and smaller insignificant spikes afterward, suggesting that an AR(1) model. 
# 

# In[158]:


plt.figure(figsize=(10, 4))
plt.psd(med_data_diff['Revenue'], NFFT=256, Fs=1)  
plt.title('Spectral Density of Differenced Time Series')
plt.show()


# The plot indicates significant peaks at frequencies corresponding to annual cycles.This suggests that the data exhibits strong seasonality on an annual basis.

# In[73]:


# Plot the decomposed components 
result.plot() 
plt.suptitle('Seasonal Decomposition of Revenue ', fontsize=16)
plt.show()


# In[159]:


# Extract the residuals
residuals = result.resid.dropna() 


# In[160]:


# Step 2: Confirm Lack of Trends in Residuals 
# Plot the residuals
plt.figure(figsize=(12, 6))
plt.plot(residuals.index, residuals, label='Residuals')
plt.title('Residuals of Decomposed Series') 
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.legend() 
plt.show()


# The data show no clear trend and it  fluctuate randomly around zero. 
# This suggests that the model has effectively removed underlying trends and seasonal patterns, leaving behind what appears to be random noise. 
# This indicates that the model is capturing the significant patterns in the data well.

# In[ ]:


#D2.Identify an autoregressive integrated moving average (ARIMA) model that accounts for the observed trend and seasonality of the time series data.


# In[78]:


pip install pmdarima


# In[161]:


import warnings
warnings.filterwarnings('ignore')


# In[162]:


import pandas as pd 
from pmdarima import auto_arima 
import matplotlib.pyplot as plt


# In[163]:


#D2 using auto ARIMA on the oriiginal dataset to find best model with the lowest AIC 
model=auto_arima(med_data['Revenue'],start_p=0,d=1,start_q=0,
          max_p=5,max_d=5,max_q=5, start_P=0,
          D=1, start_Q=0, max_P=5,max_D=5,
          max_Q=5, m=12, seasonal=True,
          error_action='warn',trace=True,
          supress_warnings=True,stepwise=True,
          random_state=493,n_fits=50) 


# In[82]:


print(model.summary())


# In[ ]:





# In[95]:


#D2 bulind  model on the trained dataset
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(train['Revenue'], order=(1,1,0),seasonal_order=(5, 1, 0, 12))
results = model.fit()
results.summary()


# In[100]:


residuals = results.resid


# In[101]:


mae = np.mean(np.abs(residuals))
print(mae)


# In[ ]:





# In[108]:


results.plot_diagnostics().show()


# In[109]:


# D3. forecast
results.forecast(31)


# In[110]:


start = len(train)
end = len(train) + len(test) - 1


# In[111]:


# Generate predictions with confidence intervals 
predictions = results.get_prediction(start=start, end=end) 
pred = predictions.predicted_mean
confidence_intervals = predictions.conf_int()


# In[112]:


pred.index = med_data.index[start:end+1] 
pred.index
confidence_intervals.index = med_data.index[start:end+1]


# In[113]:


lower_limits = confidence_intervals.iloc[:, 0] 
upper_limits = confidence_intervals.iloc[:, 1]


# In[114]:


# Plot the train, test, predictions, and confidence intervals 
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Revenue'], label='Train Data')
plt.plot(test.index, test['Revenue'], label='Test Data') 
plt.plot(pred.index, pred, color='r', label='Predictions')
plt.fill_between(pred.index, lower_limits, upper_limits, color='pink', alpha=0.3, label='Confidence Interval')
plt.legend()
plt.title('ARIMA Model Predictions with Confidence Intervals') 
plt.show()


# In[ ]:





# In[131]:


# Forecast  on the  orginal dataset
forecast_set =30
forecast= results.get_forecast(forecast_set)
mean_forecast = forecast.predicted_mean
mean_forecast 


# In[128]:


# Getting confidence intervals 
confidence_intervals = forecast.conf_int() 
lower_limits = confidence_intervals.iloc[:, 0] 
upper_limits = confidence_intervals.iloc[:, 1]


# In[129]:


last_date = med_data.index[-1] 
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),periods=forecast_set)
future_dates


# In[130]:


# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(med_data.index, med_data['Revenue'], label='Observed Data') 
plt.plot(future_dates, mean_forecast, color='r', label='Forecast') 
plt.fill_between(future_dates, lower_limits, upper_limits, color='pink', alpha=0.3, label='Confidence Interval') 
plt.legend() 
plt.title(' Forecast on Original Data')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




