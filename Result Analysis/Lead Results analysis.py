#!/usr/bin/env python
# coding: utf-8

# In[257]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[258]:


df1 = pd.read_csv('Results/Lead_test (8).csv')
#alternate between the test values
df2 = pd.read_csv('Results/Lead_train (8).csv')
#alternate between the train values


# In[259]:


df1.head()


# In[260]:


df1.columns


# In[261]:


df1 = df1.rename(columns={"Unnamed: 0": "Timestamp"})


# In[262]:


df2 = df2.rename(columns={"Unnamed: 0": "Timestamp"})


# In[263]:


df1.head()


# In[264]:


df1.index  = df1['Timestamp']
df2.index  = df2['Timestamp']
df1 = df1.drop(['Timestamp'], axis=1)
df2 = df2.drop(['Timestamp'], axis=1)


# In[265]:


df1.head()


# In[266]:


df2.head()


# In[267]:


df1.columns


# In[268]:


df2.columns


# In[269]:


df2.Volatility_Lead_5_train_LSTM = df2.Volatility_Lead_5_train_LSTM.shift(5)
df2.Volatility_Lead_11_train_LSTM = df2.Volatility_Lead_11_train_LSTM.shift(11)
df2.Volatility_Lead_22_train_LSTM = df2.Volatility_Lead_22_train_LSTM.shift(22)
df2.Volatility_Lead_5_train_real_LSTM = df2.Volatility_Lead_5_train_real_LSTM.shift(5)
df2.Volatility_Lead_11_train_real_LSTM = df2.Volatility_Lead_11_train_real_LSTM.shift(11)
df2.Volatility_Lead_22_train_real_LSTM = df2.Volatility_Lead_22_train_real_LSTM.shift(22)


df1.Volatility_Lead_5_test_LSTM = df1.Volatility_Lead_5_test_LSTM.shift(5)
df1.Volatility_Lead_11_test_LSTM = df1.Volatility_Lead_11_test_LSTM.shift(11)
df1.Volatility_Lead_22_test_LSTM = df1.Volatility_Lead_22_test_LSTM.shift(22)
df1.Volatility_Lead_5_test_real_LSTM = df1.Volatility_Lead_5_test_real_LSTM.shift(5)
df1.Volatility_Lead_11_test_real_LSTM = df1.Volatility_Lead_11_test_real_LSTM.shift(11)
df1.Volatility_Lead_22_test_real_LSTM = df1.Volatility_Lead_22_test_real_LSTM.shift(22)


# In[ ]:





# In[270]:


df1.head(25)


# In[271]:


df2.tail()


# In[272]:


EPSILON = 1e-10

def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted


def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    """
    Percentage error
    Note: result is NOT multiplied by 100
    """
    return _error(actual, predicted) / (actual + EPSILON)

def mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    return np.mean(np.square(_error(actual, predicted)))

def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error """
    return np.sqrt(mse(actual, predicted))

def mae(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Absolute Error = mad """
    return np.mean(np.abs(_error(actual, predicted)))

def mape(actual: np.ndarray, predicted: np.ndarray):
    """
    Mean Absolute Percentage Error
    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when actual[t] == 0
    Note: result is NOT multiplied by 100
    """
    return np.mean(np.abs(_percentage_error(actual, predicted)))


# In[273]:


df1.Volatility_Lead_True_test.shape


# In[274]:


f = plt.figure(figsize=(14,6))
ax1 = f.add_subplot(111)
#ax2 = f.add_subplot(122)

ax1.plot(df2.Volatility_Lead_True_train, '-', label='True Lead Values', color='b')
ax1.plot(df1.Volatility_Lead_True_test, '-',  color='b')
ax1.plot(df2.Volatility_Lead_5_train_LSTM, label='LSTM Testing_5 Prediction', color='r')
ax1.plot(df1.Volatility_Lead_5_test_LSTM, label='LSTM Training_5 Prediction', color='g')
ax1.set_title("Lead LSTM Predictions")
ax1.set_xlabel('Time')
ax1.set_ylabel('Normalized Volatality')
ax1.legend()


# In[275]:


df1[['Volatility_Lead_5_test_LSTM'	,'Volatility_Lead_5_test_real_LSTM']][0:50]


# In[276]:


df1 = df1.dropna()


# In[277]:


df1 = df1[0:-5]


# In[278]:


df1.head()


# In[279]:


rmse(df1.Volatility_Lead_5_test_real_LSTM.to_numpy(),df1.Volatility_Lead_5_test_LSTM.to_numpy())


# In[280]:


rmse(df1.Volatility_Lead_11_test_real_LSTM.to_numpy(),df1.Volatility_Lead_11_test_LSTM.to_numpy())


# In[281]:


rmse(df1.Volatility_Lead_22_test_real_LSTM.to_numpy(),df1.Volatility_Lead_22_test_LSTM.to_numpy())


# In[282]:


mae(df1.Volatility_Lead_5_test_real_LSTM.to_numpy(),df1.Volatility_Lead_5_test_LSTM.to_numpy())


# In[283]:


mae(df1.Volatility_Lead_11_test_real_LSTM.to_numpy(),df1.Volatility_Lead_11_test_LSTM.to_numpy())


# In[284]:


mae(df1.Volatility_Lead_22_test_real_LSTM.to_numpy(),df1.Volatility_Lead_22_test_LSTM.to_numpy())


# In[285]:


mape(df1.Volatility_Lead_5_test_real_LSTM.to_numpy(),df1.Volatility_Lead_5_test_LSTM.to_numpy())


# In[286]:


mape(df1.Volatility_Lead_11_test_real_LSTM.to_numpy(),df1.Volatility_Lead_11_test_LSTM.to_numpy())


# In[287]:


mape(df1.Volatility_Lead_22_test_real_LSTM.to_numpy(),df1.Volatility_Lead_22_test_LSTM.to_numpy())


# In[288]:


f = plt.figure(figsize=(14,6))
ax1 = f.add_subplot(111)
#ax2 = f.add_subplot(122)

#ax1.plot(df2.Volatility_Zinc_True_train, '-', label='True Lead Values', color='b')
ax1.plot(df1.Volatility_Lead_11_test_real_LSTM, '-',  color='b')
#ax1.plot(df2.Volatility_Zinc_5_train_LSTM, label='LSTM Testing_5 Prediction', color='r')
ax1.plot(df1.Volatility_Lead_11_test_LSTM, label='LSTM Training_5 Prediction', color='g')
ax1.set_title("Lead LSTM Predictions")
ax1.set_xlabel('Time')
ax1.set_ylabel('Normalized Volatality')
ax1.legend()


# In[ ]:




