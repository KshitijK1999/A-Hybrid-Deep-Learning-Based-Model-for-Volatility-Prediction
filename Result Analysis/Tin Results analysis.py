#!/usr/bin/env python
# coding: utf-8

# In[97]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[98]:


df1 = pd.read_csv('Results/Tin_test (9).csv')
#alternate between the test values
df2 = pd.read_csv('Results/Tin_train (9).csv')
#alternate between the train values


# In[ ]:


df1.head()


# In[ ]:


df1.columns


# In[ ]:


df1 = df1.rename(columns={"Unnamed: 0": "Timestamp"})


# In[ ]:


df2 = df2.rename(columns={"Unnamed: 0": "Timestamp"})


# In[ ]:


df1.head()


# In[ ]:


df1.index  = df1['Timestamp']
df2.index  = df2['Timestamp']
df1 = df1.drop(['Timestamp'], axis=1)
df2 = df2.drop(['Timestamp'], axis=1)


# In[ ]:


df1.head()


# In[ ]:


df2.head()


# In[ ]:


df1.columns


# In[ ]:


df2.columns


# In[ ]:


df2.Volatility_Tin_5_train_LSTM = df2.Volatility_Tin_5_train_LSTM.shift(5)
df2.Volatility_Tin_11_train_LSTM = df2.Volatility_Tin_11_train_LSTM.shift(11)
df2.Volatility_Tin_22_train_LSTM = df2.Volatility_Tin_22_train_LSTM.shift(22)
df2.Volatility_Tin_5_train_real_LSTM = df2.Volatility_Tin_5_train_real_LSTM.shift(5)
df2.Volatility_Tin_11_train_real_LSTM = df2.Volatility_Tin_11_train_real_LSTM.shift(11)
df2.Volatility_Tin_22_train_real_LSTM = df2.Volatility_Tin_22_train_real_LSTM.shift(22)

df1.Volatility_Tin_5_test_LSTM = df1.Volatility_Tin_5_test_LSTM.shift(5)
df1.Volatility_Tin_11_test_LSTM = df1.Volatility_Tin_11_test_LSTM.shift(11)
df1.Volatility_Tin_22_test_LSTM = df1.Volatility_Tin_22_test_LSTM.shift(22)
df1.Volatility_Tin_5_test_real_LSTM = df1.Volatility_Tin_5_test_real_LSTM.shift(5)
df1.Volatility_Tin_11_test_real_LSTM = df1.Volatility_Tin_11_test_real_LSTM.shift(11)
df1.Volatility_Tin_22_test_real_LSTM = df1.Volatility_Tin_22_test_real_LSTM.shift(22)


# In[ ]:


df1.head(25)


# In[ ]:


df2.tail()


# In[ ]:


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


# In[ ]:


df1.Volatility_Tin_True_test.shape


# In[ ]:


f = plt.figure(figsize=(14,6))
ax1 = f.add_subplot(111)
#ax2 = f.add_subplot(122)

ax1.plot(df2.Volatility_Tin_True_train, '-', label='True Copper Values', color='b')
ax1.plot(df1.Volatility_Tin_True_test, '-',  color='b')
ax1.plot(df2.Volatility_Tin_5_train_LSTM, label='LSTM Testing_5 Prediction', color='r')
ax1.plot(df1.Volatility_Tin_5_test_LSTM, label='LSTM Training_5 Prediction', color='r')
ax1.set_title("Copper LSTM Predictions")
ax1.set_xlabel('Time')
ax1.set_ylabel('Normalized Volatality')
ax1.legend()


# In[ ]:


df1[['Volatility_Tin_5_test_LSTM','Volatility_Tin_5_test_real_LSTM']][0:50]


# In[ ]:


df1 = df1.dropna()


# In[ ]:


df1.head()


# In[ ]:


df1 = df1[0:-5]


# In[ ]:


rmse(df1.Volatility_Tin_5_test_real_LSTM.to_numpy(),df1.Volatility_Tin_5_test_LSTM.to_numpy())


# In[ ]:


rmse(df1.Volatility_Tin_11_test_real_LSTM.to_numpy(),df1.Volatility_Tin_11_test_LSTM.to_numpy())


# In[ ]:


rmse(df1.Volatility_Tin_22_test_real_LSTM.to_numpy(),df1.Volatility_Tin_22_test_LSTM.to_numpy())


# In[ ]:


mae(df1.Volatility_Tin_5_test_real_LSTM.to_numpy(),df1.Volatility_Tin_5_test_LSTM.to_numpy())


# In[ ]:


mae(df1.Volatility_Tin_11_test_real_LSTM.to_numpy(),df1.Volatility_Tin_11_test_LSTM.to_numpy())


# In[ ]:


mae(df1.Volatility_Tin_22_test_real_LSTM.to_numpy(),df1.Volatility_Tin_22_test_LSTM.to_numpy())


# In[ ]:


mape(df1.Volatility_Tin_5_test_real_LSTM.to_numpy(),df1.Volatility_Tin_5_test_LSTM.to_numpy())


# In[ ]:


mape(df1.Volatility_Tin_11_test_real_LSTM.to_numpy(),df1.Volatility_Tin_11_test_LSTM.to_numpy())


# In[ ]:


mape(df1.Volatility_Tin_22_test_real_LSTM.to_numpy(),df1.Volatility_Tin_22_test_LSTM.to_numpy())


# In[ ]:


f = plt.figure(figsize=(14,6))
ax1 = f.add_subplot(111)
#ax2 = f.add_subplot(122)

#ax1.plot(df2.Volatility_Zinc_True_train, '-', label='True Tin Values', color='b')
ax1.plot(df1.Volatility_Tin_11_test_real_LSTM, '-',  color='b')
#ax1.plot(df2.Volatility_Zinc_5_train_LSTM, label='LSTM Testing_5 Prediction', color='r')
ax1.plot(df1.Volatility_Tin_11_test_LSTM, label='LSTM Training_5 Prediction', color='g')
ax1.set_title("Tin LSTM Predictions")
ax1.set_xlabel('Time')
ax1.set_ylabel('Normalized Volatality')
ax1.legend()

