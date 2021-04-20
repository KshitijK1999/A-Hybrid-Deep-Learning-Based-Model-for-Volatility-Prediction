#!/usr/bin/env python
# coding: utf-8

# In[323]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[324]:


df1 = pd.read_csv('Results/Nickel_test (8).csv')
#alternate between the test values
df2 = pd.read_csv('Results/Nickel_train (8).csv')
#alternate between the train values


# In[325]:


df1.head()


# In[326]:


df1.columns


# In[327]:


df1 = df1.rename(columns={"Unnamed: 0": "Timestamp"})


# In[328]:


df2 = df2.rename(columns={"Unnamed: 0": "Timestamp"})


# In[329]:


df1.head()


# In[330]:


df1.index  = df1['Timestamp']
df2.index  = df2['Timestamp']
df1 = df1.drop(['Timestamp'], axis=1)
df2 = df2.drop(['Timestamp'], axis=1)


# In[331]:


df1.head()


# In[332]:


df2.head()


# In[333]:


df1.columns


# In[334]:


df2.columns


# In[335]:


df2.Volatility_Nickel_5_train_LSTM = df2.Volatility_Nickel_5_train_LSTM.shift(5)
df2.Volatility_Nickel_11_train_LSTM = df2.Volatility_Nickel_11_train_LSTM.shift(11)
df2.Volatility_Nickel_22_train_LSTM = df2.Volatility_Nickel_22_train_LSTM.shift(22)
df2.Volatility_Nickel_5_train_real_LSTM = df2.Volatility_Nickel_5_train_real_LSTM.shift(5)
df2.Volatility_Nickel_11_train_real_LSTM = df2.Volatility_Nickel_11_train_real_LSTM.shift(11)
df2.Volatility_Nickel_22_train_real_LSTM = df2.Volatility_Nickel_22_train_real_LSTM.shift(22)


df1.Volatility_Nickel_5_test_LSTM = df1.Volatility_Nickel_5_test_LSTM.shift(5)
df1.Volatility_Nickel_11_test_LSTM = df1.Volatility_Nickel_11_test_LSTM.shift(11)
df1.Volatility_Nickel_22_test_LSTM = df1.Volatility_Nickel_22_test_LSTM.shift(22)
df1.Volatility_Nickel_5_test_real_LSTM = df1.Volatility_Nickel_5_test_real_LSTM.shift(5)
df1.Volatility_Nickel_11_test_real_LSTM = df1.Volatility_Nickel_11_test_real_LSTM.shift(11)
df1.Volatility_Nickel_22_test_real_LSTM = df1.Volatility_Nickel_22_test_real_LSTM.shift(22)


# In[336]:


df1.head(25)


# In[337]:


df2.tail()


# In[338]:


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


# In[339]:


df1.Volatility_Nickel_True_test.shape


# In[340]:


f = plt.figure(figsize=(14,6))
ax1 = f.add_subplot(111)
#ax2 = f.add_subplot(122)

ax1.plot(df2.Volatility_Nickel_True_train, '-', label='True Copper Values', color='b')
ax1.plot(df1.Volatility_Nickel_True_test, '-',  color='b')
ax1.plot(df2.Volatility_Nickel_5_train_LSTM, label='LSTM Testing_5 Prediction', color='r')
ax1.plot(df1.Volatility_Nickel_5_test_LSTM, label='LSTM Training_5 Prediction', color='g')
ax1.set_title("Copper LSTM Predictions")
ax1.set_xlabel('Time')
ax1.set_ylabel('Normalized Volatality')
ax1.legend()


# In[341]:


df1[['Volatility_Nickel_5_test_LSTM'	,'Volatility_Nickel_5_test_real_LSTM']][0:50]


# In[342]:


df1 = df1.dropna()


# In[343]:


df1.head()


# In[344]:


df1 = df1[0:-5]


# In[345]:


rmse(df1.Volatility_Nickel_5_test_real_LSTM.to_numpy(),df1.Volatility_Nickel_5_test_LSTM.to_numpy())


# In[346]:


rmse(df1.Volatility_Nickel_11_test_real_LSTM.to_numpy(),df1.Volatility_Nickel_11_test_LSTM.to_numpy())


# In[347]:


rmse(df1.Volatility_Nickel_22_test_real_LSTM.to_numpy(),df1.Volatility_Nickel_22_test_LSTM.to_numpy())


# In[348]:


mae(df1.Volatility_Nickel_5_test_real_LSTM.to_numpy(),df1.Volatility_Nickel_5_test_LSTM.to_numpy())


# In[349]:


mae(df1.Volatility_Nickel_11_test_real_LSTM.to_numpy(),df1.Volatility_Nickel_11_test_LSTM.to_numpy())


# In[350]:


mae(df1.Volatility_Nickel_22_test_real_LSTM.to_numpy(),df1.Volatility_Nickel_22_test_LSTM.to_numpy())


# In[351]:


mape(df1.Volatility_Nickel_5_test_real_LSTM.to_numpy(),df1.Volatility_Nickel_5_test_LSTM.to_numpy())


# In[352]:


mape(df1.Volatility_Nickel_11_test_real_LSTM.to_numpy(),df1.Volatility_Nickel_11_test_LSTM.to_numpy())


# In[353]:


mape(df1.Volatility_Nickel_22_test_real_LSTM.to_numpy(),df1.Volatility_Nickel_22_test_LSTM.to_numpy())


# In[354]:


f = plt.figure(figsize=(14,6))
ax1 = f.add_subplot(111)
#ax2 = f.add_subplot(122)

#ax1.plot(df2.Volatility_Zinc_True_train, '-', label='True Nickel Values', color='b')
ax1.plot(df1.Volatility_Nickel_11_test_real_LSTM, '-',  color='b')
#ax1.plot(df2.Volatility_Zinc_5_train_LSTM, label='LSTM Testing_5 Prediction', color='r')
ax1.plot(df1.Volatility_Nickel_11_test_LSTM, label='LSTM Training_5 Prediction', color='g')
ax1.set_title("Nickel LSTM Predictions")
ax1.set_xlabel('Time')
ax1.set_ylabel('Normalized Volatality')
ax1.legend()

