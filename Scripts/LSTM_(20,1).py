#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse

from datetime import date

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import LSTM, GRU, Conv1D, MaxPooling1D
from keras.utils.vis_utils import plot_model
from operator import add
import json
from matplotlib.backends.backend_pdf import PdfPages

data = pd.read_csv('Final.csv')

data['stdev_Copper'] = data['LR_Cop'].rolling(window=22, center=False).var()
data['stdev_Copper'] = data['stdev_Copper'].dropna()
data['Volatility_Copper'] = data['stdev_Copper'] * (252**0.5)
data['Volatility_Copper'] = data['Volatility_Copper']*100

data['stdev_Tin'] = data['LR_Tin'].rolling(window=22, center=False).var()
data['stdev_Tin'] = data['stdev_Tin'].dropna()
data['Volatility_Tin'] = data['stdev_Tin'] * (252**0.5)
data['Volatility_Tin'] = data['Volatility_Tin']*100

data['stdev_Nickel'] = data['LR_Nic'].rolling(window=22, center=False).var()
data['stdev_Nickel'] = data['stdev_Nickel'].dropna()
data['Volatility_Nickel'] = data['stdev_Nickel'] * (252**0.5)
data['Volatility_Nickel'] = data['Volatility_Nickel']*100

data['stdev_Lead'] = data['LR_Lead'].rolling(window=22, center=False).var()
data['stdev_Lead'] = data['stdev_Lead'].dropna()
data['Volatility_Lead'] = data['stdev_Lead'] * (252**0.5)
data['Volatility_Lead'] = data['Volatility_Lead']*100
data.dropna(inplace=True)
symbols = ['Volatility_Copper','Volatility_Tin','Volatility_Nickel','Volatility_Lead']
window_sizes = [5, 11, 22]


# In[2]:


data.head()


# In[7]:


x = pd.Timestamp('03-01-2017')
data[data['Timestamp']==x]


# In[13]:


def adj_r2_score(r2, n, k):
    return 1 - ((1 - r2) * ((n - 1) / (n - k - 1)))

def window_transform(time_series, window_size):
    X = []
    y = []
    for i in range(len(time_series) - window_size):
        X.append(time_series[i:i + window_size])
        y.append(time_series[i + window_size])

    return np.array(X), np.array(y)

def make_GARCH_lstm_model(window_size):
    model_lstm = Sequential()
    model_lstm.add(LSTM(20, input_shape=(window_size, 1), activation='relu', kernel_initializer='lecun_uniform', return_sequences=False))
    model_lstm.add(Dropout(0.05))
    model_lstm.add(Dense(1, activation='linear'))
    model_lstm.compile(loss='mean_squared_error', optimizer='adam')

    return model_lstm

def test_model(model, X_train, X_test, y_train, y_test):
    y_pred_test_ann = model.predict(X_test)
    y_train_pred_ann = model.predict(X_train)
    r2_train = mse(y_train, y_train_pred_ann)
    r2_test = mse(y_test, y_pred_test_ann)

    return r2_train, r2_test

def plot_all_model():
    print('plotting_model')
    ann_model = make_ann_model(3)
    cnn_model = make_cnn_model(3)
    lstm_model = make_lstm_model(3)
    gru_model = make_gru_model(3)

    plot_model(ann_model, to_file='ann_plot.png', show_shapes=True, show_layer_names=True)
    plot_model(cnn_model, to_file='cnn_plot.png', show_shapes=True, show_layer_names=True)
    plot_model(lstm_model, to_file='lstm_plot.png', show_shapes=True, show_layer_names=True)
    plot_model(gru_model, to_file='gru_plot.png', show_shapes=True, show_layer_names=True)


# In[14]:


data['Timestamp'] = pd.to_datetime(data['Timestamp'])


# In[15]:


output=pd.DataFrame()
output_train=pd.DataFrame()
for sym in symbols: 
  df = data[sym]
  df.index = data['Timestamp']
  df=pd.DataFrame(df)
  split_date = pd.Timestamp('01-01-2017')
  train = df.loc[:split_date]
  test = df.loc[split_date:]            
  sc = MinMaxScaler()
  train_sc = sc.fit_transform(train)
  test_sc = sc.transform(test)
  ann = []
  gru = []
  lstm = []
  cnn = []
  output[sym + "_" + "True_test"] = pd.Series(np.reshape(test_sc, ( -1)))
  output_train[sym + "_" + "True_train"] = pd.Series(np.reshape(train_sc, ( -1)))
  for win_sz in window_sizes:
    ann_result = []
    gru_result = []
    lstm_result = []
    cnn_result = []
    pred_ANN = []
    pred_LSTM = []
    pred_GRU = []
    pred_CNN = []
    pred_LSTM_train=[]
    X_train, y_train = window_transform(train_sc, win_sz)
    X_test, y_test = window_transform(test_sc, win_sz)
    X_tr_t = X_train.reshape(X_train.shape[0], win_sz, 1)
    X_tst_t = X_test.reshape(X_test.shape[0], win_sz, 1)
    for i in range(20):
      
      model = make_GARCH_lstm_model(win_sz)
      early_stop = EarlyStopping(monitor='loss', patience=6, verbose=1)
      history_model_GARCH_lstm = model.fit(X_tr_t, y_train, epochs=100, batch_size=32, verbose=1, shuffle=False,callbacks=[early_stop])

      train_acc, test_acc = test_model(model, X_tr_t, X_tst_t, y_train, y_test)
      y_pred_test_LSTM = model.predict(X_tst_t)
      y_pred_test_LSTM_train = model.predict(X_tr_t)
      lstm_result.append(test_acc)

      pred_LSTM.append(y_pred_test_LSTM)
      pred_LSTM_train.append(y_pred_test_LSTM_train)
      
    lstm.append([win_sz, min(lstm_result), np.mean(lstm_result), np.std(lstm_result)])
      
    plot_lstm = [0] * len(pred_LSTM[0])
    for pred in pred_LSTM:
        plot_lstm = list(map(add, plot_lstm, pred))

    for i in range(len(plot_lstm)):
        plot_lstm[i] = plot_lstm[i] / 20

    plot_lstm_train = [0] * len(pred_LSTM_train[0])
    for pred in pred_LSTM_train:
        plot_lstm_train = list(map(add, plot_lstm_train, pred))

    for i in range(len(plot_lstm_train)):
        plot_lstm_train[i] = plot_lstm_train[i] / 20
    
    output[sym + "_" + str(win_sz) + "_test_LSTM"] = pd.Series(np.reshape(plot_lstm, (-1)))
    output_train[sym + "_" + str(win_sz) + "_train_LSTM"] = pd.Series(np.reshape(plot_lstm_train, (-1)))
    output[sym + "_" + str(win_sz) + "_test_real_LSTM"] = pd.Series(np.reshape(y_test, (-1)))
    output_train[sym + "_" + str(win_sz) + "_train_real_LSTM"] = pd.Series(np.reshape(y_train, (-1)))
    f = plt.figure(figsize=(12,4))
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)

    ax1.plot(y_test, '-', label='True test Values', color='#1b9e77')
    ax1.plot(plot_lstm, label='LSTM test Prediction', color='#e7298a')
    ax1.set_title("Prediction")
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Normalized Stock Prices')
    ax1.legend()

    ax2.plot(y_train, '-', label='True Train Values', color='#1b9e77')
    ax2.plot(plot_lstm_train, label='LSTM Train Prediction', color='#e7298a')
    ax2.set_title("Prediction")
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Normalized Stock Prices')
    ax2.legend()
#output.to_excel('output.xlsx')   
#output_train.to_excel('output_train.xlsx') 


# In[16]:


df1 = output_train
#df1.index = data['Timestamp'].iloc[0:df1.shape[0]].values
df2 = output
#df2.index = data['Timestamp'].iloc[df1.shape[0]:(df1.shape[0]+df2.shape[0])].values


# In[17]:


#df1.to_csv('Çopper_train')
#df2.to_csv('Çopper_test')


# In[18]:


from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df2, title = "Download CSV file", filename = "Zinc_test.csv"):
    csv = df2.to_csv(index=True)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(df2)


# In[19]:


from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df1, title = "Download CSV file", filename = "Zinc_train.csv"):
    csv = df1.to_csv(index=True)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(df1)


# In[ ]:




