#!/usr/bin/env python
# coding: utf-8

# In[333]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse

from datetime import date

from keras.models import Sequential,Model
from keras.layers import concatenate
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


# In[335]:


data1 = pd.read_csv('sGarch1110Copper_252roll.csv')
data2 = pd.read_csv('sGarch1110Copper_252roll.csv')
data3 = pd.read_csv('eGarch1110Copper_252roll.csv')
data4 = pd.read_csv('eGarch1110Copper_252roll.csv')


# In[340]:


data1 = data1.dropna()
data2 = data2.dropna()
data3 = data3.dropna()
data4 = data4.dropna()


data1.reset_index(inplace=True)
data2.reset_index(inplace=True)
data3.reset_index(inplace=True)
data4.reset_index(inplace=True)


data1 = data1.drop(['index'], axis=1)
data2 = data2.drop(['index'], axis=1)
data3 = data3.drop(['index'], axis=1)
data4 = data4.drop(['index'], axis=1)


data1['x'] = data1['x'].replace([0, -np.inf], data1['x'].nsmallest(2).iloc[-1])
data2['x'] = data2['x'].replace([0, -np.inf], data2['x'].nsmallest(2).iloc[-1])
data3['x'] = data3['x'].replace([0, -np.inf], data3['x'].nsmallest(2).iloc[-1])
data4['x'] = data4['x'].replace([0, -np.inf], data4['x'].nsmallest(2).iloc[-1])


data1['x'] = data1['x'].replace([np.inf], data1['x'].nlargest(2).iloc[-1])
data2['x'] = data2['x'].replace([np.inf], data2['x'].nlargest(2).iloc[-1])
data3['x'] = data3['x'].replace([np.inf], data3['x'].nlargest(2).iloc[-1])
data4['x'] = data4['x'].replace([np.inf], data4['x'].nlargest(2).iloc[-1])


# In[341]:


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

#symbols = ['Volatility_Copper','Volatility_Tin','Volatility_Nickel','Volatility_Lead']
symbols = ['Volatility_Copper']
window_sizes = [5, 11, 22]


# In[342]:


data = data[0:-1]


# In[345]:


data.dropna(inplace=True)


# In[347]:


x = list(range(22,253))


# In[348]:


data.drop(x,axis=0,inplace=True)


# In[349]:


data2.drop(0,axis=0,inplace=True)
data4.drop(0,axis=0,inplace=True)


# In[350]:


data1 = data1[0:-1]
data3 = data3[0:-1]


# In[368]:


def adj_r2_score(r2, n, k):
    return 1 - ((1 - r2) * ((n - 1) / (n - k - 1)))

def window_transform(time_series, window_size):
    X = []
    y = []
    for i in range(time_series.shape[0] - window_size):
        X.append(time_series[i:i + window_size])
        y.append(time_series[i + window_size])
        
    return np.array(X), np.array(y)

def window_transform1(time_series, window_size):
    X = []
    y = []
    for i in range(time_series.shape[0] - window_size):
        X.append(time_series[i + window_size])
        y.append(time_series[i + window_size])
        
    return np.array(X), np.array(y)
  
def make_GARCH_lstm_model(window_size):
    model_lstm = Sequential()
    model_lstm.add(LSTM(80, input_shape=(window_size, 1), activation='relu', kernel_initializer='lecun_uniform', return_sequences=False))
    model_lstm.add(Dropout(0.05))
    model_lstm.add(Flatten())

    return model_lstm

def future_lstm_model1():
    model_lstm = Sequential()
    model_lstm.add(LSTM(80, input_shape=(1, 1), activation='relu', kernel_initializer='lecun_uniform', return_sequences=False))
    model_lstm.add(Dropout(0.05))
    model_lstm.add(Flatten())

    return model_lstm

def future_lstm_model2():
    model_lstm = Sequential()
    model_lstm.add(LSTM(80, input_shape=(1, 1), activation='relu', kernel_initializer='lecun_uniform', return_sequences=False))
    model_lstm.add(Dropout(0.05))
    model_lstm.add(Flatten())

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


# In[369]:


data['Timestamp'] = pd.to_datetime(data['Timestamp'])


# In[370]:


data.head()


# In[371]:


data2.describe()


# In[372]:


data4.describe()


# In[373]:


data1.head()


# In[374]:


data3.head()


# In[375]:


output=pd.DataFrame()
output_train=pd.DataFrame()
for sym in symbols:
  df = data[sym]
  df1 = data[sym]
  df2 = data[sym]
  df.index = data['Timestamp']
  df1.index = data['Timestamp']
  df2.index = data['Timestamp']
  df=pd.DataFrame(df)
  df1=pd.DataFrame(df1)
  df2=pd.DataFrame(df2)
  #df['forecast_vol_Copper_inp1'] = ((data1['x'].values))
  #df['forecast_vol_Copper_inp2'] = ((data3['x'].values))
  df1['forecast_vol_Copper_inp1'] = ((data2['x'].values)) 
  df2['forecast_vol_Copper_inp2'] = ((data4['x'].values)) 
  split_date = pd.Timestamp('01-01-2017')
  train = df.loc[:split_date]
  test = df.loc[split_date:]
  train1 = df1.loc[:split_date]
  test1 = df1.loc[split_date:]
  train2 = df2.loc[:split_date]
  test2 = df2.loc[split_date:]
  sc = MinMaxScaler()
  train_sc = sc.fit_transform(train)
  test_sc = sc.transform(test)
  train_sc1 = sc.fit_transform(train1)
  test_sc1 = sc.transform(test1)
  train_sc2 = sc.fit_transform(train2)
  test_sc2 = sc.transform(test2)

  ann = []
  gru = []
  lstm = []
  cnn = []
  GARCH_lstm = []
  output[sym + "_" + "True_test"] = pd.Series(np.reshape(test_sc[:,0], ( -1)))
  output_train[sym + "_" + "True_train"] = pd.Series(np.reshape(train_sc[:,0], ( -1)))
  for win_sz in window_sizes:
    ann_result = []
    gru_result = []
    lstm_result = []
    cnn_result = []
    lstm_GARCH_result = []
    pred_ANN = []
    pred_LSTM = []
    pred_GRU = []
    pred_CNN = []
    pred_GARCH_LSTM = []
    pred_GARCH_LSTM_train = []
    
    for i in range(15):
                    
      X_train, y_train = window_transform(train_sc, win_sz)
      X_test, y_test = window_transform(test_sc, win_sz)
      X_tr_t = X_train.reshape(X_train.shape[0], win_sz, 1)
      X_tst_t = X_test.reshape(X_test.shape[0], win_sz, 1)
      #y_train = np.delete(y_train,1,1)
      #y_test = np.delete(y_test,1,1)
      X_tr_t = X_tr_t.reshape(X_tr_t.shape[0], win_sz, 1)
      X_tst_t = X_tst_t.reshape(X_tst_t.shape[0], win_sz, 1)
        
      X_train1, y_train1 = window_transform1(train_sc1, win_sz)
      X_test1, y_test1 = window_transform1(test_sc1, win_sz)
    
      y_train1 = np.delete(y_train1,1,1)
      y_test1 = np.delete(y_test1,1,1)
      X_train1 = np.delete(X_train1,0,1)
      X_test1 = np.delete(X_test1,0,1)
    
      X_train1 = X_train1
      y_train1 = y_train1
      X_test1 = X_test1
      y_test1 = y_test1
      X_tr_t1 = X_train1
      X_tst_t1 = X_test1
    
      X_train2, y_train2 = window_transform1(train_sc2, win_sz)
      X_test2, y_test2 = window_transform1(test_sc2, win_sz)
    
      y_train2 = np.delete(y_train2,1,1)
      y_test2 = np.delete(y_test2,1,1)
      X_train2 = np.delete(X_train2,0,1)
      X_test2 = np.delete(X_test2,0,1)
    
      X_train2 = X_train2
      y_train2 = y_train2
      X_test2 = X_test2
      y_test2 = y_test2
      X_tr_t2 = X_train2
      X_tst_t2 = X_test2
    
      model1 = make_GARCH_lstm_model(win_sz)
      model2 = future_lstm_model1()
      model3 = future_lstm_model2()
      
      merged = concatenate([model1.output,model2.output,model3.output])
      
      z = Dense(256, activation='relu')(merged)
      z = Dropout(0.05)(z)
      z = Dense(128, activation='relu')(merged)
      z = Dropout(0.05)(z)
      z = Dense(1, activation="linear")(z)

      model = Model(inputs=[model1.input, model2.input,model3.input], outputs=z)
      model.compile(loss='mean_squared_error', optimizer='adam')

      early_stop = EarlyStopping(monitor='loss', patience=6, verbose=1)
      history_model_GARCH_lstm = model.fit([X_tr_t, X_train1,X_train2], y_train, epochs=100, batch_size=32, verbose=1, shuffle=False,callbacks=[early_stop])
      #train_acc, test_acc = test_model(model, X_tr_t, X_tst_t, y_train, y_test)
      y_pred_test_GARCH_LSTM = model.predict([X_tst_t, X_test1,X_test2])
      y_pred_test_GARCH_LSTM_train = model.predict([X_tr_t, X_train1,X_train2])
      #lstm_GARCH_result.append(test_acc)

      #pred_LSTM.append(y_pred_test_LSTM)
      pred_GARCH_LSTM.append(y_pred_test_GARCH_LSTM)
      pred_GARCH_LSTM_train.append(y_pred_test_GARCH_LSTM_train)


    #lstm.append([win_sz, min(lstm_result), np.mean(lstm_result), np.std(lstm_result)])
    #GARCH_lstm.append([win_sz, min(lstm_GARCH_result), np.mean(lstm_GARCH_result), np.std(lstm_GARCH_result)])  

    
    plot_GARCH_lstm = [0] * len(pred_GARCH_LSTM[0])
    for pred in pred_GARCH_LSTM:
        plot_GARCH_lstm = list(map(add, plot_GARCH_lstm, pred))

    for i in range(len(plot_GARCH_lstm)):
        plot_GARCH_lstm[i] = plot_GARCH_lstm[i] / 15

    plot_GARCH_lstm_train = [0] * len(pred_GARCH_LSTM_train[0])
    for pred in pred_GARCH_LSTM_train:
        plot_GARCH_lstm_train = list(map(add, plot_GARCH_lstm_train, pred))

    for i in range(len(plot_GARCH_lstm_train)):
        plot_GARCH_lstm_train[i] = plot_GARCH_lstm_train[i] / 15

    output[sym + "_" + str(win_sz) + "_test_LSTM"] = pd.Series(np.reshape(plot_GARCH_lstm, (-1)))
    output_train[sym + "_" + str(win_sz) + "_train_LSTM"] = pd.Series(np.reshape(plot_GARCH_lstm_train, (-1)))
    output[sym + "_" + str(win_sz) + "_test_real_LSTM"] = pd.Series(np.reshape(y_test, (-1)))
    output_train[sym + "_" + str(win_sz) + "_train_real_LSTM"] = pd.Series(np.reshape(y_train, (-1)))

    f = plt.figure(figsize=(12,4))
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)

    ax1.plot(y_test, '-', label='True test Values', color='b')
    ax1.plot(plot_GARCH_lstm, label='LSTM test Prediction', color='r')
    ax1.set_title("Prediction")
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Normalized Stock Prices')
    ax1.legend()

    ax2.plot(y_train, '-', label='True Train Values', color='b')
    ax2.plot(plot_GARCH_lstm_train, label='LSTM Train Prediction', color='r')
    ax2.set_title("Prediction")
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Normalized Stock Prices')
    ax2.legend()
#output.to_excel('output.xlsx')   
#output_train.to_excel('output_train.xlsx')   


# In[384]:


df1 = output_train
#df1.index = data['Timestamp'].iloc[0:df1.shape[0]].values
df2 = output
#df2.index = data['Timestamp'].iloc[df1.shape[0]:(df1.shape[0]+df2.shape[0])].values


# In[385]:


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


# In[386]:


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

