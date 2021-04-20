#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np


# In[5]:


mypath = 'Predictions/Copper'


# In[6]:


from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]


# In[7]:


onlyfiles


# In[8]:


data1 = pd.read_csv('Predictions/Copper/Copper_E-LSTM_test.csv')
data1.Volatility_Copper_5_test_LSTM = data1.Volatility_Copper_5_test_LSTM.shift(5)
data1.Volatility_Copper_11_test_LSTM = data1.Volatility_Copper_11_test_LSTM.shift(11)
data1.Volatility_Copper_22_test_LSTM = data1.Volatility_Copper_22_test_LSTM.shift(22)
data1.Volatility_Copper_5_test_real_LSTM = data1.Volatility_Copper_5_test_real_LSTM.shift(5)
data1.Volatility_Copper_11_test_real_LSTM = data1.Volatility_Copper_11_test_real_LSTM.shift(11)
data1.Volatility_Copper_22_test_real_LSTM = data1.Volatility_Copper_22_test_real_LSTM.shift(22)
data1 = data1.dropna()
data = data1['Volatility_Copper_5_test_real_LSTM']
data = pd.DataFrame(data)


# In[9]:


i=1
for files in onlyfiles:
    x = 'Predictions/Copper/{fname}'.format(fname = files)
    data1 = pd.read_csv(x)
    data1.Volatility_Copper_5_test_LSTM = data1.Volatility_Copper_5_test_LSTM.shift(5)
    data1.Volatility_Copper_11_test_LSTM = data1.Volatility_Copper_11_test_LSTM.shift(11)
    data1.Volatility_Copper_22_test_LSTM = data1.Volatility_Copper_22_test_LSTM.shift(22)
    data1.Volatility_Copper_5_test_real_LSTM = data1.Volatility_Copper_5_test_real_LSTM.shift(5)
    data1.Volatility_Copper_11_test_real_LSTM = data1.Volatility_Copper_11_test_real_LSTM.shift(11)
    data1.Volatility_Copper_22_test_real_LSTM = data1.Volatility_Copper_22_test_real_LSTM.shift(22)
    data1 = data1.dropna()
    data[i] = data1['Volatility_Copper_5_test_LSTM']
    i=i+1
    
    


# In[10]:


data.head()


# In[11]:


data.plot()


# In[12]:


onlyfiles


# In[13]:


data.head()


# In[14]:


test = []
from scipy.stats import wilcoxon
for i in range(8):
    test.append("NEW")
    for j in range(8):
        x2 = data[(i+1)]
        #x2 = [round(x,8) for x in x2]
        x2 = x2.array
        x3 = data[(j+1)]
        #x3 = [round(x,8) for x in x3]
        x3 = x3.array
        if(i!=j):
            test.append(wilcoxon(x2,x3))
        else:
            test.append(0)
        


# In[15]:


test = pd.DataFrame(test)


# In[16]:


from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df2, title = "Download CSV file", filename = "Copper.csv"):
    csv = df2.to_csv(index=True)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(test)

