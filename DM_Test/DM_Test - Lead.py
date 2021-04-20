#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:



  
# Author   : John Tsang
# Date     : December 7th, 2017
# Purpose  : Implement the Diebold-Mariano Test (DM test) to compare 
#            forecast accuracy
# Input    : 1) actual_lst: the list of actual values
#            2) pred1_lst : the first list of predicted values
#            3) pred2_lst : the second list of predicted values
#            4) h         : the number of stpes ahead
#            5) crit      : a string specifying the criterion 
#                             i)  MSE : the mean squared error
#                            ii)  MAD : the mean absolute deviation
#                           iii) MAPE : the mean absolute percentage error
#                            iv) poly : use power function to weigh the errors
#            6) poly      : the power for crit power 
#                           (it is only meaningful when crit is "poly")
# Condition: 1) length of actual_lst, pred1_lst and pred2_lst is equal
#            2) h must be an integer and it must be greater than 0 and less than 
#               the length of actual_lst.
#            3) crit must take the 4 values specified in Input
#            4) Each value of actual_lst, pred1_lst and pred2_lst must
#               be numerical values. Missing values will not be accepted.
#            5) power must be a numerical value.
# Return   : a named-tuple of 2 elements
#            1) p_value : the p-value of the DM test
#            2) DM      : the test statistics of the DM test
##########################################################
# References:
#
# Harvey, D., Leybourne, S., & Newbold, P. (1997). TesLeadg the equality of 
#   prediction mean squared errors. International Journal of forecasLeadg, 
#   13(2), 281-291.
#
# Diebold, F. X. and Mariano, R. S. (1995), Comparing predictive accuracy, 
#   Journal of business & economic statistics 13(3), 253-264.
#
##########################################################
def dm_test(actual_lst, pred1_lst, pred2_lst, h = 1, crit="MSE", power = 2):
    # RouLeade for checking errors
    def error_check():
        rt = 0
        msg = ""
        # Check if h is an integer
        if (not isinstance(h, int)):
            rt = -1
            msg = "The type of the number of steps ahead (h) is not an integer."
            return (rt,msg)
        # Check the range of h
        if (h < 1):
            rt = -1
            msg = "The number of steps ahead (h) is not large enough."
            return (rt,msg)
        len_act = len(actual_lst)
        len_p1  = len(pred1_lst)
        len_p2  = len(pred2_lst)
        # Check if lengths of actual values and predicted values are equal
        if (len_act != len_p1 or len_p1 != len_p2 or len_act != len_p2):
            rt = -1
            msg = "Lengths of actual_lst, pred1_lst and pred2_lst do not match."
            return (rt,msg)
        # Check range of h
        if (h >= len_act):
            rt = -1
            msg = "The number of steps ahead is too large."
            return (rt,msg)
        # Check if criterion supported
        if (crit != "MSE" and crit != "MAPE" and crit != "MAD" and crit != "poly"):
            rt = -1
            msg = "The criterion is not supported."
            return (rt,msg)  
        # Check if every value of the input lists are numerical values
        from re import compile as re_compile
        comp = re_compile("^\d+?\.\d+?$")  
        def compiled_regex(s):
            """ Returns True is string is a number. """
            if comp.match(s) is None:
                return s.isdigit()
            return True
        # for actual, pred1, pred2 in zip(actual_lst, pred1_lst, pred2_lst):
        #     # print(str(abs(actual))[1:-1], str(abs(pred1))[1:-1], str(abs(pred2))[1:-1])
        #     is_actual_ok = compiled_regex(str(abs(actual))[1:-1])
        #     is_pred1_ok = compiled_regex(str(abs(pred1))[1:-1])
        #     is_pred2_ok = compiled_regex(str(abs(pred2))[1:-1])
        #     if (not (is_actual_ok and is_pred1_ok and is_pred2_ok)):  
        #         msg = "An element in the actual_lst, pred1_lst or pred2_lst is not numeric."
        #         rt = -1
        #         return (rt,msg)
        return (rt,msg)
    
    # Error check
    error_code = error_check()
    # Raise error if cannot pass error check
    if (error_code[0] == -1):
        raise SyntaxError(error_code[1])
        return
    # Import libraries
    from scipy.stats import t
    import collections
    import pandas as pd
    import numpy as np
    
    # Initialise lists
    e1_lst = []
    e2_lst = []
    d_lst  = []
    
    # convert every value of the lists into real values
    actual_lst = pd.Series(actual_lst).apply(lambda x: float(x)).tolist()
    pred1_lst = pd.Series(pred1_lst).apply(lambda x: float(x)).tolist()
    pred2_lst = pd.Series(pred2_lst).apply(lambda x: float(x)).tolist()
    
    # Length of lists (as real numbers)
    T = float(len(actual_lst))
    
    # construct d according to crit
    if (crit == "MSE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append((actual - p1)**2)
            e2_lst.append((actual - p2)**2)
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAD"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(abs(actual - p1))
            e2_lst.append(abs(actual - p2))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAPE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(abs((actual - p1)/actual))
            e2_lst.append(abs((actual - p2)/actual))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "poly"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(((actual - p1))**(power))
            e2_lst.append(((actual - p2))**(power))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)    
    
    # Mean of d        
    mean_d = pd.Series(d_lst).mean()
    # Find autocovariance and construct DM test statistics
    def autocovariance(Xi, N, k, Xs):
        Xi = pd.Series(Xi)
        Xi = Xi.fillna(Xi.mean()).to_numpy()
        # print("AutoCov Function", Xi, N, k, Xs)
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N-k):
              autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
        # print("Autocov:", autoCov)
        return (1/(T))*autoCov
    gamma = []
    for lag in range(0,h):
        gamma.append(autocovariance(d_lst,len(d_lst),lag,mean_d)) # 0, 1, 2
    V_d = (gamma[0] + 2*sum(gamma[1:]))/T
    DM_stat=V_d**(-0.5)*mean_d
    harvey_adj=((T+1-2*h+h*(h-1)/T)/T)**(0.5)
    DM_stat = harvey_adj*DM_stat
    # Find p-value
    p_value = 2*t.cdf(-abs(DM_stat), df = T - 1)
    # print(mean_d, V_d)
    # print(gamma, T, h, )
    
    # Construct named tuple for return
    dm_return = collections.namedtuple('dm_return', 'DM p_value')
    
    rt = dm_return(DM = DM_stat, p_value = p_value)
    
    return rt


# In[4]:


mypath = 'Predictions/Lead'

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

onlyfiles

data1 = pd.read_csv('Predictions/Lead/Lead_E-LSTM_test.csv')
data1.Volatility_Lead_5_test_LSTM = data1.Volatility_Lead_5_test_LSTM.shift(5)
data1.Volatility_Lead_11_test_LSTM = data1.Volatility_Lead_11_test_LSTM.shift(11)
data1.Volatility_Lead_22_test_LSTM = data1.Volatility_Lead_22_test_LSTM.shift(22)
data1.Volatility_Lead_5_test_real_LSTM = data1.Volatility_Lead_5_test_real_LSTM.shift(5)
data1.Volatility_Lead_11_test_real_LSTM = data1.Volatility_Lead_11_test_real_LSTM.shift(11)
data1.Volatility_Lead_22_test_real_LSTM = data1.Volatility_Lead_22_test_real_LSTM.shift(22)
data1 = data1.dropna()
data = data1['Volatility_Lead_5_test_real_LSTM']
data = pd.DataFrame(data)

i=1
for files in onlyfiles:
    x = 'Predictions/Lead/{fname}'.format(fname = files)
    data1 = pd.read_csv(x)
    data1.Volatility_Lead_5_test_LSTM = data1.Volatility_Lead_5_test_LSTM.shift(5)
    data1.Volatility_Lead_11_test_LSTM = data1.Volatility_Lead_11_test_LSTM.shift(11)
    data1.Volatility_Lead_22_test_LSTM = data1.Volatility_Lead_22_test_LSTM.shift(22)
    data1.Volatility_Lead_5_test_real_LSTM = data1.Volatility_Lead_5_test_real_LSTM.shift(5)
    data1.Volatility_Lead_11_test_real_LSTM = data1.Volatility_Lead_11_test_real_LSTM.shift(11)
    data1.Volatility_Lead_22_test_real_LSTM = data1.Volatility_Lead_22_test_real_LSTM.shift(22)
    data1 = data1.dropna()
    data[i] = data1['Volatility_Lead_5_test_LSTM']
    i=i+1


# In[6]:


data[1].fillna(data[1].mean(),inplace=True)
data[2].fillna(data[2].mean(),inplace=True)
data[3].fillna(data[3].mean(),inplace=True)
data[4].fillna(data[4].mean(),inplace=True)
data[5].fillna(data[5].mean(),inplace=True)
data[6].fillna(data[6].mean(),inplace=True)
data[7].fillna(data[7].mean(),inplace=True)
data[8].fillna(data[8].mean(),inplace=True)


# In[7]:


data['Volatility_Lead_5_test_real_LSTM'] =  data['Volatility_Lead_5_test_real_LSTM'].astype('float64')

data[1] = data[1].astype('float64')
data[2] = data[2].astype('float64')
data[3] = data[3].astype('float64')
data[4] = data[4].astype('float64')
data[5] = data[5].astype('float64')
data[6] = data[6].astype('float64')
data[7] = data[7].astype('float64')
data[8] = data[8].astype('float64')

actual = data['Volatility_Lead_5_test_real_LSTM'].to_numpy().reshape(-1,)
pred1 = data[1].to_numpy().reshape(-1,)
pred2 = data[2].to_numpy().reshape(-1,)


# In[8]:


test = []
actual = data['Volatility_Lead_5_test_real_LSTM'].to_numpy().reshape(-1,)
for i in range(8):
    test.append("NEW")
    for j in range(8):
        x2 = data[(i+1)]
        pred1 = x2.to_numpy().reshape(-1,)
        x3 = data[(j+1)]
        pred2 = x3.to_numpy().reshape(-1,)
        if(i!=j):
            test.append(dm_test(actual,pred1,pred2,h = 1, crit="MAD"))
        else:
            test.append(0)


# In[9]:


test = pd.DataFrame(test)

from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df2, title = "Download CSV file", filename = "Lead.csv"):
    csv = df2.to_csv(index=True)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(test)

