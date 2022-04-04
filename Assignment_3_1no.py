#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from pandas import DataFrame
import statsmodels.api as sm
import numpy as np
from sklearn import linear_model

df=pd.read_csv(r'C:\\Users\\User\\Desktop\\Assignment\\Assignment 3\\airQualitydata.csv')


# In[3]:


df


# In[4]:


df1=df[["PM25","SO2","PM10","CO","O3_8"]]
df1_a=df[["SO2","PM10","CO","O3_8"]]
df1_b=df[["PM25"]]
df1
df1_a


# In[79]:


df1_b


# In[6]:


reg1=linear_model.LinearRegression()
reg1.fit(df1[["SO2","PM10","CO","O3_8"]],df1_b)


# In[7]:


reg1.coef_


# In[8]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df1_a,df1_b,test_size=0.6,random_state=0)


# In[136]:


ml=LinearRegression()
ml.fit(x_train,y_train)


# In[137]:


df1_b_pred=ml.predict(x_test)
print(df1_b_pred)


# In[142]:


print(y_test)


# In[138]:


ml.predict([[7,78,0.6,86]])


# In[139]:


from sklearn.metrics import r2_score      #Evaluating with R-square
r2_score(y_test,df1_b_pred)


# In[143]:


from sklearn.metrics import mean_squared_error,mean_absolute_error
print("MSE: "+str(mean_squared_error(y_test,df1_b_pred)))                   #Calculating MSE
print("RMSE: "+str(np.sqrt(mean_squared_error(y_test,df1_b_pred))))         #Calculating RMSE
print("MAE: "+str(mean_absolute_error(y_test,df1_b_pred)))                  #Calculating MAE


# In[141]:


print(x_test)


# In[149]:


import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
plt.scatter(y_test,df1_b_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')


# In[ ]:




