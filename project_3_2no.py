#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from pandas import DataFrame
import statsmodels.api as sm
import numpy as np
from sklearn import linear_model

df=pd.read_csv(r'C:\\Users\\User\\Desktop\\Assignment\\Assignment 3\\airQualityData.csv')
df


# In[5]:


df1=df[["PM25","SO2","PM10","CO","O3_8"]]
df1_a=df[["SO2","PM10","CO","O3_8"]]
df1_b=df[["PM25"]]
df1
df1_a


# In[6]:


df1_b


# In[7]:


from sklearn.model_selection import train_test_split
df1_a_train,df1_a_test,df1_b_train,df1_b_test=train_test_split(df1_a,df1_b,test_size=0.6,random_state=0)


# In[8]:


df1_a_train.shape


# In[9]:


df1_b_train.shape


# In[10]:


from sklearn.linear_model import Lasso
lasso1=Lasso(alpha=0.1)                                  #for alpha=0.1
lasso1.fit(df1_a_train,df1_b_train)


# In[11]:


df1_b_pred=lasso1.predict(df1_a_test)


# In[12]:


df1_b_pred


# In[13]:


lasso1.coef_


# In[14]:


lasso1.intercept_


# In[15]:


from sklearn.metrics import mean_squared_error            #Calculating MSE
print("MSE: "+str(mean_squared_error(df1_b_test,df1_b_pred)))  


# In[16]:


print("RMSE: "+str(np.sqrt(mean_squared_error(df1_b_test,df1_b_pred))))           #Calculating RMSE


# In[17]:


from sklearn.metrics import mean_absolute_error                               #Calculating MAE
print("MAE: "+str(mean_absolute_error(df1_b_test,df1_b_pred)))     


# In[18]:


from sklearn.metrics import r2_score      #Evaluating with R-square
r2_score(df1_b_test,df1_b_pred)


# In[25]:


import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
plt.scatter(df1_b_test,df1_b_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')


# In[20]:


df1_a_train2,df1_a_test2,df1_b_train2,df1_b_test2=train_test_split(df1_a,df1_b,test_size=0.6,random_state=0)
lasso2=Lasso(alpha=1)                                  #for alpha=1
lasso2.fit(df1_a_train2,df1_b_train2)
df1_b_pred2=lasso2.predict(df1_a_test2)
print("MSE: "+str(mean_squared_error(df1_b_test2,df1_b_pred2)))  
print("RMSE: "+str(np.sqrt(mean_squared_error(df1_b_test2,df1_b_pred2)))) 
print("MAE: "+str(mean_absolute_error(df1_b_test2,df1_b_pred2)))    
print("R2 score: "+str(r2_score(df1_b_test2,df1_b_pred2)))



# In[21]:


import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))                       #for alpha=1
plt.scatter(df1_b_test2,df1_b_pred2)
plt.xlabel('Actual')
plt.ylabel('Predicted')


# In[23]:


df1_a_train3,df1_a_test3,df1_b_train3,df1_b_test3=train_test_split(df1_a,df1_b,test_size=0.6,random_state=0)
lasso3=Lasso(alpha=5)                                  #for alpha=5
lasso3.fit(df1_a_train3,df1_b_train2)
df1_b_pred3=lasso3.predict(df1_a_test3)
print("MSE: "+str(mean_squared_error(df1_b_test3,df1_b_pred3)))  
print("RMSE: "+str(np.sqrt(mean_squared_error(df1_b_test3,df1_b_pred3)))) 
print("MAE: "+str(mean_absolute_error(df1_b_test3,df1_b_pred3)))    
print("R2 score: "+str(r2_score(df1_b_test3,df1_b_pred3)))


# In[24]:


import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))                       #for alpha=5
plt.scatter(df1_b_test3,df1_b_pred3)
plt.xlabel('Actual')
plt.ylabel('Predicted')


# In[ ]:




