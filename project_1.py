#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd

import numpy as np


dfdata=pd.read_csv(r'C:\\Users\\User\\Desktop\\Assignment\\Assignment 1&2\\udata.csv')
dfuser=pd.read_csv(r'C:\\Users\\User\\Desktop\\Assignment\\Assignment 1&2\\uuser.csv')
dfgenre=pd.read_csv(r'C:\\Users\\User\\Desktop\\Assignment\\Assignment 1&2\\ugenre.csv')


# In[5]:


print("1. the number of users and items is: ",len(dfdata))                #1


# In[7]:


dfdata.columns


# In[12]:


dfdata.rating


# In[17]:


print("2. Average rating of all user is: ", dfdata['rating'].mean())            #2


# In[29]:


dfdatauser=dfdata[(dfdata['user id']==10)]                                          #3
print("3. The average rating of user 10 is: ",dfdatauser['rating'].mean())


# In[31]:


dfdataitem=dfdata[(dfdata['item id']==10)]                                          #4
print("4. The average rating of item 10 is: ",dfdataitem['rating'].mean())


# In[ ]:




