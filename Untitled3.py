
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pandas import Series,DataFrame


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


sns.set_style('whitegrid')
get_ipython().magic('matplotlib inline')


# In[4]:


from sklearn.datasets import load_boston


# In[5]:


boston = load_boston()


# In[6]:


print(boston.DESCR)


# In[7]:


plt.hist(boston.target,bins=50)
plt.xlabel('Price in $1000s')
plt.ylabel('Number of houses')


# In[8]:


plt.scatter(boston.data[:,5],boston.target)
plt.ylabel('Price in $1000s')
plt.xlabel('Number of rooms')


# In[9]:


boston_df = DataFrame(boston.data)
boston_df.columns = boston.feature_names
boston_df.head()


# In[10]:


boston_df['Price'] = boston.target


# In[12]:


boston_df['Price']


# In[13]:


boston_df.head()


# In[14]:


sns.lmplot('RM','Price',data = boston_df)


# In[15]:


import sklearn
from sklearn.linear_model import LinearRegression


# In[16]:


lreg = LinearRegression()


# In[17]:


X_multi = boston_df.drop('Price',1)
Y_target = boston_df.Price


# In[18]:


lreg.fit(X_multi,Y_target)


# In[19]:


print(' The estimated intercept coefficient is %.2f ' %lreg.intercept_)


# In[20]:


print(' The number of coefficients used was %d ' % len(lreg.coef_))

