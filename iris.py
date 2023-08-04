#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets,linear_model
import seaborn ,matplotlib,numpy
from sklearn.linear_model import LinearRegression


# In[2]:



iris=datasets.load_iris()
iris
hs=pd.DataFrame(iris.data,columns=iris.feature_names)
hs


# In[3]:


hs.info()


# In[4]:


plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.title("iris")
plt.plot(hs)
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("mpl-gallery")

# make data
np.random.seed(1)
x = 4 + np.random.normal(0, 1.5, 200)

# plot:
fig, ax = plt.subplots()

ax.hist(x, bins=8, linewidth=0.5, edgecolor="white")

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 56), yticks=np.linspace(0, 56, 9))

plt.show()


# In[9]:



k=sns.scatterplot(data=hs)


# In[10]:


sns.heatmap(data=hs)


# In[11]:


k=sns.jointplot(data=hs)


# In[12]:


k=sns.violinplot(data=hs)


# In[15]:


sns.pairplot(data=hs)


# In[27]:


x=hs["sepal length (cm)"].values
y=hs["sepal width (cm)"].values
from sklearn.model_selection import train_test_split
x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
x_train


# In[28]:


x_test


# In[29]:



y_test


# In[30]:


y_train


# In[32]:


x_train=x_train.reshape(-1,-1)
y_test=y_test.reshape(-1,1)
x_test=x_test.reshape(-1,1)
y_train=y_train.reshape(-1,1)          


# In[ ]:




