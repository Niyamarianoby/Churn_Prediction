#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.preprocessing import LabelEncoder 


# In[3]:


churn=pd.read_csv(r"D:\dataset\churn.csv")
churn


# In[4]:


#drop rownumber and id
churn.drop('RowNumber',axis=1,inplace=True)
churn.drop('CustomerId',axis=1,inplace=True)


# In[5]:


churn


# In[6]:


LE=LabelEncoder()
dummy=['Surname','Gender','Geography']
for i in dummy:
    churn[i]=LE.fit_transform(churn[i].astype(str))


# In[7]:


churn.corr()


# In[8]:


#exploratory Data Analysis
churn.isnull().sum()


# In[9]:


f=churn.iloc[:,1].mean()


# In[10]:


churn.iloc[:,1]=churn.iloc[:,1].fillna(f)


# In[11]:


f=churn.iloc[:,7].mean()


# In[12]:


churn.iloc[:,7]=churn.iloc[:,7].fillna(f)


# In[13]:


f=churn.iloc[:,10].mean()


# In[14]:


churn.iloc[:,10]=churn.iloc[:,10].fillna(f)


# In[15]:


churn.isnull().sum()


# In[16]:


print(churn.duplicated().sum())


# In[17]:


churn


# In[18]:


train,test=train_test_split(churn,test_size=0.2)


# In[19]:


train


# In[20]:


test


# In[21]:


trainx=train.iloc[:,0:12]
trainy=train.iloc[:,12]
testx=test.iloc[:,0:12]
testy=test.iloc[:,12]


# In[25]:


#Decision Tree


# In[26]:


#model_building
modeldt=DTC(criterion='entropy')
predrf=modeldt.fit(trainx,trainy).predict(testx)


# In[27]:


predrf


# In[28]:


testy


# In[30]:


acc=np.mean(predrf==testy)


# In[31]:


acc


# In[ ]:


#Naive Bayes


# In[32]:


from sklearn.naive_bayes  import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import confusion_matrix


# In[33]:


#modelbuilding
a=GaussianNB()
predgaussian=a.fit(trainx,trainy).predict(testx)


# In[34]:


predgaussian


# In[35]:


testy


# In[36]:


cm1=confusion_matrix(testy,predgaussian)


# In[37]:


cm1


# In[38]:


accgaussian=1554+36/(1554+44+366+36)


# In[39]:


accgaussian


# In[41]:


acc=np.mean(predgaussian==testy)


# In[42]:


acc


# In[43]:


b=MultinomialNB()
predmultnb=a.fit(trainx,trainy).predict(testx)
predmultnb


# In[44]:


testy


# In[45]:


cm2=confusion_matrix(testy,predmultnb)
cm2


# In[46]:


accmult=1554+36/(1554+44+366+36)


# In[47]:


accmult


# In[48]:


acc=np.mean(predmultnb==testy)


# In[49]:


acc


# In[50]:


c=BernoulliNB()
res=c.fit(trainx,trainy).predict(testx)


# In[51]:


res


# In[52]:


testy


# In[54]:


cm3=confusion_matrix(testy,res)


# In[55]:


cm3


# In[56]:


acc=np.mean(res==testy)


# In[57]:


acc


# In[58]:


#Random Forest


# In[59]:


from sklearn.ensemble import RandomForestClassifier as RF


# In[61]:


modelrf=RF(n_estimators=10,n_jobs=12,oob_score=True,criterion='entropy')
predrf=modelrf.fit(trainx,trainy).predict(testx)


# In[62]:


predrf


# In[63]:


testy


# In[64]:


acc=np.mean(predrf==testy)


# In[65]:


acc


# In[66]:


#K Nearest Neighbor


# In[68]:


from sklearn.neighbors import KNeighborsClassifier as KNC


# In[73]:


acc=[]
for i in range(3,100,2):
    modelknn=KNC(n_neighbors=i)
    modelknn.fit(trainx,trainy)
    predknn=modelknn.predict(testx)
    knnacc=np.mean(predknn==testy)
    acc.append([i,knnacc])


# In[74]:


predknn


# In[75]:


testy


# In[76]:


acc=np.mean(testy==predknn)


# In[77]:


acc


# In[1]:


from sklearn.svm import SVC


# In[22]:


#model_building
model_linear=SVC(kernel='linear')
pred_linear=model_linear.fit(trainx,trainy).predict(testx) 


# In[23]:


acc=np.mean(pred_linear==testy)
acc


# In[24]:


model_poly=SVC(kernel='poly')
predpoly=model_poly.fit(trainx,trainy).predict(testx)


# In[25]:


acc1=np.mean(predpoly==testy)
acc1


# In[26]:


model_rbf=SVC(kernel='rbf')#gaussian-radius basis function
predrbf=model_rbf.fit(trainx,trainy).predict(testx)


# In[27]:


acc2=np.mean(predrbf==testy)
acc2


# In[28]:


model_sigmoid=SVC(kernel='sigmoid')
predsig=model_rbf.fit(trainx,trainy).predict(testx)


# In[29]:


acc3=np.mean(predsig==testy)
acc3


# In[ ]:




