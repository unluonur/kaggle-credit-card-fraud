#!/usr/bin/env python
# coding: utf-8

# ### You can download the data from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix

data = pd.read_csv('creditcard.csv')

print('data loaded')


# In[2]:


x = data.iloc[:,1:-1]
y = data.iloc[:,-1]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

print('data prepared')


# In[3]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

start = time.time()
lr.fit(X_train, y_train)
print('training completed in %s seconds'%(time.time() - start))

start = time.time()
y_pred = lr.predict(X_test)
print('prediction completed in %s seconds'%(time.time() - start))

cm = confusion_matrix(y_test, y_pred)

print(cm)


# In[4]:


from sklearn.svm import SVC

svc = SVC(kernel='rbf')

start = time.time()
svc.fit(X_train, y_train)
print('training completed in %s seconds'%(time.time() - start))

start = time.time()
y_pred = svc.predict(X_test)
print('prediction completed in %s seconds'%(time.time() - start))

cm = confusion_matrix(y_test, y_pred)

print(cm)


# In[5]:


from sklearn.svm import SVC

svc = SVC(kernel='poly')

start = time.time()
svc.fit(X_train, y_train)
print('training completed in %s seconds'%(time.time() - start))

start = time.time()
y_pred = svc.predict(X_test)
print('prediction completed in %s seconds'%(time.time() - start))

cm = confusion_matrix(y_test, y_pred)

print(cm)


# In[6]:


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

start = time.time()
gnb.fit(X_train, y_train)
print('training completed in %s seconds'%(time.time() - start))

start = time.time()
y_pred = gnb.predict(X_test)
print('prediction completed in %s seconds'%(time.time() - start))

cm = confusion_matrix(y_test, y_pred)

print(cm)


# In[8]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion='entropy')

start = time.time()
dt.fit(X_train, y_train)
print('training completed in %s seconds'%(time.time() - start))

start = time.time()
y_pred = dt.predict(X_test)
print('prediction completed in %s seconds'%(time.time() - start))

cm = confusion_matrix(y_test, y_pred)

print(cm)
    


# In[13]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=3, criterion='entropy')

start = time.time()
rfc.fit(X_train, y_train)
print('training completed in %s seconds'%(time.time() - start))

start = time.time()
y_pred = rfc.predict(X_test)
print('prediction completed in %s seconds'%(time.time() - start))

cm = confusion_matrix(y_test, y_pred)

print(cm)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')

start = time.time()
knn.fit(X_train, y_train)
print('training completed in %s seconds'%(time.time() - start))

start = time.time()
y_pred = knn.predict(X_test)
print('prediction completed in %s seconds'%(time.time() - start))

cm = confusion_matrix(y_test, y_pred)

print(cm)

