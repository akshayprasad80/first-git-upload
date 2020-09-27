#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[132]:


data=pd.read_csv('train_data.csv')


# In[65]:





# In[133]:


data.head()


# In[67]:


#info about the Dataset
data.info()


# In[134]:


#Shape of the dataset
data.shape


# In[135]:


#Description about the Dataset
data.describe()


# In[136]:


#Datatypes of the features
data.dtypes


# In[72]:


#Checking for missing values
data.isnull().sum()


# In[137]:


data['LoanAmount']=data['LoanAmount'].fillna(data['LoanAmount'].mean())


# In[138]:


data['Credit_History']=data['Credit_History'].fillna(data['Credit_History'].median())


# In[139]:


data.isnull().sum()


# In[140]:


data.dropna(inplace=True)


# In[141]:


data.shape


# In[142]:


data['Gender'].value_counts()


# In[143]:


data['Married'].value_counts()


# In[144]:


data['Education'].value_counts()


# In[145]:


data['Self_Employed'].value_counts()


# In[146]:


data['Property_Area'].value_counts()


# In[147]:


plt.boxplot(data['ApplicantIncome'])


# In[148]:


plt.boxplot(data['CoapplicantIncome'])


# In[149]:


plt.boxplot(data['LoanAmount'])


# In[166]:



plt.boxplot(data['Loan_Status'])


# In[151]:


plt.boxplot(data['Credit_History'])


# In[167]:


print(pd.crosstab(data['Property_Area'],data['Loan_Status']))


# In[168]:


sns.countplot(data['Property_Area'],hue=data['Loan_Status'])


# In[169]:


print(pd.crosstab(data['Gender'],data['Loan_Status']))


# In[170]:


sns.countplot(data['Gender'],hue=data['Loan_Status'])


# In[171]:


sns.countplot(data['Married'],hue=data['Loan_Status'])


# In[172]:


sns.countplot(data['Education'],hue=data['Loan_Status'])


# In[173]:


data['Loan_Status'].replace('N',0,inplace=True)
data['Loan_Status'].replace('Y',1,inplace=True)


# In[174]:


plt.title('Correlation Matrix')
sns.heatmap(data.corr(),annot=True)


# In[175]:


df2=data.drop(labels=['ApplicantIncome'],axis=1)


# In[176]:


df2=df2.drop(labels=['CoapplicantIncome'],axis=1)


# In[177]:


df2=df2.drop(labels=['LoanAmount'],axis=1)


# In[178]:


df2=df2.drop(labels=['Loan_Amount_Term'],axis=1)


# In[179]:


df2=df2.drop(labels=['Loan_ID'],axis=1)


# In[180]:


df2.head()


# In[181]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
le=LabelEncoder()
ohe=OneHotEncoder()


# In[182]:


df2['Property_Area']=le.fit_transform(df2['Property_Area'])


# In[183]:


df2['Dependents']=le.fit_transform(df2['Dependents'])


# In[184]:


df2=pd.get_dummies(df2)


# In[185]:


df2.dtypes


# In[186]:


df2=df2.drop(labels=['Gender_Female'],axis=1)


# In[187]:


df2=df2.drop(labels=['Married_No'],axis=1)


# In[188]:


df2=df2.drop(labels=['Education_Not Graduate'],axis=1)


# In[189]:


df2=df2.drop(labels=['Self_Employed_No'],axis=1)


# In[190]:


df2.head()


# In[191]:


plt.title('Correlation Matrix')
sns.heatmap(df2.corr(),annot=True)


# In[192]:


df2=df2.drop('Self_Employed_Yes',1)


# In[193]:


df2=df2.drop('Dependents',1)


# In[194]:


df2=df2.drop('Education_Graduate',1)


# In[195]:


X=df2.drop('Loan_Status',1)


# In[208]:


from sklearn.model_selection import train_test_split
X=data.iloc[:, :-1]
y=data.iloc[:, 12]


# In[210]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=6)


# In[211]:


print('Shape of X_train is: ',X_train.shape)
print('Shape of X_test is: ',X_test.shape)
print('Shape of y_train is: ',y_train.shape)
print('Shape of y_test is: ',y_test.shape)


# In[221]:


from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd


# In[231]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=6)


# In[232]:


print('Shape of X_train is: ',x_train.shape)
print('Shape of X_test is: ',x_test.shape)
print('Shape of Y_train is: ',y_train.shape)
print('Shape of y_test is: ',y_test.shape)


# In[233]:


from sklearn.linear_model import LogisticRegression
log=LogisticRegression()


# In[234]:


log.fit(x_train,y_train)


# In[245]:


log.score(x_train,y_train)
pred=log.predict(x_test)


# In[246]:


from sklearn.metrics import accuracy_score


# In[249]:


from sklearn import metrics
accuracy_score(y_test,pred)


# In[250]:


metrics.confusion_matrix(y_test,pred)


# In[251]:


metrics.recall_score(y_test,pred)


# In[252]:


metrics.precision_score(y_test,pred)


# In[253]:


metrics.f1_score(y_test,pred)


# In[254]:


data={'y_test':y_test,'pred':pred}
pd.DataFrame(data=data)


# In[255]:


from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()


# In[256]:


clf.fit(x_train,y_train)


# In[257]:


clf.score(x_train,y_train)


# In[258]:


pred1=clf.predict(x_test)


# In[259]:


accuracy_score(y_test,pred1)


# In[260]:


metrics.confusion_matrix(y_test,pred1)


# In[261]:


metrics.f1_score(y_test,pred1)


# In[262]:


metrics.f1_score(y_test,pred1)


# In[263]:


metrics.recall_score(y_test,pred1)


# In[264]:


metrics.precision_score(y_test,pred1)


# In[ ]:




