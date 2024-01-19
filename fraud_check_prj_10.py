#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import confusion_matrix,accuracy_score


# In[2]:


fraud_check = pd.read_csv("E:/Python docs/Fraud_check.csv")
print(fraud_check.head(5))


# In[4]:


##Converting the Taxable income variable
fraud_check["income"]="<=30000"
fraud_check.loc[fraud_check["Taxable.Income"]>=30000,"income"]="Good"
fraud_check.loc[fraud_check["Taxable.Income"]<=30000,"income"]="Risky"


# In[5]:


##Droping the Taxable income variable
fraud_check.drop(["Taxable.Income"],axis=1,inplace=True)


# In[6]:


fraud_check.rename(columns={"Undergrad":"undergrad","Marital.Status":"marital","City.Population":"population","Work.Experience":"experience","Urban":"urban"},inplace=True)
## As we are getting error as "ValueError: could not convert string to float: 'YES'".
## Model.fit doesnt not consider String. So, we encode


# In[8]:


from sklearn import preprocessing
le=preprocessing.LabelEncoder()
for column_name in fraud_check.columns:
    if fraud_check[column_name].dtype == object:
        fraud_check[column_name] = le.fit_transform(fraud_check[column_name])
    else:
        pass


# In[10]:


##Splitting the data into X and y
X = fraud_check.iloc[:,0:5]
y = fraud_check.iloc[:,5]


# In[11]:


## Collecting the column names
colnames = list(fraud_check.columns)
predictors = colnames[0:5]
target = colnames[5]


# In[12]:


target


# In[14]:


##Splitting the data into train and test

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,stratify = y)


# In[17]:


##Model building
from sklearn.ensemble import RandomForestClassifier as RF
model = RF(n_jobs = 3,n_estimators = 15, criterion = "entropy")
model.fit(X_train,y_train)


# In[34]:


model.estimators_
model.classes_
model.n_classes_


# In[21]:


##Predictions on train data
prediction = model.predict(X_train)


# In[22]:


prediction


# In[23]:


##Accuracy
# For accuracy 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_train,prediction)


# In[24]:


accuracy


# In[37]:


print("Actual - ",list(y_test))


# In[44]:


print("Actual - ",list(y_test))
print("Predict- ", list(prediction))


# In[ ]:




