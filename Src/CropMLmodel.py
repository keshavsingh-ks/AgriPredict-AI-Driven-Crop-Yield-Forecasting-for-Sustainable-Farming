#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing 

# In[47]:


import pandas as pd
import numpy as np
import os


# In[48]:


data = pd.read_csv("Dataset.csv")


# In[46]:


data = data.replace('?', np.nan)
print("Crop data. Size={}\nNumber of missing values".format(data.shape))
print(data.isna().sum())
df=data.fillna(data.ph.median())


# In[5]:


print("Concatanated dataset. Size={}\nNumber of missing values".format(df.shape))
print(df.isna().sum())
df.to_csv(os.path.join('Dataset.csv'), index=False)
print("Dataset created successfully")


# In[6]:


data.head()


# In[7]:


data.tail()


# In[8]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[9]:


dataset = pd.read_csv('Dataset.csv')


# In[10]:


dataset.head()


# In[11]:


type(dataset)


# In[12]:


dataset.shape


# In[13]:


dataset.info()


# In[14]:


dataset.describe().T


# In[15]:


dataset.isnull().sum()


# # Model Cretaion

# In[16]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()


# In[17]:


dataset['class'] = labelencoder.fit_transform(dataset['label'])
dataset.head(5)


# In[18]:


dataset.corr()
sns.heatmap(dataset.corr(), annot = True)
plt.show()


# In[19]:


X = dataset.iloc[:, [1,2,3,4]].values
Y = dataset.iloc[:, [5]].values


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42 )


# In[21]:


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)


# In[22]:


print(X_train)


# In[23]:


from sklearn.svm import SVC
svc = SVC(kernel = 'linear', random_state = 42)
svc.fit(X_train, Y_train)


# In[24]:


from sklearn.tree import DecisionTreeClassifier
dectree = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
dectree.fit(X_train, Y_train)


# In[25]:


from sklearn.ensemble import RandomForestClassifier
ranfor = RandomForestClassifier(n_estimators = 11, criterion = 'entropy', random_state = 42)
ranfor.fit(X_train, Y_train)


# In[26]:


Y_pred_svc = svc.predict(X_test)
Y_pred_dectree = dectree.predict(X_test)
Y_pred_ranfor = ranfor.predict(X_test)


# In[27]:


from sklearn.metrics import accuracy_score


# In[28]:


accuracy_svc = accuracy_score(Y_test, Y_pred_svc)
accuracy_dectree = accuracy_score(Y_test, Y_pred_dectree)
accuracy_ranfor = accuracy_score(Y_test, Y_pred_ranfor)


# In[29]:


print("Support Vector Classifier: " + str(accuracy_svc * 100))
print("Decision tree: " + str(accuracy_dectree * 100))
print("Random Forest: " + str(accuracy_ranfor * 100))


# In[30]:


c=accuracy_svc * 100
d=accuracy_dectree * 100
e=accuracy_ranfor * 100


# In[31]:


scores = [c,d,e]
algorithms = ["Support Vector Machine","Decision Tree","Random Forest"]    

for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")


# In[32]:


sns.set(rc={'figure.figsize':(8,8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")
sns.barplot(algorithms,scores)


# In[33]:


import joblib 
joblib.dump(ranfor, 'ranfor_model1.pkl') 
ranfor_from_joblib = joblib.load('ranfor_model1.pkl')
ranfor_from_joblib.predict([[20.87974371,82.00274423,6.502985292,202.9355362]])
print("Model successfully created...!")


# In[34]:


ranfor_from_joblib.predict([[20.87974371,82.00274423,6.502985292,202.9355362]])


# In[35]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred_ranfor)


# In[36]:


plt.figure(figsize = (10,10))
sns.heatmap(pd.DataFrame(cm), annot=True)


# In[37]:



X1 = dataset.iloc[:, [1,2,3,4]].values
Y1 = dataset.iloc[:, [6]].values


# In[38]:


from sklearn.model_selection import train_test_split
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X1, Y1, test_size = 0.20, random_state = 42 )


# In[39]:


from sklearn.ensemble import RandomForestClassifier
ranfor = RandomForestClassifier(n_estimators = 11, criterion = 'entropy', random_state = 42)
ranfor.fit(X_train1, Y_train1)


# In[40]:


import joblib 
joblib.dump(ranfor, 'ranfor_model2.pkl') 
ranfor_from_joblib = joblib.load('ranfor_model2.pkl')  
print("Model successfully created...!")


# In[41]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import datetime
import time


# In[ ]:




