# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 15:02:06 2021

@author: workstation
"""

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
dataset.head()
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

dataset.info(verbose=True)
dataset.describe()
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#KNN
from sklearn.neighbors import KNeighborsClassifier
test_scores = []
train_scores = []

for i in range(1,15):

    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))
## score that comes from testing on the same datapoints that were used for training
max_train_score = max(train_scores)
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))

## score that comes from testing on the datapoints that were split in the beginning to be used for testing solely
max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))

#Visualising for KNN

plt.figure(figsize=(12,5))
p = sns.lineplot(range(1,15),train_scores,marker='*',label='Train Score')
p = sns.lineplot(range(1,15),test_scores,marker='o',label='Test Score')

#Logistic regression
dataset.head()
dataset.describe()  
dataset.info()  
dataset.isnull().sum()
dataset.drop("User ID",axis = 1,inplace = True)
dataset.head()
dataset["Age"].unique()
# creating bins for our age:
bins = np.linspace(18,56,4)
label = ["18-31","32-45","45-60"]

dataset["age_range"] = pd.cut(dataset["Age"],bins = bins,labels=label,include_lowest = True)
dataset
# creating dummy variables:
gender = pd.get_dummies(dataset["Gender"],drop_first = True)
age=pd.get_dummies(dataset["age_range"],drop_first= True)
dataset= pd.concat([dataset,age,gender],axis=1)

dataset= dataset.drop(["Gender","Age","age_range"],axis = 1)

#scaling is done after train test split:
from sklearn.model_selection import train_test_split
np.random.seed(0)
dataset_train,dataset_test = train_test_split(dataset,train_size=0.7,random_state =100)
dataset_train.shape

# building the first model with all the parameters:
y_train = dataset_train.pop("Purchased")
X_train = dataset_train

X_train_sm = sm.add_constant(X_train)
log_reg_1 = sm.GLM(y_train,X_train_sm,family = sm.families.Binomial()).fit()
log_reg_1.summary()

from statsmodels.stats.outliers_influence import variance_inflation_factor

#VIF
VIF = pd.DataFrame()
VIF["Features"] = X_train.columns
VIF["VIF"] = [variance_inflation_factor(X_train.values,i) for i in range(X_train.shape[1])]
VIF = VIF.sort_values(by = "VIF",ascending = False)

#removing 32-45
X_train = X_train.drop("32-45",axis = 1)
X_train_sm = sm.add_constant(X_train)
log_reg_2 = sm.GLM(y_train,X_train_sm,family = sm.families.Binomial()).fit()
log_reg_2.summary()

# Random Forest Classification
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()