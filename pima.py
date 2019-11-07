# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 15:02:42 2019

@author: Acer
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Load Data
#------------------------------------------------------------------------------
df = pd.read_csv( 'diabetes.csv')
#------------------------------------------------------------------------------
#dataset info
print(df.info())
print(df.describe())
print(df.isnull().count())
#------------------------------------------------------------------------------
#x and y for train_test_split
X = df.iloc[:, :-1]     #x = df.drop(['Outcome'],axis=1)
y = df.iloc[:, -1]      #y = df['Outcome']
 
#------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
#------------------------------------------------------------------------------
from sklearn.naive_bayes import GaussianNB
#Model
gb = GaussianNB()

#fiting the model
gb.fit(X_train, y_train)

#prediction
y_pred = gb.predict(X_test)

#Accuracy, Precision, Recall, F1
print("Accuracy ", gb.score(X_test, y_test)*100)
print('Precision score: ', format(precision_score(y_test, y_pred)))
print('Recall score: ', format(recall_score(y_test, y_pred)))
print('F1 score: ', format(f1_score(y_test, y_pred)))

#Plot the confusion matrix
sns.set(font_scale=1.5)
cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot=True, fmt='g')
plt.show()
#------------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
#Model
LR = LogisticRegression()

#fiting the model
LR.fit(X_train, y_train)

#prediction
y_pred = LR.predict(X_test)

#Accuracy
print("Accuracy ", LR.score(X_test, y_test)*100)
print('Precision score: ', format(precision_score(y_test, y_pred)))
print('Recall score: ', format(recall_score(y_test, y_pred)))
print('F1 score: ', format(f1_score(y_test, y_pred)))
#Plot the confusion matrix
sns.set(font_scale=1.5)
cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot=True, fmt='g')
plt.show()