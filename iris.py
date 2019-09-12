# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 12:14:28 2019

@author: nssingh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
dataset = load_iris()

X=dataset.data
y= dataset.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)

from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)

y_predict=log_reg.predict(X_test)

from sklearn.metrics import confusion_matrix
cn=confusion_matrix(y_test,y_predict)

log_reg.score(X_test,y_test)
log_reg.score(X_train,y_train)

y_pred_1=log_reg.predict(X_train)

cnm1=confusion_matrix(y_train,y_pred_1)



log_reg.score(X_train,y_train)
 from sklearn.metrics import precision_score, recall_score, f1_score
 precision_score(y_predict,y_test, average='macro')
 precision_score(y_predict,y_test, average='micro')
 recall_score(y_test,y_predict, average='macro')
 recall_score(y_test,y_predict, average='micro')
 f1_score(y_test,y_predict,average='macro')