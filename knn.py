# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 17:18:23 2019

@author: nssingh
"""

import numpy as num
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.datasets import load_breast_cancer
datasets=load_breast_cancer()

X=datasets.data
y= datasets.target

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()

knn.fit(X,y)

knn.score(X,y)
 y_predict=knn.predict(X)
 
 from sklearn.metrics import confusion_matrix
 cm=confusion_matrix(y,y_predict)
 