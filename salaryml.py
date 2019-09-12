# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 09:10:56 2019

@author: nssingh
"""

import numpy as num
import matplotlib.pyplot as plt
import pandas as pd


dataset= pd.read_csv('dataset/sal.csv',names = ['age',
                                                  'workclass',
                                                  'fnlwgt',
                                                  'education',
                                                  'education-num',
                                                  'marital-status',
                                                  'occupation',
                                                  'relationship',
                                                  'race',
                                                  'gender',
                                                  'capital-gain',
                                                  'capital-loss',
                                                  'hours-per-week',
                                                  'native-country',
                                                  'salary'],na_values =' ?')

X = dataset.iloc[:, 0:14].values
y= dataset.iloc[:,-1].values
#X1= dataset.iloc[:, 0:14].values
#y1= dataset.iloc[:,-1].values
#
#plt.plot(X1,y1)
#X1=X1.toarray()
#plt.show()


from sklearn.preprocessing import Imputer
imp = Imputer()

X1[:,[0,2,4,10,11,12]]= imp.fit_transform(X1[:,[0,2,4,10,11,12]])

test= pd.DataFrame(X1[:,[1,3,5,6,7,8,9,13]])

test[0].value_counts()
test[1].value_counts()
test[2].value_counts()
test[3].value_counts()
test[4].value_counts()
test[5].value_counts()
test[6].value_counts()
test[7].value_counts()

test[0]= test[0].fillna(' private')
test[1] = test[1].fillna(' HS-grad')
test[2] = test[2].fillna(' Married-civ-spouse')
test[3] = test[3].fillna(' Prof-speciality')
test[4] = test[4].fillna(' Husband')
test[5] = test[5].fillna(' white')
test[6] = test[6].fillna(' Male')
test[7] = test[7].fillna(' United-States')

X1[:,[1,3,5,6,7,8,9,13]] = test

from sklearn.preprocessing import LabelEncoder #string to integr
lab = LabelEncoder()
  
X[:,1]= lab.fit_transform(X[:,1].astype(str))
X[:, 3] = lab.fit_transform(X[:, 3])
X[:, 5] = lab.fit_transform(X[:, 5])
X[:, 6] = lab.fit_transform(X[:, 6].astype(str))
X[:, 7] = lab.fit_transform(X[:, 7])
X[:, 8] = lab.fit_transform(X[:, 8])
X[:, 9] = lab.fit_transform(X[:, 9])
X[:, 13] = lab.fit_transform(X[:, 13].astype(str))

from sklearn.preprocessing import OneHotEncoder  #the coloms wich is change into integr forcefully
one= OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13],n_values=None)

X = one.fit_transform(X)
X = X.toarray()

from sklearn.preprocessing import StandardScaler # to piut all the data as same scale that is scaling
sc= StandardScaler()

X = sc.fit_transform(X)
y = lab.fit_transform(y)
lab.classes_

from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(X,y)
lin_reg.score(X,y)

plt.plot(X,y)
plt.show()

















