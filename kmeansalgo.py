# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 12:12:28 2019

@author: nssingh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_blobs
x,y=make_blobs(n_samples=300, centers=5,cluster_std=0.8)

plt.scatter(x[:,0],x[:,1])
plt.show()

wcv=[]

from sklearn.cluster import KMeans
for i in range(1,16):
    km=KMeans(n_clusters=i)
    km.fit(x)
    wcv.append(km.inertia_)
plt.plot(range(1,16),wcv)
plt.show()


km=KMeans(n_clusters=5)
y_predict=km.fit_predict(x)

plt.scatter(x[y_predict==0,0],x[y_predict==0,1])
plt.scatter(x[y_predict==1,0],x[y_predict==1,1])
plt.scatter(x[y_predict==2,0],x[y_predict==2,1])
plt.scatter(x[y_predict==3,0],x[y_predict==3,1])
plt.scatter(x[y_predict==4,0],x[y_predict==4,1])
plt.show()
import graphviz

pip install