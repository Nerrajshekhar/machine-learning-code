# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:27:42 2019

@author: nssingh
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
dataset=load_iris()

X=dataset.data
y=dataset.target

from sklearn.svm import SVC
svm=SVC()
svm.fit(X,y)


svm.score(X,y)