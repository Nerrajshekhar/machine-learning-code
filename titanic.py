# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 23:31:35 2019

@author: nssingh
"""

import numpy as num
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('titanic.csv')

import seaborn as sns
corr_mat=dataset.corr()
sns.heatmap(corr_mat,annot=True)
pd.scatter_matrix(dataset)