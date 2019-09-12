# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 00:11:15 2019

@author: nssingh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('bank-additional1.csv')

X=dataset.iloc[:,:20].values
y=dataset.iloc[:,-1].values

