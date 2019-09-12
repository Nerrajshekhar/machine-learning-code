import numpy as num
import matplotlib.pyplot as plt
import pandas as pd

#from sklearn.datasets import fetch_mldata
#dataset=fetch_mldata('MNIST original')
from sklearn.datasets import fetch_mldata
dataset=fetch_mldata('MNIST original')

X= dataset.data
y= dataset.target

some_digit = X[66969]
some_digit_image=some_digit.reshape(28,28)

plt.imshow(some_digit_image)
plt.show()

from sklearn.tree import DecisionTreeClassifier
dtf=DecisionTreeClassifier(max_depth=10)
dtf.fit(X,y)

dtf.score(X,y)

dtf.predict(X[[17,2007,34005,46007,56009,69076],0:784])

from sklearn.tree import export_graphviz

export_graphviz(dtf,out_file="tree.dot")

import graphviz
with open("tree.dot") as f:
    dot_graph= f.read()
graphviz.Source(dot_graph)







dataset=pd.read_csv('housing.csv')


import seaborn as sns
corr_mat= dataset.corr()
sns.heatmap(corr_mat, annot=True)
pd.scatter_matrix(dataset)

