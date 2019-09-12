import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


X=np.random.random(100)
y=4*X+7+np.random.random(100)

plt.scatter(X,y)
plt.show()
X=np.c_[X,np.ones(100)]
 theta= np.linalg.inv(X.T@X)@(X.T@y)
 
# mat=np.array([[1,2],[3,4]])
# mat*mat
 #mat@mat
 
 dataset = pd.read_excel('dataset/blood.xlsx')
 X = dataset.iloc[2:,1].values
 y = dataset.iloc[2:,-1].values
 
 X= X.reshape(-1,1)
 
 plt.scatter(X,y)
 plt.show()
 
 from sklearn.linear_model import LinearRegression
 lin_reg=  LinearRegression()
 lin_reg.fit(X, y)
 
 lin_reg.score(X, y)
 
 plt.scatter(X, y)
 
 
 plt.plot(X, lin_reg.predict(X),c='r')
 plt.scatter(X, lin_reg.predict(X), c='r')
 plt.show()
 
 lin_reg.coef_
 lin_reg.intercept_
 
 lin_reg.predict([[26]])
 
 plt.plot()
 
