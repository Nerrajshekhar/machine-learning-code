import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


m=100
X=6*np.random.randn(m,1)-3
y=0.5*X**2+X+2+np.random.randn(m,1)

plt.scatter(X,y)
plt.axis([-7,7,0,9])
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2, include_bias= False)
X_poly=poly.fit_transform(X)


from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X_poly,y)

X_new=np.linspace(-7,7,100).reshape(-1,1)
X_poly_new=poly.fit_transform(X_new)
y_new=lin_reg.predict(X_poly_new)

plt.scatter(X,y)
plt.plot(X_new,y_new, c="r")
plt.axis([-7,7,0,9])
plt.show()