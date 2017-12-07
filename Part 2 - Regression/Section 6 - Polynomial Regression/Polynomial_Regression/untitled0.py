# Polynomial regression 

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

dataset = pd.read_csv('Position_Salaries.csv')
# position
X = dataset.iloc[:, 1:2]
# Salary 
y = dataset.iloc[:, 2]

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Linear Model 
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')

from sklearn.preprocessing import PolynomialFeatures 
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2 = lin_reg2.fit(X_poly, y)

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg2.predict(X_poly), color = 'blue')
plt.title('yoyoyy')
plt.xlabel('Level of job')
plt.ylabel('salary')
plt.show()
