# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# Inthis case we will not include the catagorical listing because this just 
# describes the hirearchy of the positions that has already been listed in level
# we do 1:2 instead of just 1 so the value will be considered a matrix
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
#  In an instance where we have such a small amount of data is doesn't make sense
# to split it up to train and test
"""om sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling'
# No feature scaling in Poly regression because it uses the same library as linear  regressor
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

""" Comparing the regression models """

# Linear regression model 
from sklearn.linear_model import LinearRegression
# Linreq is going to be the linear regression and the poly willbe lin_req 2
lin_reg = LinearRegression()
lin_reg.fit(X, y) 

# Polynomial regression model 
# import a new class that will allow us to use polynomial regression terms
from sklearn.preprocessing import PolynomialFeatures 
# poly_reg will transform the matrix of features of x^1, X^2, x^3 however many 
# we want to use using the 'regree' funnction 
poly_reg = PolynomialFeatures(degree = 4)
# We are using fit_transform it into a poly of X
X_poly = poly_reg.fit_transform(X)
# We see that poly_req automatically includes the constant 0 so we dont have to add
# it
# not we just need to fit it to a model 
# second lenear regression object (not len_reg) so we can tell the difference 
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)


# Visualize the LR
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title("Turth or bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualize the POLYNOMIAL REGRESSION 
# to make it more exact we increment by 1 withint the min and max of X
X_grid = np.arange(min(X), max(X), 0.1)
# Since this will create a vector so we will make it a matrix
X_grid = X_grid.reshape((len(X_grid), 1))
# Plot it 
plt.scatter(X, y, color = 'red')
# Here we cant just change to lin_reg2 but this is still an obejct of the LINEAR
# regression class So we need to add something to get the right predictions 
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title("Turth or bluff (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Predicint a new result with Linear Regression
# we are going to do this for a specific salary level
lin_reg.predict(6.5)

# Predicting a new result with polynomial regression
lin_reg2.predict(poly_reg.fit_transform(6.5))
