
# ========================================================================
#  StandardScaler now works only on 2d arrays. It can be handled in a number of ways, for example we can define y as 1d array and then reshape it, or And we can change y object to be defined as a 2d array, but then we have another problem: ML methods require for dependent (output) variable to be a 1d array, so we are to do another reshaping to apply SVR method
# =======================================================================

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
# defining y as 2d array
y = dataset.iloc[:, 2:3].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Fitting SVR to the dataset
from sklearn.svm import SVR
# creating the regressor
regressor = SVR(kernel = 'rbf')
# here we need change y to 1d array
regressor.fit(X, y.reshape(-1, 1))
# ========================================================================
# Now we face a different problem (thanks to updates again): current .predict() 
# method requires an object of the same shape (dimensions) as was X object 
# which was used to fit our model. So we create a matrix of of number: 
# np.array([[6.5]]). 
# ========================================================================
# Predicting a new scaled result data value instead of the fitted values done above 
# we transform 
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))
# now we predict based on the UNScalled value 
# predict the scaled result and convert to an actual 
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))


# show the result
print(y_pred)

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
print(y_pred)
# since the CEO is so far from the data points the data point is considered 
# an outlier 


