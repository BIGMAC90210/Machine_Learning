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
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling'
# No feature scaling in Poly regression because it uses the same library as linear  regressor
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting the regressor model tothe dataset

# Predicting a new result with polynomial regression
y_pred = regressor.predict(6.5)

# Visualizing the results 
# Here we cant just change to lin_reg2 but this is still an obejct of the LINEAR
# regression class So we need to add something to get the right predictions 
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title("Turth or bluff (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


# HIGHER RESOLUTION Visualizing the results 
# Here we cant just change to lin_reg2 but this is still an obejct of the LINEAR
# regression class So we need to add something to get the right predictions 
# to make it more exact we increment by 1 withint the min and max of X
X_grid = np.arange(min(X), max(X), 0.1)
# Since this will create a vector we will make it a matrix
X_grid = X_grid.reshape((len(X_grid), 1))
# Plot it 
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title("Turth or bluff (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
