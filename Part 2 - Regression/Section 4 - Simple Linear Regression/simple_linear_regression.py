# Simple Linear Regression

# Importing the libraries
import numpy as np
# visualize the graph using the below library 
import matplotlib.pyplot as plt
import pandas as pd
import csv


# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
# X is the indepedent variable which is number of years
# Y is the dependent so its the salary
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the training and test set
# we are going to import the cross validation library
from sklearn.cross_validation import train_test_split 
# now we build the training and test sets 14HBLmX1xEQmzNCy843qE3YkWUZAfWj2yX
# under train_test_split we first put the independent vriables 'x' in this case
# then we do the dependant variable 'y' in this case
# then we set the test size so .5 would be half goes to training set and half to test
# test size is usualy 20-25% 
# so here we will have two observations in test set and 8 in the training set 
# random_state is included for the course so we have the same logical results 
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = .33333, random_state = 0)
# so we build a machine learning model by building relatioship from the independent model to the dependent model 
# then we test if the model can be applied or corelated on the test set 


# FEATURE SCALLING 
# most of the libraries we use will take care of FEATURE SCALLING
# for simple linear regression the above is the case
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# fitting SLR to training set 
# from the above we import the linear regression class
from sklearn.linear_model import LinearRegression 
# do finish it with () because it will return a function that is the object 
# of itself
regressor = LinearRegression()
# Then we make an object out of this class that will be the SLRegressor 
# To fit it we will use a `method` in this case the `fit` method 
# fit the dependent then the indepeendent of the training data 
regressor.fit(X_train, y_train)
# If we run the above three commands it will create SLR and fit the data
# to the training by predicting the dependent variable based on the independent 
# So we have learned the corelations on the test set and trained the program

# Predicting the Test set results 
y_pred = regressor.predict(X_test)
# so y_test are the real salaries 
# y_pred and the predicted salaries based on SLR model 

# Now we visualize the results 
# We are going to call on the scatter class on the training independent 
# Then we add the y axis of the real salaries which is y_train 
# and we add custom collors 
plt.scatter(X_train, y_train, color  = 'red')
# Now we need the predictions of the trainin sets
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary Vs Experience (Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
# Specify this is the end of the graph and its ready to plot 
plt.show()

# Now do test set 
plt.scatter(X_test, y_test, color  = 'red')
# Mpt dp we cjamge tjos tp X_test?
# NO. Because out model is already trained on the training set 
# If we trained again we would create a whole new set of datapoints
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary Vs Experience (Test Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
# Specify this is the end of the graph and its ready to plot 
plt.show()



