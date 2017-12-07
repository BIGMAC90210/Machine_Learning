# Data Preprocessing TEMPLATE

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

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
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.2, random_state = 0)
# so we build a machine learning model by building relatioship from the independent model to the dependent model 
# then we test if the model can be applied or corelated on the test set 


# FEATURE SCALLING 
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""










