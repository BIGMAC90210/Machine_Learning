# Data Preprocessing TEMPLATE

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# taking care of missing values 
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# not we fit the obect to matrix x 
imputer = imputer.fit(X[:, 1:3])
# replace the missing data with the mean 
X[:, 1:3] = imputer.transform(X[:, 1:3])


# Encoding Catagorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# calling labelencoder class with no arguments passed as 
labelencoder_X = LabelEncoder()
# now we are using this object on the desired variable 
# and we are assigning the values to each 
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# but the issue is still the same. 
# since 1 is > 0 and 2>1 the program will think that these countries are of higher vlaue
# so we have to prevent the machine from thinking this via dummy variables 
# We have to specify the index that is the column of the catagorical which is `0`
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Now we take care of the purchased column 
 # We will just need to use label encoder because it is the dependant variable 
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

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
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# fit transformation to data then transform it 
# We fit the training data so we can build the model off of it 
X_train = sc_X.fit_transform(X_train)
# Perform standardization by centering the scaling 
# x = x - mu (which is the mean) / standard deviation
# the below is the same since we already fit the model we can just tranform 
# We do not fit the test data as we want to predict 
X_test = sc_X.transform(X_test)









