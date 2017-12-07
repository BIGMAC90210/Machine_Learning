# Basic Prediction Model 

# import libraries 
# Handle arrays 
import numpy as np
# Analysis tools
import pandas as pd 
# plot 
import matplotlib as plt


dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Catagorical Variables 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Label encoder changes it into values 
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# one hot encoder transforms it into columns of 2 values 
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.cross_validation import train_test_split

# time to split the data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 0)

# Scaling the features 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
