# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
# independent 
X = dataset.iloc[:, :-1].values
# Dependent is the profit 
y = dataset.iloc[:, 4].values

# change the catagorical variables 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
# After this we will have additional columns at the beggining that are going 
# to replace the states (dummy variables)

# Avoiding the dummy variable trap 
# So we are removing the first column from X
X = X[:, 1:]
#but our library is going to take care of it for us 


# Splitting the dataset into the Training set and Test set
# here we have 50 observations so well split it 10 and 40 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling is not necessary for MLR as library takes care of this for us 

# Creating the model fitting multiple independents 
from sklearn.linear_model import LinearRegression 
regressor= LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set result 
# 4 independent and 1 dependent means we would need 5 dimensions to graph
y_pred = regressor.predict(X_test)
# This gives us the predicted profits 

# Building the optimal model using backward elimination $ python -mpip install statsmodels

# we are going to determine and remove the insignificant variables so that 
# our team has independent variables that each variable has a great impact
# on the dependent 
import statsmodels.formula.api as sm 
# There is an X0 (x0 =1) as there is a B0 in the MLR formula but it is not in the library
# so we need to add this constant to our matrix because witho7ut it we have  no constant 
# we have to add an array of 50 liunes and 1 column which we can go through np
# Finally we have to convert it to interger type so we dont get an error 
# The we use axis to either run an row of these values or a column of these values 
# axis = 0 for row and axis = 1 for column 
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

# Backward elimination
# x_opt will contain a matrix of optimal variable 
# First we take all the columns and then pick them off one by one 
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# Now we need to select a significant level that we can apply to the variables 
#  Step 1 SL = .05
# Step 2 fit the model with all possible predictions 
# the privous one we used was the Linearregressor class then new one if OLS class
# endog is the dependent 
# exog is the array but the intercept is not included by default. Thats why we have the 
# x = np.append ... shtuph
# then we end it in .fit to fit the regressor to Odinary Leaast squares
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# STEP 3 Consider what has the highest P value.If > 5% then go to step 4. If below
# then we are done 
# regressor_OLS.summary()
# The lowe the P value the more significant the variable 
# So we remove x2 because its above the sig and so on and so on
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# Keep comparing the X_opt with x so you can see what the new index is and
# remove the appropriate data
# At this point index 2 is above 5%. this corresponds to index 4 on x matrix so we remove it
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# Here we get the second independent variable is above 5% 

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# This means that RD spent is the most powerful predictor of the profit 






