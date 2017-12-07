# Multiple Linear Regression

# Importing the dataset
dataset = read.csv('50_Startups.csv')

# Encoding categorical data
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling is not necessary as it will be taken care of by the regressor we use
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Multiple Linear Regression to the Training set
# we will always create a regressor and fit a prediction variable 
# instead of listing all independent variables we can use "." in Prodit ~ to include all
regressor = lm(formula = Profit ~ .,
               data = training_set)
# to look at regressor run              summary(regressor)
# notice that R made dummy variables for state and automatically removed one 
# at this point we see that R.D.Spend has the highest significance so we could change to Propfile ~ R.D.Spend
# and turn it into single linear regression

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)
# predicted profits of 10 observations. so take the output in console and compare with test_set profit 

# building an optimal team for the model using backward elimination
# for State we don't have to create dummy becuase we already set that above 
# We are going to use dataset instead of training_set here to have complete info about which 
#independent variables are significant 
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)
summary(regressor)
# Now we remove independent variables to remove the unnecessary bias
# Here we find that state2 and state3 has no effect on the dependent 
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)
summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)
summary(regressor)
# Now we have that we are at a 6% significant level with the "." denoting
regressor = lm(formula = Profit ~ R.D.Spend,
               data = dataset)
summary(regressor)







