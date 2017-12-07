# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting the Linear Reression 
# f1 to look at the docs 
# add a "." because there is only one independent variable 
lin_reg = lm(formula = Salary ~ .,
             data = dataset)
# summary(lin_reg)

# Fitting Polynomial Regression 
# to the data set we add the polynomial features in a new dataset
# level 2 column will be created by raising the power of 2 of the numeric column
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
# We can continue to add degrees of regression 
poly_reg = lm(formula = Salary ~ .,
              data = dataset)
# summary(poly_reg)

# Visualizing the Linear Regression 
#install.packages('ggplot2')
# library(ggplot2)
geom_curve
ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level , y = predict(lin_reg, newdata = dataset)),
            colour = 'blue') +
  ggtitle("Truth or bluff (Linear Regression)") +
  xlab("Label") + 
  ylab("Salary")

# Visualizing the Polynomial Regression  
ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level , y = predict(poly_reg, newdata = dataset)),
            colour = 'blue') +
  ggtitle("Truth or bluff (Polynomial Regression)") +
  xlab("Label") + 
  ylab("Salary")

# predicting a new result with Linear regression 
# first input lin_req is the regressor 
# second point is a single observation point which we have to add 
# so we are gonna predict this on level 6.5 data frame 
y_pred = predict(lin_reg, data.frame(Level = 6.5))

# predicting a new result with Polynomial Regression
# we have to add each level for the regressions 
y_pred = predict(poly_reg, data.frame(Level = 6.5,
                                      Level2 = 6.5^2,
                                      Level3 = 6.5^3,
                                      Level4 = 6.5^4))


