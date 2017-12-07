# Random Forest Regression


# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting the Regression Model to the dataset
# Create your regressor here
# install.packages('randomForest')
# X is the independent which we provide via the dataset
# y is expecting a vector though so we have to use '$' then the name
library(randomForest)
set.seed(1234)
regressor = randomForest(x = dataset[1],
                         y = dataset$Salary,
                         ntree = 500)
# In the above we alter ntree for number of trees 

# Predicting a new result
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualising the Random forest Regression Model results (for higher resolution and smoother curve)
# install.packages('ggplot2')
# Need to make it in high resolution as this is a continuous regression model 
library(ggplot2)
# If lines are not verticle increase the resolutionX
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Random Forest Model)') +
  xlab('Level') +
  ylab('Salary')
