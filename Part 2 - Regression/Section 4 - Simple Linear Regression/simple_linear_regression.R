# Simple Linear Regression

# Importing the dataset
dataset = read.csv('Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = .3333333333)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# simple linear regression pacakge takes care of feature scaling

# Fitting SLR to training data 
# The below notation means the salary is proportional to years experience
# second argument is what data we want to train the model 
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)
# Now we can execute      summary(regressor)  to find out more info 
# specifically the coefficients section. This is the statistical significance 
# '*' represent statistical significance. So we know that there will be a strong 
# linear relationship between the dependenct (salary) will have on independent (YearsExperience)
# P value is below 5% the independent is highly significant 
# Over it is less significant 

# Predict Test set results 
y_pred = predict(regressor, newdata = test_set)

# Visualizing the training set 
#install the ggplot2 library 
#library(ggplot2) '
# Now we plot using scatter plotting in geom by specifying x and y axis
# using aes the exastatic function
# We use the years of experience in the training and salary of training. So this is a plot 
# of all the real values 
# then after the second + we plot the regression line based on predicted salaries
# we dont use y_pred because its based on test_set
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') + 
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') + 
  ggtitle("Salary vs Experience (Training set)") +
  xlab("Years of experience") +
  ylab("Salary")

# Now we predict by applying to the test set 
# We already trained the line and don't want to build new predictions based on the new points
# so we leave the training sets in place 
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') + 
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') + 
  ggtitle("Salary vs Experience (Test set)") +
  xlab("Years of experience") +
  ylab("Salary")

