# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Data.csv')

# Taking care of missing data
# is.na checks to see if all the values in the specified column are missing
# na.rm to compute the missing value 
# the third ',' is for is the value is not missing
dataset$Age = ifelse(is.na(dataset$Age), ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)), dataset$Age)

# repeat for salary 
dataset$Salary = ifelse(is.na(dataset$Salary),ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),dataset$Salary)

# Encoding the country column
# 'c' is a vector that will contain the countries 
# then we choose the label we assign to the countries 
dataset$Country = factor(dataset$Country,
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1, 2, 3))

# Now we repeat    
dataset$Purchased = factor(dataset$Purchased,
                         levels = c('No', 'Yes'),
                         labels = c(0, 1))

# splitting dataset into training set 
# we has to install using         install.packages('caTools') 
# then we had to go to packages tag and select or we could     library(caTools)
# instead of using random state like in python we are going to do it manually through seed with a random  number 
set.seed(42)
# Now we split it 
# We use the $Purchased as this is the dependent variable 
# split ratio is the percentag ewe want to put in TRAINING set. We did 20% in python so we will do the same here)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# Now we can run `split` and see true means it goes to training set and false to the test set
# Now we can create the sets differently. Each set is a subset of the data set
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


#Feature Scaling 
# using the below causes issues because we transfomed the words/factors into numbers 
# training_set = scale(training_set)
# test_set = scale(test_set)
# so we are going to exclude these
training_set[, 2:3] = scale(training_set[, 2:3])
test_set[, 2:3] = scale(test_set[, 2:3])
