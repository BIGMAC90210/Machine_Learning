# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Data.csv')
# dataset = dataset[, 2:3]

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
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])
