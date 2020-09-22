# Machine-Learning-Project
House-votes dataset (the task is to predict whether the voter is a republican or a democrat based on their votes; the data description
file can be found below). It has 16 binary attributes and 2 classes.
Estimate the accuracy of Naive Bayes algorithm using 5-fold cross validation on the house-votes-84 data set. 
Estimate the precision, recall, accuracy, and F-measure (You need to choose the appropriate options for missing values).

#PROCEDURE FOR K-CROSS VALIDATION
1. Shuffle the dataset randomly.
2. Split the dataset into k groups
3. For each unique group:
Take the group as a hold out or train data set
Take the remaining groups as a test data set
4. Fit a model on the training set and evaluate it on the test set
5. Retain the evaluation score and discard the model
6. Summarize the skill of the model using the sample of model evaluation scores

#PROCEDURE
Importing the csv file in Jupyter notebook
Cleaning the dataset by replacing missing values with its mode
Shuffling the rows of the dataset 
Splitting the shuffled data set in 80:20 
Estimating the probability of Class Names
Implemented Naïve Bayes algorithm using test train datasets and its probabilities
Implemented confusion matrix using the predicted values
Lastly computed the accuracy, precision, recall and f-measure using the confusion matrix
