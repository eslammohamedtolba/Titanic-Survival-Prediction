# Titanic-Survival-Prediction
This is a simple Titanic survival prediction model that uses a logistic regression algorithm to predict whether passengers on the Titanic survived or not based on various features.

## Prerequisites
Before running the code, make sure you have the following dependencies installed:
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Numpy

## Overview of the Code
1-Load the Titanic dataset and display basic information about it.
2-Data Cleaning:
- Check for missing values.
- Remove the 'Cabin' column due to a high number of missing values.
- Fill missing values in the 'Age' and 'Embarked' columns with the mean and mode, respectively.
  
3-Data Visualization:
- Visualize the count of passengers who survived.
- Visualize the count of passengers by 'Embarked' port.
- Visualize the count of passengers with different numbers of parents/children ('Parch').
- Visualize the count of passengers in each passenger class ('Pclass').
- Visualize the count of male and female passengers.
- Plot the distribution of 'Age' and 'Fare' columns.
- Find relations between 'Sex' and 'Survived', 'Age' and 'Survived', and 'Pclass' and 'Survived'.

4-Data Preprocessing: Label encode textual columns 'Sex' and 'Embarked' into numeric values.
5-Split the dataset into input features (X) and labels (Y). Remove unnecessary columns.
6-Split the data into training and testing sets using a 60/40 split.
7-Create a logistic regression model, train it on the training data, and predict on both the training and test data.
8-Evaluate the model's accuracy on the training and test data.
9-Create a predictive system to predict survival for a new set of input data.

## Model Accuracy
The model has achieved an accuracy of 80% on the test data.

## Contribution
Contributions to this project are welcome. You can help improve the accuracy of the model, add new features, or enhance the data preprocessing and visualization steps.
Please feel free to make any contributions and submit pull requests.


