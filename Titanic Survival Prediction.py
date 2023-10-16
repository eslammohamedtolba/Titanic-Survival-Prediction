# import required dependencies
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np


# loading the dataset
Titanic_dataset = pd.read_csv("train.csv")
# show the first five rows of dataset
Titanic_dataset.head()
# show the last five rows of dataset
Titanic_dataset.tail()
# show dataset shape
Titanic_dataset.shape
# show some statistical info about the dataset
Titanic_dataset.describe()



# Check if there is any none(missing) values in the dataset to decide if will make a data cleaning or not
Titanic_dataset.isnull().sum()
# Remove Cabin column because it has a lot of missing values that are more than the half of number of rows
Titanic_dataset = Titanic_dataset.drop(columns=['Cabin'],axis=1)
# Replacing all none(missing) values in Age column with the column mean
Titanic_dataset['Age'].fillna(Titanic_dataset['Age'].mean(),inplace=True)
# Replacing all none(missing) values in Embarked column with the column mode
Titanic_dataset['Embarked'].fillna(Titanic_dataset['Embarked'].mode()[0],inplace=True)



# count all values in Survived columns and its repetition and plot it
plt.figure(figsize=(5,5))
Titanic_dataset['Survived'].value_counts()
sns.countplot(x ='Survived',data = Titanic_dataset)
plt.show()
# count all values in Embarked columns and its repetition and plot it
Titanic_dataset['Embarked'].value_counts()
sns.countplot(x ='Embarked',data = Titanic_dataset)
plt.show()
# count all values in Parch columns and its repetition and plot it
Titanic_dataset['Parch'].value_counts()
sns.countplot(x ='Parch',data = Titanic_dataset)
plt.show()
# count all values in Pclass columns and its repetition and plot it
Titanic_dataset['Pclass'].value_counts()
sns.countplot(x ='Pclass',data = Titanic_dataset)
plt.show()
# count all values in Sex columns and its repetition and plot it
Titanic_dataset['Sex'].value_counts()
sns.countplot(x ='Sex',data = Titanic_dataset)
plt.show()
plt.figure(figsize=(5,5))
# plot the distribution of Age column
sns.distplot(Titanic_dataset['Age'],color = 'red')
plt.show()
# plot the distribution of Fare column
sns.distplot(Titanic_dataset['Fare'],color= 'blue')
plt.show()
# find relation between Sex and Survived columns
plt.figure(figsize=(7,7))
sns.countplot(x = 'Sex',hue = 'Survived',data = Titanic_dataset)
# find relation between Sex and Survived columns
plt.figure(figsize=(40,20))
sns.countplot(x = 'Age',hue = 'Survived',data = Titanic_dataset)
# find relation between Pclass and Survived columns
plt.figure(figsize=(7,7))
sns.countplot(x = 'Pclass',hue = 'Survived',data = Titanic_dataset)



# Make a labelencoding by converting textual columns into numeric columns
Titanic_dataset.replace({'Sex':{'male':1,'female':0},'Embarked':{'S':0,'C':1,'Q':2}},inplace=True)



# split dataset into input and label data
X = Titanic_dataset.drop(columns=['Survived','PassengerId','Name','Ticket'],axis=1)
Y = Titanic_dataset['Survived']
print(X)
print(Y)
# split data into train and test data
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.6,random_state=2)
print(X.shape,x_train.shape,x_test.shape)
print(Y.shape,y_train.shape,y_test.shape)



# create the model and train it
LRModel = LogisticRegression()
LRModel.fit(x_train,y_train)
# Make model predict train and test input data
predicted_train_data = LRModel.predict(x_train)
predicted_test_data = LRModel.predict(x_test)
# avaluate the model accuracy
accuracy_train = accuracy_score(predicted_train_data,y_train)
accuracy_test = accuracy_score(predicted_test_data,y_test)
print(accuracy_train,accuracy_test)



# Make a predictive system
input_data = (3,0,31,1,0,18,0)
# convert input data into 1D numpy array
input_array = np.array(input_data)
# convert 1D input array into 2D
input_2D_array = input_array.reshape(1,-1)
# predict the output
if LRModel.predict(input_2D_array)[0]==1:
    print("this person survived")
else:
    print("this person didn't survive")




