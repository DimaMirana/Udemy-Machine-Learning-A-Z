# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling //no need as the library takes care of that
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #regressor is the machine
regressor.fit(X_train, y_train) #regressior learn the co-relation between x and y

# Predicting the test set results
y_pred = regressor.predict(X_test)

#visualize the training test result
plt.scatter(X_train, y_train,color = 'red') #plot all the observation or real values
plt.plot(X_train,regressor.predict(X_train), color = 'blue') #the prediction of the X_train
plt.title('salary Vs experiance(Training Set)')
plt.xlabel('years of experiance')
plt.ylabel('salary')
plt.show()

#visualize the testing test result
plt.scatter(X_test, y_test,color = 'red') #plot all the test values
plt.plot(X_train,regressor.predict(X_train), color = 'blue') #alreday eqn is made in regressor using X_train already. Then if use the X_test then the exact line will formed using the new value
plt.title('salary Vs experiance(Testing Set)')
plt.xlabel('years of experiance')
plt.ylabel('salary')
plt.show()




