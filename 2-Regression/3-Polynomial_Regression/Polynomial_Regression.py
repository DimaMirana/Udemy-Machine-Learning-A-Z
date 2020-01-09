# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Not Splitting the dataset into Training set and Test set as there is not enough data
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

#Fitting the dataset into linear regrassion -for comparing the result with polynomial regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# fitting the polynomial regression - creating the polynomial model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) #turn X into X_poly-> x1 ,x2,...xn
X_poly = poly_reg.fit_transform(X) #X_poly have 3 column 1st col in the b0 = 1 
poly_reg.fit(X_poly, y)
lin_reg2 = LinearRegression() #include the X_poly into linear reg model
lin_reg2.fit(X_poly,y)

# Visualizing the linear regression results
plt.scatter(X,y,color = 'red')
plt.plot(X, lin_reg.predict(X),color = 'blue')
plt.title('Truth or bluff(Linear Regression:)')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

# Visualizing the polynomial regression results 
plt.scatter(X,y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color = 'blue')
plt.title('Truth or bluff(Polynomial Regression:)')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

# Visualizing the polynomial regression results with higher resolution
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color = 'red')
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color = 'blue')
plt.title('Truth or bluff(Polynomial Regression:)')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict([[6.5]]) #transform it into 2d array otherwise get error

# Predicting a new result with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))







