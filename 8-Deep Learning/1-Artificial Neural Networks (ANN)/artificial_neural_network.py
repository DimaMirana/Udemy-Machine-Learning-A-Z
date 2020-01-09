# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) #Grography
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) #Gender 
onehotencoder = OneHotEncoder(categorical_features = [1]) #dummy var for geography
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] #remove 1st col to remove dummy var trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling -> compulsory to ease the calculation so no var dominate other one
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras #build neural network using tensorflow
from keras.models import Sequential #sequential model ->initialize neural network
from keras.layers import Dense #dense -> require to build layers in ann

# We initialize our model using sequence of layers

# Initialising the ANN
classifier = Sequential()

#Step - 1 Dense will take care of innitialize weight
#Step - 2 each feature will go to one input node. here 11 IV rectifier fntn
#step - 3 pass Activation fntn to y sygmoid fntn

# Adding the input layer and the first hidden layer
#Dense ->weight , activation fntn, no of nodes we choose in the layer, no of input node
# output_dim -> 6 ->next layer have 6 node, init = uniform init the weight randomly close to zero activation->activation function,input_dim -> no of input var it add in the initialize step
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11)) 

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu')) 

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid')) 

# Compiling the ANN apply stochestic gradient descent in the ann
#optimizer -> algo to find the optimal set of weight here stochestic gradient descent adam is a kind of sgd 
#loss -> loss fntn need to optimize to find optimal set of weights if y= 2 category then loss - binary_crossentropy ,y>= 3 -> categorical_crossentropy
#metrics->criterion choose to evaluate the model use accuracy criterin to improves the models performance this perams expects a list so we write accuracy in a list
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) 

# Fitting the ANN to the Training set
#batch size -> no of observations after which you wanna update the weight np_epoch ->no of round the whole reaining set pass through the ann
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) # conver probability into true or false

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#accuracy -> the true values in the cm/total no of row to



























