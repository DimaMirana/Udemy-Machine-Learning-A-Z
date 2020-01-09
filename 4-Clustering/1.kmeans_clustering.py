# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values
#y = dataset.iloc[:, 3].values we don't know what to look for

# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of cluster')
plt.ylabel('wcss')
plt.show()

# Applying data means to the mall dataset
kmeans = KMeans(n_clusters = 5, init='k-means++',max_iter=300,n_init=10,random_state = 0)
y_kmneans = kmeans.fit_predict(X) #fit the data and set the data into the cluster it belongs

# Visualising the clusters
plt.scatter(X[y_kmneans ==0, 0],X[y_kmneans ==0 , 1],s= 100,c = 'red', label = 'careful')
plt.scatter(X[y_kmneans ==1, 0],X[y_kmneans ==1 , 1],s= 100,c = 'blue', label = 'standard')
plt.scatter(X[y_kmneans ==2, 0],X[y_kmneans ==2 , 1],s= 100,c = 'green', label = 'Target')
plt.scatter(X[y_kmneans ==3, 0],X[y_kmneans ==3 , 1],s= 100,c = 'cyan', label = 'careless')
plt.scatter(X[y_kmneans ==4, 0],X[y_kmneans ==4 , 1],s= 100,c = 'magenta', label = 'sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'centroids')
plt.title('Cluster of clients')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()















