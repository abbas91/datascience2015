# Linear Regression


# load lib {numpy, sklearn}
from sklearn import datasets
import numpy as np

# loading datasets
isir = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target

# Spliting datasets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Scaling data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() # - define scaler object
sc.fit(X_train) # fit the object with data to get meansure
X_train_std = sc.transform(X_train) # scale data
X_test_std = sc.transform(X_test) # scale data



# KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski') # p = 1 {manhatten Dist} : p = 2 {Euclidean} 
knn.fit(X_train_std, y_train)

















