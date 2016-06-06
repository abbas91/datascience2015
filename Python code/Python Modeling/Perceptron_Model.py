# - Perceptron Model - #

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

# fitting the model
from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0) # define model object
ppn.fit(X_train_std, y_train) # fit the object with data

# Predict on test with object
y_pred = ppn.predict(X_test_std)
print('Misclassification: %d' % (y_test != y_pred).sum())

# Can load more metrics from sklearn
from sklearn.metrics import accuracy_score
print('Accuraycy: %.2f' % accuracy_score(y_test, y_pred))

# plot decision boundaries - P53-54








# SGD implementation 
from sklearn.linear_model import SGDClassifier
svm = SGDClassifier(loss='perceptron') # ??





