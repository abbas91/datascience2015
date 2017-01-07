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




# Fitting Model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0) # C {penality parameter}
lr.fit(X_train_std, y_train)

lr.predict_proba(X_test_std[0,:]) # P() of predict on one sample
" array([[ 0.000, 0.063, 0.937 ]]) " # three classes p()s












# SGD implementation 
from sklearn.linear_model import SGDClassifier
svm = SGDClassifier(loss='log') # ??

























