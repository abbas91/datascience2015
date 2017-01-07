# Support Vector Machine


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
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=0) # linear kernel SVM / 'C' torlent to misclassification
svm = SVC(kernel='rbf', C=10.0, random_state=0, gamma=0.10) # Radius kernel SVM / 'C' torlent to misclassification / gamma: large over fit, small under fit

svm.fit(X_train_std, y_train)









# SGD implementation 
from sklearn.linear_model import SGDClassifier
svm = SGDClassifier(loss='hinge') # ??








