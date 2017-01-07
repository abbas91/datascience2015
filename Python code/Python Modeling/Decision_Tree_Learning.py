# Decision Tree Learning


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
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0) # impurity meansure // depth of tree
tree.fit(X_train, y_train)



# plot tree
" download 'http://www.graphviz.org' "

from sklearn.tree import export_graphviz
export_graphviz(tree,
	            out_file='tree.dot',
	            feature_name=['petal length', 'petal width']) # save to tree.dot

$ dot -Tang tree.dot -o tree.png # transform to png










# -- Random Forests
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy', # impurity meansure
	                            n_estimators=10, # learners
	                            random_state=1,
	                            n_jobs=2) # cores
forest.fit(X_train, y_train)











