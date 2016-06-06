# Ensemble Model # - combine different classifiers
# ---- Majority Vote




# [1] [Stacking] - all model use the same Train set

# ---------------------------- MajorityVoteClassifier --------------------------------- #

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator

class MajorityVoteClassifier(BaseEstimator,
                             ClassifierMixin):
    def __init__(self, classifiers,
                 vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for
                                  key, value in
                                  _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights
        
    def fit(self, x, y):
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(x,
                                       self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self
    
    def predict(self, x):
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(x),
                                 axis=1)
        else:
            predictions = np.asarray([clf.predict(x)
                                      for clf in
                                      self.classifiers_]).T
            
            maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                                           axis=1,
                                           arr=predictions)
            maj_vote = self.lablenc_.inverse_transform(maj_vote)
            return maj_vote
        
    def predict_proba(self, x):
        probas = np.asarray([clf.predict_proba(x)
                             for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba
    
    def get_params(self, deep=True):
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out
        
# ---------------------------------------------------------------------------------------- #
# -------------------------- Usage
########### Loading DataSet #############
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import MajorityVoteClassifier

iris = datasets.load_iris()
x, y = iris.data[50:, [1,2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)

import os
os.chdir('/Users/mli/Desktop/Ensmble_model')
os.getcwd()

########### Split Data ################
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.5, random_state=1)

########### Simulation ################
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np
clf1 = LogisticRegression(penalty='l2', C=0.001, random_state=0)
clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')

pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])

mv_clf = MajorityVoteClassifier.MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
clf_labels = ['Logistic regression', 'Descision Tree', 'KNN', 'Majority Vote']
all_clf = [pipe1, clf2, pipe3, mv_clf]

for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=Y_train,
                             cv=10,
                             scoring='roc_auc')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

#################### Tuning the ensemble model ##############################
mv_clf.get_params()
" Idnentify the paras for each learner "
from sklearn.grid_search import GridSearchCV
params = {'decisiontreeclassifier_max_depth': [1, 2],
          'pipeline-1_clf_C': [0.001, 0.01, 100.0]}
grid = GridSearchCV(estimator=mv_clf,
	                param_grid=params,
	                cv=10,
	                scoring='roc_auc')
grid.fit(X_train, Y_train)
grid.best_params # best paras























# [2] - [Bagging] - All model use different bootstrap sample trainset
# --- Effective to reduce variance, ineffective in reducing bias (Choose learners with low bias)

from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
tree = DecisionTreeClassifier(criterion='entropy', 
	                          max_depth=None, # use unpured tree
	                          random_state=1)
bag = BaggingClassifier(base_estimator=tree, 
	                    n_estimators=500,
	                    max_samples=1.0,
	                    max_features=1.0,
	                    bootstrap=True,
	                    bootstrap_features=False,
	                    n_jobs=1,
	                    random_state=1)

bag = bag.fit(X_train, Y_train)
Y_train_pred = bag.predict(X_train)
Y_test_pred = bag.predict(X_test)
bag_train = accuracy_score(Y_train, Y_train_pred) # train accuracy score
bag_test = accuracy_score(Y_test, Y_test_pred) # test accuracy score











# [3] - [AdaBoosting] - Iterate to train model on the 'error' datasets
# ---- effective on lower bias, tends to overfit (High variance)
# classifier1 (error) -> reassign weights to error -> (full train new weights) -> classifier2 ...
# majority vote : {classifier1 - n}

from sklearn.ensemble import AdaBoostClassifier
tree = DecisionTreeClassifier(criterion='entropy',
	                          max_depth=None,
	                          random_state=0)
ada = AdaBoostClassifier(base_estimator=tree,
	                     n_estimators=500,
	                     learning_rate=0.1,
	                     random_state=0)
ada = ada.fit(X_train, Y_train)
Y_train_pred = ada.predict(X_train)
Y_test_pred = ada.predict(X_test)
ada_train = accuracy_score(Y_train, Y_train_pred) # train accuracy score
ada_test = accuracy_score(Y_test, Y_test_pred) # test accuracy score























































































































































