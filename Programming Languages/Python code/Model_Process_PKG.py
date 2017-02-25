##################################
#                                #
#                                #
#    Model Process for Python    #
#                                #
#                                #
##################################

# -------------- Model Process for Python

import sklearn as sk # Machine laerning libraries










--------------------------------------------------------------------------------------------------------
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>                                  >
>                                  >
>           scikit learn           >
>                                  >
>                                  >
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Python for Machine Learning - scikit learn

" http://scikit-learn.org/stable/install.html "


# Scikit-learn requires:
Python (>= 2.6 or >= 3.3),
NumPy (>= 1.6.1),
SciPy (>= 0.9).


# - install
$ pip install -U scikit-learn


# // Step 1
# ----------------------------------- # 
#                                     #
#           Package Loading           #
#                                     #
# ----------------------------------- #
import sklearn as sk # Machine Learning in Python











# // Step 2
# ----------------------------------- # 
#                                     #
#           Loading Data              #
#                                     #
# ----------------------------------- #















# // Step 3
# ----------------------------------- # 
#                                     #
#         Preprocessing Data          #
#                                     #
# ----------------------------------- #

# [1] deal with missing data
import pandas as pd
df.isnull().sum() # count missing by columns

# - Eliminate samples / features with missing value
df.dropna() # rows
df.dropna(axis=1) # columns
df.dropna(how="all") # only drop rows all columns are NaN
df.dropna(thresh=4) # drop rows that have not at least 4 non-NaN values
df.dropna(subset=['C']) # Only drrop rows where NaN appear in specific columns (here 'C')

# - Imputing the missing values
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='mean', axis=0) # initiate the imputor - also, median, most freq, etc
imr = imr.fit(df) # measure the data
imputed_data = imr.transform(df.values) # imputing the data



# [2] Handling categorical data
import pandas as pd
# - ordinal features
size_mapping = { # mapping the label to the values
	'label1': 3,
	'label2': 2,
	'label3': 1
}
df['ordinal_var'] = df['ordinal_var'].map(size_mapping) # transform the 'label' to values

# - target variable ('class')
import numpy as np
class_mapping = {label:idx for idx, label in
                 enumerate(np.unique(df['target_var']))} # mapping label to value
df['target_var'] = df['target_var'].map(class_mapping) # transform the 'label' to values

# - convert back to label >>
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['target_var'] = df['target_var'].map(inv_class_mapping)


# - in 'sklearn' default pkg
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['target_var'].values) # transform 'label' to value

class_le.inverse_transform(y) # transform from value to 'label'



# - nominal features (Dummy coding)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0]) # specify index of the target column
ohe.fit_transform(x).toarrary() # transform 'x' df / if not want 'sparse matrix', set '(..., sparse=False)' when initiate

# or
import pandas as pd
pd.get_dummies(df[['var1', 'var2', 'var3']]) # only convert the string columns to dummy vars



# [3] Feature scaling
# - normalization [0, 1]
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test) # use the same above

 
# - standardization [center - 0] - better for SVM, logic reg (initate coe with 0s)
from sklearn.preprocessing import StandardScaler
stdsc = standardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test) 






# [4] Feature selection / feature extraction

# -> feature selection

# (1)(With panalty L1, L2) for many irrelvant features dataset
# L1 - {abs|w|}; features deleted
# L2 - {sqrt{|w|}} weaken all features
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penality='l1', c=0.1) # or l2, c = lambda



# (2) Sequential feature selection 
# - SBS(Squential Backward Selection) - self-implement
# -------------------- class define ------------------------- #
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

class SBS():
	def __init__(self, estimator, k_features,
		scoring=accuracy_score,
		test_size=0.25, random_state=1):

		self.scoring = scoring
		self.estimator = clone(estimator)
		self.k_features = k_features
		self.test_size = test_size
		self.random_state = random_state
    
    def fit(self, x, y):
    	X_train, X_test, Y_train, Y_test = /
    	         train_test_split(x, y, test_size=self.test_size,
    	         	              random_state=self.random_state)

    	dim = X_train.shape[1]
    	self.indices_ = tuple(range(dim))
    	self.subsets_ = [self.indices_]
    	score = self._calc_score(X_train, Y_train,
    		                     X_test, Y_test, self.indices_)

    	self.score_ = [score]

    	while dim > self.k_features:
    		scores = []
    		subsets = []

    		for p in combinations(self.indices_, r=dim-1):
    			score = self._calc_score(X_train, Y_train,
    				                     X_test, Y_test, p)
    			scores.append(score)
    			subsets.append(p)


    		best = np.argmax(scores)
    		self.indices_ = subsets[best]
    		self.subsets_.append(self.indices_)
    		dim -= 1

    		self.scores_.append(scores[best])
    	self.k_score_ = self.scores_[-1]

    	return self

    def transform(self, x):
    	return x[:, self.indices_]

    def _calc_score(self, X_train, Y_train,
    	                  X_test, Y_test, indices):
        self.estimator.fit(X_train[:, indices], Y_train)
        Y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(Y_test, Y_pred)
        return score

# -------------------- end -------------------------------- #

# use
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, Y_train) # measure the features in X_train_std

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs,scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()

" see the plot .. high accuracy + simple "

k5 = list(sbs.subsets_[8]) # 5 features () - names
print(df.columns[1:][k5])
" the features kept "

knn.fit(X_train_std[:, k5], Y_train)
print('Training accuracy:', knn.score(X_train_std[:, k5], Y_train))
print('Test accuracy:', knn.score(X_test_std[:, k5], Y_test))



# (3) Random forest - feature importance 
from sklearn.ensemble import RandomForestClassifier
feat_labels = df.columns[1:]
forest = RandomForestClassifier(n_estimators=10000,
	                            random_state=0,
	                            n_jobs=-1)
forest.fit(X_train, Y_train)
importances = forest.feature_importances
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
	print("%2d) %-*s %f" % (f + 1, 30,
		                    feat_labels[indices[f]],
		                    importances[indices[f]]))
" features         importance"

x_selected = forest.transform(X_train, threshold=0.15) # if use it (% importance) to perform feature selection






# -> feature extraction
# [Data needs to be scaled first]

# (1) Principal Component Analysis (PCA) - linear separatible
# --- Evaluate Importance of PCA
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
eigen_vals = pca.explained_variance_ratio_ # Egan-values for each PCAs (Importance)
# --- Fitting Model with PCA
pca = PCA(n_components=2) # Only take first 2 PCs
lr = LogisticRegression()
X_train_pca = pca.fit_transform(X_train_std) # fit with trainset
X_test_pca = pca.transform(X_test_std) # only transform with testset
lr.fit(X_train_pca, Y_train)



# (2) Linear Discriminant Analsysi (LDA) - linear separatible
# --- Evaluate Importance of LDA
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
lda = LDA(n_components=None)
X_train_lda = lda.fit_transform(X_train_std, Y_train) # fit with trainset, (x, y) supervized
eigen_vals = lda.explained_variance_ratio_ # Egan-values for each LDAs (Importance)


# --- Fitting Model with LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, Y_train) # fit with trainset, (x, y) supervized
X_test_lda = lda.transform(X_test_std) # only transform with testset
lr.fit(X_train_lda, Y_train)


# (3) Kernel Principal Component Analysis (K-PCA) - non-linear separatible
from sklearn.decomposition import KernelPCA
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15) # can choose other kernel methods, 2 PCAs = features
X_skernpca = scikit_kpca.fit_transform(X_train_std)

# - Explore Visually (Separatible?)
# > Normal PCA
import matplotlib as plt
from sklearn.decomposition import PCA
scikit_pca = PCA(n_components=2) # only first 2 PCAs
X_spca = scikit_pca.fit_transform(X_train_std) 

# > K PCA
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_kspca = scikit_kpca.fit_transform(X_train_std)


# X_spca <=> X_kspca
# > ---- Plot
fig, ax = plt.subplot(nrows=1, ncols=2, figsize=(7,3))
ax[0].scatter(X_spca[y==0, 0], X_spca[y==0, 1],
	          color='red', marker='^', alpha=0.5)
ax[0].scatter(X_spca[y==1, 0], X_spca[y==1, 1],
	          color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_spca[y==0, 0], np.zeros((50,1))+0.02,
	          color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y==1, 0], np.zeros((50,1))-0.02,
	          color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.show()







# // Step 4
# ----------------------------------- # 
#                                     #
#         Modeling in progress        #
#                                     #
# ----------------------------------- #

# >>>>>>>>>>>>>>>>>>>>> Split Data Set
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
x, y = df.iloc[:, 1:].values, df.iloc[:, 0].values # split X, Y
le = LabelEncoder() # - Encode the target variable
y = le.fit_transform(y) # convert to numeric
le.transform(['label1', 'label2']) # check how it encoded

X_train, X_test, Y_train, Y_test = \
                       train_test_split(x, y, test_size=0.3, random_state=0) # test set %



# >>>>>>>>>>>>>>>>>>>>> Modeling 
from sklearn.preprocessing import StandardScaler # transformer 1
from sklearn.decomposition import PCA # transformer 2
from sklearn.linear_model import LogisticRegression # estimator 1
from sklearn.pipeline import Pipeline # modeling pipeline

# ----------------- Hold out model ------------------------------ #
pipe_lr = Pipeline([('scl', StandardScaler()), # transformer 1
                    ('pca', PCA(n_components=2)), # transformer 2
                    ('clf', LogisticRegression(random_state=1)) # estimator 1
	                ])
pipe_lr.fit(X_train, Y_train) # pass the data into the pipe
pipe_lr.score(X_test, Y_test) # default give accurate rate



# ----------------- Cross-Validation model ------------------------------ #
from sklearn.cross_validation import cross_val_score # CV object
scores = cross_val_score(estimator=pipe_lr,  # Use the completed pipeline
                         X=X_train,          # Feed X_train
                         y=Y_train,          # Feed Y_train
                         cv=10,              # Do 10 fold CV
                         n_jobs=1            # CPU used, -1 (use all)
	                     )
np.mean(scores) # will generate all scores for all fold, you can get mean to estimate general




# ----------------- Using Learning Curve --------------------------------- #
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
pipe_lr = Pipeline([
            ('scl', StandardScaler()),
            ('clf', LogisticRegression(penalty='12', random_state=0))
	        ])
train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr,
	                                                    X=X_train,
	                                                    y=Y_train,
	                                                    train_sizes=np.linspace(0.1, 1.0, 10), # Split train data into bins
	                                                    cv=10,
	                                                    n_jobs=1)
# Ploting
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
	     color='blue', marker='o',
	     markersize=5,
	     label='training accuracy')
plt.fill_between(train_sizes,
	             train_mean + train_std,
	             train_mean - train_std,
	             alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean,
	     color='green', linestyle='--',
	     marker='s', markersize=5,
	     label='validation accuracy')
plt.fill_between(train_sizes,
	             test_mean + test_std,
	             test_mean - test_std,
	             alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.0])
plt.show()





# ----------------- Using Validation Curve --------------------------------- #
import matplotlib.pyplot as plt
from sklearn.learning_curve import validation_curve
pipe_lr = Pipeline([
            ('scl', StandardScaler()),
            ('clf', LogisticRegression(penalty='12', random_state=0))
	        ])

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(estimator=pipe_lr,
	                                         X=X_train,
	                                         y=Y_train,
	                                         param_name='clf__C', # give the hyper-para a name to access the model object
	                                         param_range=param_range, # suppy the range for this para
	                                         cv=10)

# Ploting
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean,
	     color='blue', marker='o',
	     markersize=5,
	     label='training accuracy')
plt.fill_between(param_range,
	             train_mean + train_std,
	             train_mean - train_std,
	             alpha=0.15, color='blue')
plt.plot(param_range, test_mean,
	     color='green', linestyle='--',
	     marker='s', markersize=5,
	     label='validation accuracy')
plt.fill_between(param_range,
	             test_mean + test_std,
	             test_mean - test_std,
	             alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter XXXX')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.0])
plt.show()



# ----------------- Grid Search ------------------------------------- #
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
pipe_svc = Pipeline([
                     ('scl', StandardScaler()),
                     ('clf', SVC(random_state=1))
	                ])
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
param_grid = [
              {'clf__C': param_range,
              'clf_kernel': ['linear']},
              {'clf__C': param_range,
              'clf__gamma': param_range,
              'clf_kernel': ['rbf']}
             ]

gs = GridSearchCV(estimator=pipe_svc,
	              param_grid=param_grid, # search grid
	              scoring='accuracy', # score metrics
	              cv=10,
	              n_jobs=-1)
gs = gs.fit(X_train, Y_train)
gs.best_score_ # best paras

clf = gs.best_estimator_
clf.fit(X_train, Y_train)
clf.score(X_test, Y_test) # result


# ---------------- Grid Search with nested CV ----------------------- #
gs = GridSearchCV(estimator=pipe_svc,
	              param_grid=param_grid, # search grid
	              scoring='accuracy', # score metrics
                  cv=2, # internal cv
                  n_jobs=-1
	              )
scores = scross_val_score(gs, 
	                      X_train, Y_train, 
	                      scoring='accuracy',
	                      cv=5 # outter cv
	                      )
np.mean(scores) # accuracy rate
np.std(scores) # +- on accuracy rate




# -------------- Evaluation metrics ---------------------------------- #
# - Confusion Matrix
from sklearn.metrics import confusion_matrix
pipe_svc.fit(X_train, Y_train) # Use pipe to fit train
Y_pred = pipe_svc.predict(X_test) # Use to predict
confmat = confusion_matrix(y_true=Y_test, y_pred=Y_pred)
print confmat

# - Precision
from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(y_true=Y_test, y_pred=Y_pred)
# - Recall
recall_score(y_true=Y_test, y_pred=Y_pred)
# - F1 score
f1_score(y_true=Y_test, y_pred=Y_pred)


# --- use in cv, pipe
from sklearn.metrics import make_scorer, f1_score
scorer = make_scorer(f1_score, pos_label=0) # if want to change positive class (default = 1)
gs = GridSearchCV(estimator=pipe_svc,
	              param_grid=param_grid, # search grid
	              scoring=scorer, # use new metric
	              cv=10,
	              n_jobs=-1)


# - ROC AUC curve scores
pipe_lr = pipe_lr.fit(X_train, Y_train)
Y_pred = pipe_lr.predict(X_test)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
roc_auc_score(y_true=Y_test, y_score=Y_pred) # ROC AUC


# - Scoring for multi-class classification
# - Micro AVG of percision 
scorer = make_scorer(score_func=precision_score,
	                 pos_label=1,
	                 greater_is_better=True,
	                 average='micro') 
# - Macro AVG of percision
scorer = make_scorer(score_func=precision_score,
	                 pos_label=1,
	                 greater_is_better=True,
	                 average='macro') 












>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>                                  >
>                                  >
>           ????????????           >
>                                  >
>                                  >
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>










































