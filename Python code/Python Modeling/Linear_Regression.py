# Linear Regression

##########################################
#                                        #
#                                        #
#               E   D   A                #
#                                        #
#                                        #
##########################################

# Create Scatter plot 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', context='notebook')
cols = ["col1", "col2", "col3", "col4"]
sns.pairplot(df[cols], size=2.5)
plt.show()

" --- Scatter plot with bar plot for each var --- "

sns.reset_orig() # reset seaborn 


# Create correlation plot
import numpy as np
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
cols = ["col1", "col2", "col3", "col4"]
hm = sns.heatmap(cm,
	             cbar=True,
	             annot=True,
	             square=True,
	             fmt='.2f',
	             annot_kws={'size': 15},
	             yticklabels=cols,
	             xticklabels=cols)
plt.show()



# Estimate coefficient of a regression model
from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(x, y)
slr.coef_ # all parameters
slr.intercept_ # intercept

# --- Normal equation
" P - 290 "





# Fitting an robust regression model using RANSAC (Robust Methods)
" Select a random number of samples to be inliers and fit model "
" Test all other data points with trainned model and add those meets defined torlance as inliners "
" Re-fit the model with all inliers "
" Validate the error of that model / back to first if not meet requirement "

from sklearn.linear_model import RANSACRegressor
ransac = RANSACRegressor(LinearRegression(),
	                     max_trials=100,
	                     min_samples=50,
	                     residual+metric=lambda x: np.sum(np.abs(x), axis=1),
	                     residual_threshold=5.0,
	                     random_state=0)
ransac.fit(x, y)



# How to evaluate a regression model

# -- Residual plots -> non-linearity / outliers (Nonlinear pattern? center by y = 0?)
plt.scatter(Y_train_pred, Y_train_pred - Y_train, 
	        c='bule', marker='o', label='Training data')
plt.scatter(Y_test_pred, Y_test_pred - Y_test, 
	        c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.show()


# -- Use MSE(Mean Squared Error)
from sklearn.metrics import mean_squared_error
mean_squared_error(Y_train, Y_train_pred) # Trainning MSE
mean_squared_error(Y_test, Y_test_pred) # Test MSE

# -- R square
from sklearn.metrics import r2_score
r2_score(Y_train, Y_train_pred)
r2_score(Y_test, Y_test_pred)




# Use regularization methods for regression (Tackle overfitting)
" L2 - X^2 (Shrinking to small values / Not 0) "
" L1 - |X| (Shrinking to 0) "


# Ridge Regression
" L2 "
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)


# LASSO regression
" L1 "
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.0)


# Elastic Net
" L1 + L2 "
from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet(alpha=1.0, l1_ratio=0.5) # if l1 = 1.0 = Lasso








# Polynomial Regression - non-linear
" Add powered terms in equation "
from sklearn.preprocessing import PolynomialFeatures
lr = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X_train)

lr.fit(X_quad, Y_train)
Y_quaud_fit = lr.predict(quadratic.fit_transform(X_test))







# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X_train, Y_train)



# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=1000,
	                           criterion='mse',
	                           random_state=1,
	                           n_jobs=-1)
forest.fit(X_train, Y_train)
Y_train_pred = forest.predict(X_train)
Y_test_pred = forest.predict(X_test)
























































































































