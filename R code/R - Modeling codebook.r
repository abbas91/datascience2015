## Modeling

## >>>> Basic R Function Structure
#       > Loop
#       > Modeling Framework
...
## >>>> Dataset Spliting Methods
#       > Validationset (Hold-out)
#       > K-fold Validation
#       > LOOV Validation
#       > Bootstrap Sampling
...
## >>>> Machine learning Argolrithm
#       > Linear Regression 
#       > Logistic Regression 
#       > Neural Networks 
#       > Support Vector Machine 
#       > K-mean Clustering 
#       > Anomaly Detection 
#       > Collaborative Filtering
...
## >>>> Optimization Functions
#       > Batch Gradient Descent
#       > Stochastic Gradient Descent
#       > Min-batch Gradient Descent
...
## >>>> Validation Metrics





## ---------------------------{Basic R Function Structure}
## [1]Loop
# 
## [2]Modeling Framework <<<<<<<<<<1>>>>>>>>>>>>>
#
...
## ----------------------------{Dataset Spliting Methods}
## [1]Validationset (Hold-out)
##    <1> Sample from population set
          rate = round((nrow(Data[Data$Y == 1,]) / nrow(Data)),1) ## Keep population Y dist
          index1 = sample(1:(rate*5000),(rate*5000)) ## -- Ex.5000 sample size
          index2 = sample(1:((1-rate)*5000),((1-rate)*5000))
          sample.Y1 = Data[Data$Y == 1,][index1[1:(rate*5000)],]
          sample.Y2 = Data[Data$Y == -1,][index2[1:((1-rate)*5000)],]
          sample.data = rbind(sample.Y1,sample.Y2)
##    <2> Data Partition into Train / Validation / Test
          sample.data = data.frame(sample.data)
          index = sample(1:#all, #all) ##Randomlize original dataset
          train = sample.data[index[1:(5000*0.6)],] ##60%
          validation = sample.data[index[((5000*0.6)+1):(5000*0.8)],] ##20%
          test = sample.data[index[((5000*0.8)+1):5000],] ##20%
#
...
## [2]K-fold Validation
##    <1> Tuning
rmse_cv = function (n, train) { ## -> n parameter
      m = nrow(train)
      num = sample(1:10, m, replace=T) ## define the number of folds, Ex. 10 fold (Assign fold number for each observation)
      rmse = numeric(10) ## -> Create a container to store rmse Same number as folds
           for (i in 1:10) { ## -> for loop to iterating through each fold
               data.t = train[num!=i,]
               data.v = train[num==i,]
               model = model(Y...n, data=data.t) ## n is parameters / Select model
               pred = predict(model, newdata=data.v)
               rmse[i] = sqrt(mean((data.v$var - pred)^2)) ## Define other metrics if you want
               }
      return(mean(rmse))
}
      p = c(1,4,6,8,10) ## define "n" parameters
      rmse = sapply(p, rmse_cv, train) ## a vector n mean(rmse) for each parameter
##     <2> Testing
      n = which.min(rmse) ##Ex. use small one (Best)
      plot(1:n, rmse, type = 'b') ##Plot to eyeball
      final.model = model(Y~....n.., data=train) ##Use the best "n" with all train set
      pred = predict(final.model, newdata=test) ## Test on "Test set"
      rmse = sqrt(mean((test$var - pred)^2)) ## Test rmse
##     <3> Deployment
      final.model = model(Y~....n.., data=Data) ##Use all data
#
...
## [3]LOOV Validation

#
...
## [4]Bootstrap Sampling
##    <1> Tuning
rmse_bs = function(n, train) { ## -> n parameter
      m = nrow(train)
      rmse = numeric(10) ## -> Create a container to store rmse
          for (i in 1:10) { ## -> for loop to iterating 10 times
               index = sample(1:m, m, replace=T) ## create same size boostrap data set (Each time)
               data.t = train[index,]
               data.v = train[-index,]
               model = model(Y...n, data=data.t) ## n is parameters / Select model
               pred = predict(model, newdata=data.v)
               rmse[i] = sqrt(mean((data.v$var - pred)^2))
               }
      return(mean(rmse))
}
      p = c(1,4,6,8,10) ## define "n" parameters
      rmse = sapply(p, rmse_bs, train) ## a vector n mean(rmse) for each parameter
##     <2> Testing
      n = which.min(rmse) ##Ex. use small one (Best)
      plot(1:n, rmse, type = 'b') ##Plot to eyeball
      final.model = model(Y~....n.., data=train) ##Use the best "n" with all train set
      pred = predict(final.model, newdata=test) ## Test on "Test set"
      rmse = sqrt(mean((test$var - pred)^2)) ## Test rmse
##     <3> Deployment
      final.model = model(Y~....n.., data=Data) ##Use all data
#
...
## -----------------------------{Machine learning Argolrithm}
## [1]Linear Regression
##    <1> Pain code
##     - Hypothesis
       Prediction <- x %*% theta   
##     - Mean squared error cost function
       cost <- function(x, y, theta) {
               sum( (x %*% theta - y)^2 ) / (2*length(y))
       }	   
##     - Normal Equation
       normal.equaltion <- function (x,y) {
                           theta <- vector()
                           y.predict <- vector()
                           theta <- as.matrix(solve(t(x) %*% x) %*% (t(x) %*% y)) ##coefficients
                           prediction <- x1 %*% theta ##Predicted Y
                           x.list <- list(theta, prediction)
                            return (x.list)
                            }
##     - Gradient Descent(batch)
##       (1)squared error cost function
            cost <- function(x, y, theta) {
                    sum( (x %*% theta - y)^2 ) / (2*length(y))
                    }
##       (2)learning rate and iteration limit
            alpha <- 0.01
            num_iters <- 1000
##       (3)keep history
            cost_history <- double(num_iters)
            theta_history <- list(num_iters)
##       (4)initialize coefficients
            theta <- matrix(c(0,0), nrow=2)
##       (5)add a column of 1's for the intercept coefficient
            x <- cbind(1, matrix(x))
##       (6)gradient descent(batch)
            for (i in 1:num_iters) {
                 error <- (x %*% theta - y) ## each cost function
                 delta <- t(x) %*% error / length(y) ##derivative term
                 theta <- theta - alpha * delta ##update theta with learning rate * derivative term
                 cost_history[i] <- cost(x, y, theta) ##record the cost value of each iteration
                 theta_history[[i]] <- theta ##record each theta version after iteration
                 }
                 print(theta)
##        (7)Learning curve
             plot(cost_history, 
			      type='line', 
				  col='blue', 
				  lwd=2, main='Cost function', ylab='cost', xlab='Iterations')
##    <2> Package
##     - "ml"package
       regression <- lm(y ~ ., data = x)
	                 lm(y ~ x1+x2+x1*x2, data = x) # interaction term
					 lm(y ~ x1+x2+x2^2, data = x) # polynomal 
	   summary(regression) #general statistics
#
##    <3> RHadoop Packages
#
#
...
## [2]Logistic Regression
##    <1> Pain code
##     - Hypothesis
       Prediction <- (exp(x %*% theta)) / (1 + exp(x %*% theta)) 
##     - cost function
       cost <- function (x,y,theta) {
               sum(y %*% log((exp(x %*% theta)) / (1 + exp(x %*% theta)))
               + (1 - y) %*% log(1 - ((exp(x %*% theta)) / (1 + exp(x %*% theta))))) * (-(1/length(y)))
               }
##     - Gradient Descent(batch)
##       (1)logistic cost function
            cost <- function (x,y,theta) {
               sum(y %*% log((exp(x %*% theta)) / (1 + exp(x %*% theta)))
               + (1 - y) %*% log(1 - ((exp(x %*% theta)) / (1 + exp(x %*% theta))))) * (-(1/length(y)))
               }
##       (2)learning rate and iteration limit
            alpha <- 0.01
            num_iters <- 1000
##       (3)keep history
            cost_history <- double(num_iters)
            theta_history <- list(num_iters)
##       (4)initialize coefficients
            theta <- matrix(c(0,0), nrow=2)
##       (5)add a column of 1's for the intercept coefficient
            x <- cbind(1, matrix(x))
##       (6)gradient descent(batch)
            for (i in 1:num_iters) {
                 error <- (y %*% log((exp(x %*% theta)) / (1 + exp(x %*% theta)))
                       + (1 - y) %*% log(1 - ((exp(x %*% theta)) / (1 + exp(x %*% theta))))) ## each cost function
                 delta <- t(x) %*% error / length(y) ##derivative term
                 theta <- theta - alpha * delta ##update theta with learning rate * derivative term
                 cost_history[i] <- cost(x, y, theta) ##record the cost value of each iteration
                 theta_history[[i]] <- theta ##record each theta version after iteration
                 }
                 print(theta)
##        (7)Learning curve
             plot(cost_history, 
			      type='line', 
				  col='blue', 
				  lwd=2, main='Cost function', ylab='cost', xlab='Iterations')
##    <2> Package
##     - "glm"Package
       logistic <- glm(Y ~ ., data = x, family = "binomial")
	   	           glm(y ~ x1+x2+x1*x2, data = x, family = "binomial") # interaction term
				   glm(y ~ x1+x2+x2^2, data = x, family = "binomial") # polynomal 
	   summary(logistic) #general statistics
#
##    <3> RHadoop Packages
#
#
...
## [3]Neural Networks
##    <1> Pain code
#
##    <2> Package
#
##    <3> RHadoop Packages
#
...
## [4]Support Vector Machine
##    <1> Pain code
#
##    <2> Package
#
##    <3> RHadoop Packages
#
...
## [5]K-mean Clustering
##    <1> Pain code
#
##    <2> Package
#
##    <3> RHadoop Packages
#
...
## [6]Anomaly Detection
##    <1> Pain code
#
##    <2> Package
#
##    <3> RHadoop Packages
#
...
## [7]Collaborative Filtering
##    <1> Pain code
#
##    <2> Package
#
##    <3> RHadoop Packages
#




## ------------------------------{Optimization Functions}
## [1]Batch Gradient Descent
#
## [2]Stochastic Gradient Descent
#
## [3]Min-batch Gradient Descent
#
...
## -------------------------------{Validation Metrics}
## [1]
#
## [2]
#
## [3]
#
...



















