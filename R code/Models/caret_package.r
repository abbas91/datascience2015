site url >>> "http://topepo.github.io/caret/bytag.html"


# [1] Visualization

library(AppliedPredictiveModeling)
library(caret)

# >>> Scatterplot Matrix
transparentTheme(trans = .4)
featurePlot(x = data[, 1:4],
            y = data$Y, # ----------------------- categorical Y
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 3))

# >>> Overlayed Density Plots
transparentTheme(trans = .9)
featurePlot(x = data[, 1:4],
                  y = data$Y, # ----------------------- categorical Y
                  plot = "density",
                  ## Pass in options to xyplot() to 
                  ## make it prettier
                  scales = list(x = list(relation="free"),
                                y = list(relation="free")),
                  adjust = 1.5,
                  pch = "|",
                  layout = c(4, 1), # by row, col
                  auto.key = list(columns = 3)) # level of Y

# >>> Box Plots
featurePlot(x = data[, 1:4],
                  y = data$Y, # ----------------------- categorical Y
                  plot = "box",
                  ## Pass in options to bwplot() 
                  scales = list(y = list(relation="free"),
                                x = list(rot = 90)),
                  layout = c(4, 1),
                  auto.key = list(columns = 2))

# >>> Scatter Plots
Var <- c("var1", "var2", "var3") # var names of the x
theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)
featurePlot(x = Data[, Var],
            y = Data$Y, # ----------------------- continuious Y
            plot = "scatter",
            type = c("p", "smooth"), # add soomth line
            span = .5, # add soomth line
            layout = c(3, 1)) # 3 plots in one row




# [2] Pre-proccessing

# >>> dummy var
processed.data <- model.matrix(Y ~ ., data = data) # base R function
processed.data <- dummyVars(Y ~ ., data = data) # generate dummy for all level

# >>> zero- near zero variance (Skewed var)
# -- "frequency ratio" (the frequency of the most prevalent value over the second most frequent value)
# -- "percent of unique values" (the number of unique values divided by the total number of samples)
nzv <- nearZeroVar(data, saveMetrics= TRUE) # analyze every var with detail information
nzv <- nearZeroVar(mdrrDescr) # return the positions of the variables that are flagged to be problematic.
filtered.data <- data[, -nzv] # exclude those nz vars

# >>> Identifying Correlated Predictors
descrCor <- cor(data)
summary(descrCor[upper.tri(descrCor)]) # summary of the correlations amoung vars
highlyCorDescr <- findCorrelation(descrCor, cutoff = .75) # positions of all |x| >= 0.75
data <- data[,-highlyCorDescr] # filter out those vars
descrCor2 <- cor(data)
summary(descrCor2[upper.tri(descrCor2)]) # check summary again to see change

# >>> Linear Dependencies
comboInfo <- findLinearCombos(data)
comboInfo # linear dependent combination vars
$linearCombos ; $remove
data[, --comboInfo$remove] # filter out those linear comb

# >>> center, sacle, imputation, transformation (preProcess) function
preProcValues <- preProcess(data.train, method = c("center", "scale", "pca", "BoxCox")) # many other methods / not actually change data / when "predict" -> new class "predict.preProcess" is used
model <- predict(preProcValues, data.train)
model <- predict(preProcValues, data.test) # all works

# >>> Class Distance Calculation (Categorical Y)
"""contains functions to generate new predictors variables based on distances to class centroids 
(similar to how linear discriminant analysis works). For each level of a factor variable, 
the class centroid and covariance matrix is calculated. For new samples, 
the Mahalanobis distance to each of the class centroids is computed and can be used as an additional predictor. 
This can be helpful for non-linear models when the true decision boundary is actually linear. """
centroids <- classDist(predict.Y.train, true.Y.train)
distances <- predict(centroids, predict.Y.test)
distances <- as.data.frame(distances)



# [3] data Spliting

# >>> Simple Splitting Based on the Outcome
set.seed(3456) # ----------------------------------------- Y categorical
trainIndex <- createDataPartition(data$Y, p = .6, # data % split
                                  list = FALSE, # return not list
                                  times = 1) # times split
Data.Train <- data[trainIndex,]
data2  <- data[-trainIndex,]
validIndex <- createDataPartition(data2$Y, p = .75,
                                  list = FALSE,
                                  times = 1)
Data.Valid <- data[validIndex,]
Data.Test  <- data[-validIndex,]

set.seed(3456) # ----------------------------------------- Y numeric
trainIndex <- createDataPartition(data$Y, p = .6, # data % split
                                  list = FALSE, # return not list
                                  groups = min(5, length(data$Y)), # quantile by 5
                                  times = 1) # times split
Data.Train <- data[trainIndex,]
data2  <- data[-trainIndex,]
validIndex <- createDataPartition(data2$Y, p = .75,
                                  list = FALSE,
                                  groups = min(5, length(data$Y)), # quantile by 5
                                  times = 1)
Data.Valid <- data[validIndex,]
Data.Test  <- data[-validIndex,]


createResample(data$y, times = 10, list = TRUE) # returns # times of same size bootstrap sample set (position number)
createFolds(y, k = 10, list = TRUE, returnTrain = FALSE) # divide data into k fold set in a list (position number)
createMultiFolds(y, k = 10, times = 5) 


# >>> Splitting Based on the Predictors
""" Also, the function maxDissim can be used to create sub–samples using a maximum dissimilarity approach (Willett, 1999). 
Suppose there is a data set A with m samples and a larger data set B with n samples. 
We may want to create a sub–sample from B that is diverse when compared to A. 
To do this, for each sample in B, the function calculates the m dissimilarities between each point in A. 
The most dissimilar point in B is added to A and the process continues. """
newSample <- maxDissim(old.data, sample.pool, n = 100) # sample 100 from sample.pool based on 
#                                                        max dissamiliarity between old.data - sample.pool
minDiss ???
sumDiss ???


# >>> Data Splitting for Time Series
""" Simple random sampling of time series is probably not the best way to resample times series data. 
Hyndman and Athanasopoulos (2013)) discuss rolling forecasting origin techniques that move 
the training and test sets in time. """
createTimeSlices(y, initialWindow, horizon = 1, 
                 fixedWindow = TRUE, skip = 0)
                    ???
                     parameters = "initialWindow": the initial number of consecutive values in each training set sample
                                  "horizon": The number of consecutive values in test set sample
                                  "fixedWindow": A logical: if FALSE, the training set always start at the first sample and the training set size will vary over data splits


# [4] Variable Importance - model based or non-model based
importance <- varImp(data)
?varImp
plot(importance, top = 20) # plot vars importance



# [5] Measuring Model Performce
predict.test.Y <- predict(model, test.data) # ------------ classification
postResample(predict.test.Y, test.data$Y) # accuracy rate, Kappa
sensitivity(predict.test.Y, test.data$Y) # sensitivity
specificity
posPredValue
negPredValue
confusionMatrix(predict.test.Y, test.data$Y) # full measurement set of classification
confusionMatrix(model) # information for train result (resampled)

test_results <- predict(model, test.data, type = "prob") # ------------- multiple >3 classes > get p() for each class
test_results$obs <- test.data$Y # add real class label as var in it
mnLogLoss(test_results, lev = levels(test_results$obs)) # negative of the multinomial log-likelihood (smaller is better) based on the class probabilities

test_results$pred <- predict(model, test.data) # add predict value into test_result ------------- multiple >3 classes
multiClassSummary(test_results, lev = levels(test_results$obs)) # logloss, ROC, Accuracy, Kappa, Sensitivity,
#                                                                 Specificity, Pos_Pred_Value, Neg_Pred_Value
#                                                                 Detection_Rate, Balanced_Accuracy


# >>> Evaluate classification model performance via class proability (lift curce + probability calibration)
evalResults <- data.frame(Class = data.test$class) # all classes
evalResults$FDA <- predict(fdaModel, data.test, type = "prob")[,"Class1"] # just class 1
evalResults$LDA <- predict(ldaModel, data.test, type = "prob")[,"Class1"]
evalResults$C5.0 <- predict(c5Model, data.test, type = "prob")[,"Class1"]
evalResults # all classes, model1 p() for class1, model2 p() for class1 ..
# lift curve ???
trellis.par.set(caretTheme())
liftData <- lift(Class ~ FDA + LDA + C5.0, data = evalResults)
plot(liftData, values = 60, auto.key = list(columns = 3,
                                            lines = TRUE,
                                            points = FALSE))
# probability calibration ???
trellis.par.set(caretTheme())
calData <- calibration(Class ~ FDA + LDA + C5.0,
                       data = evalResults,
                       cuts = 13)
plot(calData, type = "l", auto.key = list(columns = 3,
                                          lines = TRUE,
                                          points = FALSE))


# [6] parallel computation
doParallel()




# ---------------------------------------- Modeling ----------------------------------------------- #
# [7] Basic Syntax

# >>> Model Training and Parameter Tuning

# trainControl Function (training data / Pre-processing / Modeling measurement)
fitControl <- trainControl(## 10-fold CV
                           method = "repeatedcv",
                           number = 10,
                           ## repeated ten times
                           repeats = 10)

# expand,grid Function (Parameters Tunning Grid) - full model list "http://topepo.github.io/caret/modelList.html"
gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9),
                        n.trees = (1:30)*50,
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

# Training Models (Calculate models average performance for each parameter combination)
set.seed(825)
gbmFit2 <- train(Class ~ ., data = training,
                 method = "gbm",
                 trControl = fitControl,
                 verbose = FALSE,
                 ## Now specify the exact models 
                 ## to evaluate:
                 tuneGrid = gbmGrid,
                 preProc = c("center", "scale"),
                 metric = "ROC")

# Plot Result for all combination
trellis.par.set(caretTheme())
plot(gbmFit2)

# Select the best model
best???
oneSE???
whichTwoPct <- tolerance(gbmFit2$results, metric = "ROC",
                         tol = 2, maximize = TRUE) # 2% less than best model (ordering the models from simplest to complex)
cat("best model within 2 pct of best:\n")
gbmFit2$results[whichTwoPct,1:6] # get performance result of it / Select best version from all versions

# Extracting Predictions and Class Probabilities
predict(gbmFit2, newdata = testing, type = "class")
predict(gbmFit2, newdata = testing, type = "prob")


# Compare differences between Models - on training set (In this way we reduce the within-resample correlation that may exist)
# Adjust control object
fitControl <- trainControl(## 10-fold CV
                           method = "repeatedcv",
                           number = 10,
                           ## repeated ten times
                           repeats = 10,
                           resamples = "final") # add this for using "resample" later
# Model - 1
set.seed(825)
gbmFit2 <- train(Class ~ ., data = training,
                 method = "gbm",
                 trControl = fitControl,
                 verbose = FALSE,
                 ## Now specify the exact models 
                 ## to evaluate:
                 tuneGrid = gbmGrid,
                 preProc = c("center", "scale"),
                 metric = "ROC") # use same KPI
gbmFit2
# Model - 2
set.seed(825)
svmFit <- train(Class ~ ., data = training,
                 method = "svmRadial",
                 trControl = fitControl,
                 preProc = c("center", "scale"),
                 tuneLength = 8,
                 metric = "ROC") # use same KPI
svmFit
# Model - 3
set.seed(825)
rdaFit <- train(Class ~ ., data = training,
                 method = "rda",
                 trControl = fitControl,
                 tuneLength = 4,
                 metric = "ROC") # use same KPI
rdaFit
# Collect the resampling result
resamps <- resamples(list(GBM = gbmFit3,
                          SVM = svmFit,
                          RDA = rdaFit))
summary(resamps) # conduct 100 bootstrap resamples for each model, compare sets of result
"""
Call:
summary.resamples(object = resamps)

Models: GBM, SVM, RDA 
Number of resamples: 100 

ROC 
      Min. 1st Qu. Median   Mean 3rd Qu. Max. NA's
GBM 0.5714  0.8464 0.9085 0.8963  0.9526    1    0
SVM 0.6786  0.9107 0.9557 0.9445  0.9844    1    0
RDA 0.6508  0.8571 0.9206 0.9074  0.9653    1    0

Sens 
     Min. 1st Qu. Median   Mean 3rd Qu. Max. NA's
GBM 0.625  0.8507 0.8750 0.8821       1    1    0
SVM 0.500  0.8750 0.8889 0.9090       1    1    0
RDA 0.500  0.8750 0.8889 0.9008       1    1    0

Spec 
      Min. 1st Qu. Median   Mean 3rd Qu. Max. NA's
GBM 0.4286  0.7143 0.7500 0.7664  0.8571    1    0
SVM 0.2857  0.7143 0.8571 0.8155  0.9062    1    0
RDA 0.2857  0.7143 0.7500 0.7643  0.8571    1    0
"""
# Plot
trellis.par.set(theme1)
bwplot(resamps, layout = c(3, 1)) # 3 models in 1 row - boxplot

# Use T-test to evaluate the null hypothesis "Differences between models?" / Select best model from other models
difValues <- diff(resamps)
difValues
summary(difValues)

# Train Model with known parameter combination set
fitControl <- trainControl(method = "none", classProbs = TRUE) # without resample / tunning

set.seed(825)
gbmFit4 <- train(Class ~ ., data = training,
                 method = "gbm",
                 trControl = fitControl,
                 verbose = FALSE,
                 ## Only a single model can be passed to the
                 ## function when no resampling is used:
                 tuneGrid = data.frame(interaction.depth = 4, # define one set of parameter combination set
                                       n.trees = 100,
                                       shrinkage = .1,
                                       n.minobsinnode = 20),
                 metric = "ROC")
gbmFit4
# predict
predict(gbmFit4, newdata = testing)
predict(gbmFit4, newdata = testing, type = "prob")


















