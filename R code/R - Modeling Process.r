################################
#                              #
#        Modeling Process      #
#                              #
################################

# Objective
"""
The objective of this modeling project
"""





# >>> Step1 - Loading Packages
install.packages("caret")
library(caret)

# - launch list of libraries
install.packages("needs")
library(needs)
## 
## Load `package:needs` in an interactive session to set auto-load flag
needs(dplyr, tidyr, stringr, lubridate, readr, ggplot2, MASS, pander, formattable, viridis)




# >>> Step2 - Loading Data in

[R - ETL Workflow] > [Input Data File]

# >>> Step3 - Data checking (Pre-Mugging)

[R - EDA Workflow] > ALL

# >>> Step4 - Data Mugging: check error

[R - ETL Workflow] > [Manipulate Data]

# >>> Step5 - Data checking (Post-Mugging)

[R - EDA Workflow] > ALL

# >>> Step6 - Data pre-processing based on model assumption (Can done in modeling)

[R - ETL Workflow] > [Manipulate Data]

# >>> Step7 - Data checking (Post-Data ore-processing)

[R - EDA Workflow] > ALL


# >>> Step8 - Spliting Dataset for Modeling
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




# -- Learning Curve Building [Optional]
# plot learning curve
fitControl <- trainControl(## 10-fold CV
                           method = "repeatedcv",
                           number = 10,
                           ## repeated ten times
                           repeats = 10) 

set.seed(29510)
glm_data <- learing_curve_dat(dat = Data.Train_valid, 
                              outcome = "first_complete_y",
                              test_prop = 2/8, 
                              ## `train` arguments:
                              method = "glm",
                              family = binomial,
                              metric = "Kappa",
                              trControl = fitControl)


ggplot(glm_data, aes(x = Training_Size, y = Kappa, color = Data)) + 
  geom_smooth(method = loess, span = .8) + 
  theme_bw()










# >>> Step9 - Modeling Data with multiple model selection (caret package)

" >>> Run in parallel [optional]"
# [1] Parallel + Reproducible Results
library(doParallel); library(caret)
#create a list of seed, here change the seed for each resampling
set.seed(123)
seeds <- vector(mode = "list", length = 11) #length is = (n_repeats*nresampling)+1
for(i in 1:10) seeds[[i]]<- sample.int(n=1000, 3) #(3 is the number of tuning parameter, mtry for rf, here equal to ncol(iris)-2)

seeds[[11]]<-sample.int(1000, 1)#for the last model

 #control list
fitControl <- trainControl(## 10-fold CV
                           method = "repeatedcv",
                           number = 10,
                           ## repeated ten times
                           repeats = 10) 

 #run model in parallel - plot learning curve
 cl <- makeCluster(detectCores())
 registerDoParallel(cl)
 getDoParWorkers() # Check number of clusters registered


set.seed(29510)
glm_data <- learing_curve_dat(dat = Data.Train_valid, 
                              outcome = "first_complete_y",
                              test_prop = 2/8, 
                              ## `train` arguments:
                              method = "glm",
                              family = binomial,
                              metric = "Kappa",
                              trControl = fitControl)

 stopCluster(cl)

ggplot(glm_data, aes(x = Training_Size, y = Kappa, color = Data)) + 
  geom_smooth(method = loess, span = .8) + 
  theme_bw()





" >>> Evaluate Single Model"
# [1] trainControl Function (training data / Pre-processing / Modeling measurement)
fitControl <- trainControl(## 10-fold CV
                           method = "repeatedcv",
                           number = 10,
                           ## repeated ten times
                           repeats = 10) 

# [2] expand,grid Function (Parameters Tunning Grid) - full model list
gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9),
                        n.trees = (1:30)*50,
                        shrinkage = 0.1,
                        n.minobsinnode = 20)
"http://topepo.github.io/caret/modelList.html" # find out more arguments for each model
"<<Load model package before using it in caret>>"

# [3] Training Models (Calculate models average performance for each parameter combination)
set.seed(825)
gbmFit2 <- train(Class ~ ., data = training,
                 method = "gbm",
                 trControl = fitControl,
                 verbose = FALSE,
                 ## Now specify the exact models 
                 ## to evaluate:
                 tuneGrid = gbmGrid,
                 preProc = c("center", "scale"), # --- Pre-pro
                 metric = "ROC")

# Plot Result for all combination
trellis.par.set(caretTheme())
plot(gbmFit2)

# [3] Select the best model ***
best???
oneSE???
whichTwoPct <- tolerance(gbmFit2$results, metric = "ROC",
                         tol = 2, maximize = TRUE) # 2% less than best model (ordering the models from simplest to complex)
cat("best model within 2 pct of best:\n")
gbmFit2$results[whichTwoPct,1:6] # get performance result of it / Select best version from all versions






" >>> Combine models to select the best!"
# [1] Add "resample" to control
fitControl <- trainControl(## 10-fold CV
                           method = "repeatedcv",
                           number = 10,
                           ## repeated ten times
                           repeats = 10,
                           resamples = "final") # add this for using "resample" later


# [2] Use best grid combination + retrain on trainset for each model object
# Model - 1
gbmGrid <-  expand.grid(interaction.depth = 9, # Best combination developed from steps above
                        n.trees = 100,
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

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
svmGrid <-  expand.grid(interaction.depth = 9, # Best combination developed from steps above
                        n.trees = 100,
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

set.seed(825)
svmFit <- train(Class ~ ., data = training,
                 method = "svmRadial",
                 trControl = fitControl,
                 tuneGrid = svmGrid,
                 preProc = c("center", "scale"),
                 tuneLength = 8,
                 metric = "ROC") # use same KPI
svmFit

# Model - 3
rdaGrid <-  expand.grid(interaction.depth = 9, # Best combination developed from steps above
                        n.trees = 100,
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

set.seed(825)
rdaFit <- train(Class ~ ., data = training,
                 method = "rda",
                 trControl = fitControl,
                 tuneGrid = rdaGrid,
                 tuneLength = 4,
                 metric = "ROC") # use same KPI
rdaFit

# [3] Collect the resampling result
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

# [4] Use T-test to evaluate the null hypothesis "Differences between models?" / Select best model from other models
difValues <- diff(resamps)
difValues
summary(difValues)
"Best Model Identified"

















