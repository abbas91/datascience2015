# Modeling Process 


# Step1 - Loading Packages
install.packages()



# Step2 - Loading Data in





# Step3 - Data checking 





# Step4 - Data Mugging / Transforming







# Step5 - Spliting Dataset for Modeling

# -- Split method required






# Step6 - Initial Models Testing + Learning Curve -[May go Prev]

# -- Test Model

# -- Validation Metric

# -- Learning Curve plot
# >>> Partition train set for generating "learning curve"
set.seed(599)
folds <- createFolds(TRAIN_SET$Y)
TRAIN_SET.learn <- lapply(folds, 
                          function(ind, dat) dat[ind,], 
                          dat = TRAIN_SET)
for (i in 1:10) {
  print(nrow(TRAIN_SET.learn[[i]]))
}
rm(folds, i)
# >>> Initial Models Testing + Learning Curve -[May go Prev]
Model.cost.test <- numeric(length=10)
Model.cost.train <- numeric(length=10)
for (i in 1:10) {
  # >>> Train models - can do basic tunning
  Model.fun <- rpart(Y~., 
                     data=do.call("rbind", TRAIN_SET.learn[1:i]), 
                     method="class")
  #Model.fun <- prune(Model.fun, 
                     #cp=Model.fun$cptable[which.min(Model.fun$cptable[,"xerror"]),"CP"]) #Automatic
  # >>> Prediction
  predict.train <- ifelse(predict(Model.fun, type = c("class")) == "YES", 0, 1)
  predict.test <- ifelse(predict(Model.fun, newdata=TEST_SET, type = c("class")) == "YES", 0, 1)
    
  # >>> performance - error.rate
  Model.cost.test[i] <- mean(predict(Model.fun, newdata=TEST_SET, type = c("class")) != TEST_SET$Y)
  Model.cost.train[i] <- mean(predict(Model.fun, type = c("class")) != do.call("rbind", TRAIN_SET.learn[1:i])$Y)
}

Model.cost.table <- as.data.frame(rbind(Model.cost.test, Model.cost.train))
Model.cost.table <- cbind(c("Test Cost", "Train Cost"), Model.cost.table)
rownames(Model.cost.table) <- c()
names(Model.cost.table) <- c("Cost_Type", "10%", "20%", "30%", "40%", "50%",
                                          "60%", "70%", "80%", "90%", "100%")
rm(Model.cost.test, Model.cost.train, predict.train, predict.test, i)
# >>> melt plot format
mdf <- melt(Model.cost.table, id.vars="Cost_Type", 
            value.name="Error_rate", 
            variable.name="Percet_of_Trainset")
# >>> Plot Learning Curve
ggplot(data=mdf, aes(x=Percet_of_Trainset, y=Error_rate, group = Cost_Type, colour = Cost_Type)) +
  geom_line() +
  geom_point( size=4, shape=21, fill="white") +
  ggtitle("Learning Curve Plot")




# Step 7 - 



