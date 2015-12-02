# <<<<<<<<<<<<<<<<<<<<<<<<<Parallel Computation in R>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# [1] Package ---- "doParallel"
### Installation of package with all dependent packages
chooseCRANmirror()
install.packages("doParallel", dependencies = c("Depends", "Imports")) 
# also installing the dependencies ‘foreach’, ‘iterators’
#package ‘foreach’ successfully unpacked and MD5 sums checked
#package ‘iterators’ successfully unpacked and MD5 sums checked
#package ‘doParallel’ successfully unpacked and MD5 sums checked
### End of installation, this needs to be run only once

# load the doParallel for direct use
library(doParallel)
# make a cluster with all possible threads (not cores)
cl <- makeCluster(detectCores())
# register the number of parallel workers (here all CPUs)
registerDoParallel(cl)
# return number of parallel workers
getDoParWorkers() 
# insert parallel calculations here

# -------------------------------- #
#            Function              #
# -------------------------------- #

# stop the cluster and remove  Rscript.exe childs (WIN)
stopCluster(cl)
# END


















# [2] Package ---- "doSNOW" * used with "foreach"
### Installation of the doSNOW parallel library with all dependencies
chooseCRANmirror()
install.packages("doSNOW", dependencies = c("Depends", "Imports")) 

##Loading required package: foreach
##foreach: simple, scalable parallel programming from Revolution Analytics
##Use Revolution R for scalability, fault tolerance and more.
##http://www.revolutionanalytics.com
##Loading required package: iterators
##Loading required package: snow

# Cluster
# load doSnow library
library(doSNOW)

# Create compute cluster of 4 (try 64)
# One can increase up to 128 nodes
# Each node requires 44 Mbyte RAM under WINDOWS.
cluster = makeCluster(4, type = "SOCK")
cluster = makeCluster(20, type = "MPI")
# register the cluster
registerDoSNOW(cluster)

# insert parallel computation here

# -------------------------------- #
#            Function              #
# -------------------------------- #

# stop cluster and remove clients
stopCluster(cluster)

# insert serial backend, otherwise error in repetetive tasks
registerDoSEQ()
# END




















# [3] Package ---- "parallel" 
# Library parallel() is a native R library, no CRAN required
library(parallel)

# detect true cores requires parallel()
nCores <- detectCores(logical = FALSE)
# detect threads
nThreads <- detectCores(logical = TRUE)
# detect threads
cat("CPU with",nCores,"cores and",nThreads,"threads detected.\n")

# automatically creates socketCluster under WIN, fork not allowed
# maximum number of cluster nodes is 128
cl <- makeCluster(nThreads); cl;
# insert parallel calculations here

# -------------------------------- #
#            Function              #
# -------------------------------- #

# stop the cluster and remove parallel instances
stopCluster(cl)
# END

















# ------------------ Function --------------------------- #
# [1] 'foreach' package >>> 'doSNOW'
install.packages("doSNOW")
library(doSNOW)
install.packages("doRNG")
library(doRNG)
result <- foreach(i=1:3) %dopar% sqrt(i) # simply variable use exp 'for each'
result <- foreach(a=1:3, b=rep(10, 3)) %dopar% (a + b) # specify variables to use in expression a line = b, or match small one
result <- foreach(n = 1:5) %:% foreach(m = 1:3) %do% max.eig(n, m) # can define variables in different length
                                                                   # like nested for loop, for every 'n' do 'm'
result <- foreach(n = 1:100, .combine = c) %:% when (isPrime(n)) %do% n # %:% when(condition) to filter what's going to expression
result <- foreach(i=1:4, .combine='c') %dopar% exp(i) # combine result in a vector
result <- foreach(i=1:4, .combine='cbind') %dopar% rnorm(4) # 'cbind' each result
result <- foreach(i=1:4, .combine='rbind') %dopar% rnorm(4) # 'rbind' each result
result <- foreach(i=1:4, .combine='+') %dopar% rnorm(4) # 'add' each elements within each result
result <- foreach(i=1:4, .combine='*') %dopar% rnorm(4) # 'Multiply' each elements within each result
result <- foreach(i = 1:5, .options.RNG = 123) %dorng% runif(3) # '.option.RNG' = 'set.seed'
# use user defined function
myfun <- function (vector) { # one argument
	a <- vector * 5
	return (a)
}
myfun <- function (vector) { # one argument
	lm(y~., data=vector) # use a model in fun
	return (a)
}
result <- foreach(i=1:4, .combine='myfun', .multicombine=TRUE, .maxcombine=10) %dopar% rnorm(4) # 'combine' define 'fun'
                                                                                             # 'multicombine' allows many arguments
                                                                                             # 'maxcombine' defines max arguments
# user define a package
x <- matrix(runif(500), 100)
y <- gl(2, 50)
library(randomForest)
result <- foreach(ntree=rep(250, 4), .combine=combine, .packages='randomForest') %dopar% randomForest(x, y, ntree=ntree)
# Applying fun across nodes using a package, load package on each node
results = foreach(n = 1:100, .combine = c) %dopar% {
    library(boot); sd(boot(random.data[, n], bmed, R = 10000)$t)
} # in expression load package, will load on all nodes





# [2] 'plyr' package >>> 'doParallel'
install.package("plyr")
library(plyr)
# a-metrix / d-data.frame / l-list
# xxply(data, .(var1, var2), fun=myfun, .parallel=T)
aaply
adply
alply

daply
ddply
dlply

laply
ldply
llply








# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Optimization function in Model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>