############################################
#                                          #
#                                          #
#            Optimization in R             #
#                                          #
#                                          #
############################################

> System Information

> Parallel Computation

> Optimization Methods

>




>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>                                     >
>                                     >
>        System Information           >
>                                     >
>                                     >
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

"""
A call of gc causes a garbage collection to take place. This will also take place automatically without user intervention, and the primary purpose of calling gc is for the report on memory usage.

However, it can be useful to call gc after a large object has been removed, as this may prompt R to return memory to the operating system.

R allocates space for vectors in multiples of 8 bytes: hence the report of 'Vcells', a relict of an earlier allocator (that used a vector heap).

When gcinfo(TRUE) is in force, messages are sent to the message connection at each garbage collection of the form
"""
> gc()

gc() #- do it now
gcinfo(TRUE) #-- in the future, show when R does it
x <- integer(100000); for(i in 1:18) x <- c(x, i)
gcinfo(verbose = FALSE) #-- don't show it anymore

gc(TRUE)

gc(reset = TRUE)







>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>                                     >
>                                     >
>        Parallel Computation         >
>                                     >
>                                     >
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


>>>>> doParallel >>>>>>>


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
cl <- makeCluster(detectCores(), type='PSOCK')
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













>>>>> doSNOW >>>>>>>


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
















>>>>>> parallel >>>>>>>>>

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




















>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>                                     >
>                                     >
>        Optimization Methods         >
>                                     >
>                                     >
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< in Model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# &&&&&&&&&&&&&&&&&& Blind Search
""" Full blind search assumes the exhaustion of all alternatives, any previous search does not 
affect how next solutions are tested, given full search space is tested, optimum solution is always found """

""" Only applible to discrete search space and in two ways:
    1. Setting full search space in a matrix and then sequentially test each row of the matrix
    2. In a recursive way, setting search space as a tree, each brunch denotes a possible value for a given
       variable and all solutions appear at the leaves - (Type1 - depth first, starts at root, goes to brunch asap 
       	                                                  Type2 - breadth first, starts at root, search all nodes at a given level)
"""

""" Disadvantage - computational infeasible """


# [Full blind search]

# (1) --------------- full blind search
#     search - matrix with solutions X D
#     FUN - evaluation function
#     type - "min" or "max"
#     ... - extra?
fsearch <- function (search, FUN, type = "min", ...) {
	x <- apply(search, 1, FUN, ...) # run FUN over all search rows
	ib <- switch(type, min=which.min(x), max=which.max(x))
	return (list(index=id, sol=search[ib,], eval=x[ib]))
}






# (1) --------------- depth first full search
#     l - level of the tree
#     b - branch of the tree
#     domain - vector list of size D with domain values
#     FUN = eval function
#     type = "min" or "max"
#     D - dimension (number of variables)
#     x - current solution vector
#     bcur - current best sol
#     ... - extra?
dfsearch <- function (l=1, b=1, domain, FUN, type="min", D=length(domain), x=rep(NA,D), bcur=switch(type, min=list(sol=NULL, eval=Inf),
	                                                                                                      max=list(sol=NULL, eval=-Inf)),...) {
	if((l-1) == D) { # "leave" with solution x to be tested:
      f=FUN(x,...); fb=bcur$eval
      ib=switch(type, min=which.min(c(fb,f)),
      	              max=which.max(c(fb,f)))
      if(ib == 1) return (bcur) else return(list(sol=x, eval=f))
} else { # go through sub branches
	for (j in 1:length(domain[[l]])) {
		x[l] = domain[[l]][j]
		bcur = dfsearch(l+1, j, domain, FUN, type, D=D, x=x, bcur=bcur, ...)
	}
	return (bcur)
  }
}





# [Grid search]
""" Reduce the space of solutions by implementing a regular hyper dimensional search with a 
given step size. Grid search is particularly used for hyperparameter optimization of machine learning algorithms, 
such as NNL, SVM. So, it is faster.
"""

"""
Disadvantage - more easily stuck by local minium
"""




# [Monte Carlo search]













>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>                                     >
>                                     >
>         Big Memory Projects         >
>                                     >
>                                     >
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



> bigmemory


> biganalytics



> bigtabulate



> synchronicity



> bigalgebra



install.packages("bigmemory", repos="http://R-Forge.R-project.org")




















