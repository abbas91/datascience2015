## ---------------------- Loading packages ------------------------ ##
install.packages("reshape")
install.packages("Rcpp")
install.packages("ggplot2")
install.packages("reshape2")

library(reshape)
library(Rcpp)
library(ggplot2)
library(reshape2)

## --------------------------- Acquire.FUN -------------------------- ##
## Returns Rawdataset / numeric-transformed-dataset (List) ## ---- OUTPUT
acquire.fun <- function (file.name) {
      data1 <- read.csv(file.name, header = TRUE, sep="\t", na.strings = "NA", fileEncoding="UTF-16LE") ##Local
      data.sub <- data1[,c(2,3,17,18,24,25,27)]

## -- Clean up data (1)
supply_TYPE <- ifelse(data.sub$supply_type %in% 'web', 'web',
                     ifelse(data.sub$supply_type %in% 'facebook_sidebar', 'facebook_sidebar',
                           ifelse(data.sub$supply_type %in% 'mobile_web', 'mobile_web',
                                 ifelse(data.sub$supply_type %in% 'mobile_app', 'mobile_app',
                                       'Other'))))
data.sub <- data.sub[,-7]
data.sub <- cbind(data.sub, supply_TYPE)
row.number <- nrow(data.sub)
## -- Clean up data (2)
 roadblock_TYPE <- vector(length = row.number)
 index <- sample(row.number, (row.number/2))
 roadblock_TYPE[index] <- 'roadblock'
 roadblock_TYPE[-index] <- 'no_roadblock'
 data.sub <- data.sub[,-5]
  data.sub <- cbind(data.sub, roadblock_TYPE)
   data.sub$advertiser_name <- as.character(data.sub$advertiser_name)
## -- cluster dataset
data.sub.train <- data.sub[,-c(1,2)]
## -- Convert to numeric & Normalize
## FUN
oo <- function (data) {  
  for (i in 1:ncol(data)) {
    data[,i] <- as.numeric(data[,i])
  }
  return (data)
}
 data.sub.numeric <- oo(data.sub.train)
 data.sub.normal <- rescaler(data.sub.numeric, "range")
 data.all <- list(data.sub, data.sub.normal)
 return (data.all)
}  



## --------------------------- Explore.FUN -------------------------- ##
## Returns sample raw data / PDF plot of transformed data / raw-data / Transofrmed data ## ---- OUTPUT
explore.fun <- function (data.all) {
   raw.data <- data.all[[1]]
   numeric.data <- data.frame(data.all[[2]])  
## ---- High Level Summary (raw Data)
data.sample <- data.frame(head(raw.data, 100))
write.table(data.sample, file = "raw.data.sample.csv",col.names=T, sep=",")
## PCA 
pca.table <- prcomp(numeric.data, scale = TRUE)
variance <- pca.table$sdev^2 #Variance explained by each pc
Pvar <- variance / sum(variance)
pca.vector <- pca.table$x[,1:2] #Get 1st 2nd pc vectors
data.all[[3]] <- pca.vector
## ---- (PDF)
pdf("Normalized.data.plot.pdf")
  boxplot(numeric.data, main="Normalized Dataset")
dev.off()
pdf("Proportion of Variance explained (PVE).pdf")
  plot(Pvar, xlab="Principal Component", ylab="Proportion of Variance Explained", ylim=c(0,1), type='b')
dev.off()
return (data.all)
}

## numeric.data <- step2.output[[2]] ; pca.vector <- step2.output[[3]] ; k <- 3; raw.data <- step2.output[[1]]; Result <- step3.output

## --------------------------- Analyse.FUN -------------------------- ##
## Generates 4 segments of data for those 4 groups (list) ## ---- OUTPUT
analyze.fun <- function (data.all,k) {
  raw.data <- data.all[[1]]
  numeric.data <- data.all[[2]]
  pca.vector <- data.all[[3]]
## RUN Clustering
 table.k <- kmeans(numeric.data, k, nstart = 50)
 group <- table.k$cluster
 Result <- data.frame(cbind(group, raw.data))
 pca.plot <- data.frame(cbind(group, pca.vector))
## Generate PCA plot
pdf("PCA.cluster.plot.pdf")
  qplot(PC1, PC2, data = pca.plot, colour = factor(group), main = "PCA.Cluster.plot") 
dev.off()
 return (Result)
}






## --------------------------- Result.FUN -------------------------- ##
## Generates PDF plot for each metric in those 4 group ## ---- OUTPUT
result.fun = function (Result) {
  Result <- Result[,-3]
  number.column <- ncol(Result)
  var.name <- names(Result)
  k.number <- length(unique(Result$group))
## Split cluster dataset
z <- list()
for (i in 1:k.number) {
  z[[i]] <- subset(Result, Result$group == i)
}
  
  ## -- Multiplot FUN -- ##
  multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
    require(grid)
    
    # Make a list from the ... arguments and plotlist
    plots <- c(list(...), plotlist)
    
    numPlots = length(plots)
    
    # If layout is NULL, then use 'cols' to determine layout
    if (is.null(layout)) {
      # Make the panel
      # ncol: Number of columns of plots
      # nrow: Number of rows needed, calculated from # of cols
      layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                       ncol = cols, nrow = ceiling(numPlots/cols))
    }
    
    if (numPlots==1) {
      print(plots[[1]])
      
    } else {
      # Set up the page
      grid.newpage()
      pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
      
      # Make each plot, in the correct location
      for (i in 1:numPlots) {
        # Get the i,j matrix positions of the regions that contain this subplot
        matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
        
        print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                        layout.pos.col = matchidx$col))
      }
    }
  }
  
  ## -- plot1
  c1.st = qplot(supply_TYPE, data = cluster1, geom = 'histogram', main = "cluster1 - supply_TYPE")
  c2.st = qplot(supply_TYPE, data = cluster2, geom = 'histogram', main = "cluster2 - supply_TYPE")
  c3.st = qplot(supply_TYPE, data = cluster3, geom = 'histogram', main = "cluster3 - supply_TYPE")
  c4.st = qplot(supply_TYPE, data = cluster4, geom = 'histogram', main = "cluster4 - supply_TYPE")
  multiplot(c1.st, c2.st, c3.st, c4.st, cols=2)
## ------  Plot
cluster.n <- length(z)

plot1 <- rep(list(rep(list(NULL),7)),4)
plot1 <- list()

for (c in 1:28) {
  for (i in 1:4) {
    for (j in 1:7) { ## Need check
      plot1[[c]] <- qplot(z[[i]][,j], geom = 'histogram', main = var.name[j])
    }
  }
}
## ----- Plot PDF 
for (i in 1:cluster.n) {
  PDF()
}

plot1[[6]]

plot.dd <- function (data) {
  p <- list()
  var.name <- names(data)
  for (j in 1:ncol(data)) {    
    p <- qplot(data[,j], geom = 'histogram', main = var.name[j]) 
  }
  return (p)
}


plot1 <- plot.dd(z[[1]])

plot1[[1]]



a <- qplot(z[[1]][,1], geom = 'histogram', main = var.name[1])

b <- qplot(z[[1]][,2], geom = 'histogram', main = var.name[2])

multiplot(a,b, cols=2)



## ******************************* Test RUN *********************************** ##

# Acquire
setwd("C://AWB Project//Analysis//Analysis 2//1//Acquire")
file.name = "Data-Martet.csv"
step1.output = acquire.fun(file.name)
## Check OUTPUT
str(step1.output[[1]])
str(step1.output[[2]])

# Explore
setwd("C://AWB Project//Analysis//Analysis 2//1//Explore")
step2.output = explore.fun(step1.output)
## Check OUTPUT
str(step2.output[[1]])
str(step2.output[[2]])

# Analyze
setwd("C://AWB Project//Analysis//Analysis 2//1//Analyze")
step3.output = analyze.fun(step2.output,4) ## Specify number of clusters
## Check OUTPUT
str(step3.output)

# Result
setwd("C://AWB Project//Analysis//Analysis 2//1//Result")
step4.output = result.fun()
## Check OUTPUT





