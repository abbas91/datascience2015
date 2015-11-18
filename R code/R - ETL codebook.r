## ETL

## >>>> Input Data File
#       > From Local
#       > From Web
#       > From SQL
...
## >>>> Manipulate Data
#       > Create subset
#       > Combine / Create / Modify variables
#       > Transform data distribution / Form
#       > Scale / Normalization
#       > Create emplty table / Convert table
#       > Merge / Re-arrange Dataset
#       > Imputation
...
## >>>> Output Data File
#       > Output to local
#       > Output to Tableau
...








## -----------------{input} 

## [1]From Local
##    <*> Open local file connection
          setwd("dir")
		  con <- file("xxx.txt", "xxx2.txt")
		         gzfile() # file compressed with gzip
				 bzfile() # file compressed with bzip2
				 url() # opens a connection to a webpage
		  
##    <1> csv
          class <- c("numeric", "character", "factor", "numeric", "numeric") # Read local csv file with pre-defined class
          data <- read.csv("data.csv", colClasses = class)
          view1 = read.csv("xxx.csv", header = TRUE, stringsAsFactors = FALSE) ##Local
		  view1 = read.csv(unz("xxxx.csv.zip", "xxxx.csv"), stringsAsFactors = F) ##Load zip file
##    <2> Excel

##    <3> txt
          view1 = read.table("xxx.txt", header = TRUE, stringsAsFactors = FALSE) ##Local
		  view1 = read.table(file = "clipboard", header = TRUE) ##Copy/paste
		  view1 = readLines("xxx.txt", n = 10)
##    <4> r.code
          source("xxx.r")
##    <5> Large Dataset
#         Calculate Memory Requirements -> ex. 1,500,000 rows / 120 columns (all numeric data)
#                                          1,500,000 X 120 X 8 bytes/numeric
#                                         = 1440000000 bytes
#                                         = 1440000000 / 2^20 bytes/MB
#                                         = 1,373.29 MB
#                                         = 1.34 GB
##        (1) Define the data before loading can make it faster
           initial <- read.table("xxxx.txt", nrows = 100)
           classes <- sapply(initial, class)
           View1 <- read.table("xxxx.txt",
		                      colClasses = classes)	# The more you define better	   
#
## [2]From Web
##    <1> Data table
          install.packages("XML") ##Load data from web
          library(XML)
          URL = ("http://.....//....")
          view1 = readHTMLTable(URL, encoding = "UTF-8", colClasses="charater") ##A list, can use [[#]] to access right data
##    <2> Data file
          URL <- "http:........."
          download.file(URL,"filename") ##Download from internet to the working directory		  
#
...
## [3]From SQL
          install.packages("RODBC") ##SQL databases
          library(RODBC)
          mydb = odbcConnect("my_dsn", uid = "my_username", pwd = "my_password")
          query = "select * from table"
          view1 = sqlQuery(channel = mydb, query = query, stringsAsFactors = FALSE)
          odbcClose(mydb) ##close connection
...
#
## [4]From ...
#
...
## -------------------{Manipulate Data}
## [1]Create subset
##    <1> dplyr dataframe management ## Fast in C++
          install.packages("dplyr")
          library(dplyr)
##    <2> SELECT FUN () *choose columns by name
          names(data) ## Get sense of the var names
          Subset <- select(data, varname1:varname7)
          Subset <- select(data, -(varname1:varname7)) ##Include everything except..
          Subset <- select(data, c(varname1,varname3,varname5))
          Subset <- select(data, ends_with("2")) ## match strings
          Subset <- select(data, starts_with("2"))
          Subset <- select(data, regx) ##check ?select to see more
##     <3> FILTER FUN () *set logical rule to filter data
          Subset <- filter(data, varname1 > 30 & varname2 < 70) # subset of data that match the filter (&, ||)
##     <4> ARRANGE FUN () *reorder dataset by variable or columns
          Subset <- arrange(data, varname1)
          Subset <- arrange(data, desc(varname1))
##     <5> RENAME FUN () *rename variables
          Subset <- rename(data, varname.new1 = varname1, varname.new2 = varname2)
##     <6> MUTATE / TRANSMUTE FUN () *transform variables, create new variables
          Subset <- mutate(data, varname.new = varname2 - mean(varname3)) # Add a new transformed var
          Subset <- transmute(data, varname.new = varname2 - mean(varname3)) #drop all non-transformed vars
##     <7> GROUP BY () *summarize information by grouping data
          group <- group_by(data, group.var)
          summarize(group, state.var1 = mean(varname1, na.rm = t), state.var2 = median(varname2, na.rm = t))
##      - Ex. create categorical from quantile of numeric_var
       qq <- quantile(data$varname.numeric1, seq(0, 1, 0.2), na.rm = TRUE)
       data <- mutate(data, varname.new.quint = cut(varname.numeric1, qq))
       group <- group_by(chicago, varname.new.quint)
       summarize(group, state.var1 = mean(varname1, na.rm = t), state.var2 = median(varname2, na.rm = t))
##     <8> %>% () *pipe multiple functions (any function, can be "as.data.frame, unique, nrow") into 
       first() %>% second() %>% thrid() # without saving into local memory
##     <9> Other subset extraction
       view2 = subset(view1, var1 %in% c("xxx", "xxxx", "xxx") & var2 >= 10 & var3 == "") ## Get subset of data 
#
## [2]Create subset
#     url - "https://s3.amazonaws.com/assets.datacamp.com/img/blog/data+table+cheat+sheet.pdf"
#     <1> data.table packages faster processing big table, treat data as table, as functioned as data.frame
          install.packages("data.table") 
          library(data.table) 
#     <2> Read data local / create data.table / convert data.table
          data <- fread("xxx.csv", header = TRUE, stringsAsFactors = FALSE) # read -> data/table
          data <- data.table(var1=c(2,3,4,5),
                             var2=c(3,4,5,6)) # create / if not in same length -> recyling
          data <- as.data.table(data) # convert - may be slow when big
          setDT(data) # convert - fast
#     <3> subsetting rows
          data[3:5,]; data[var3 == "x"] #all rows has x in column var3
          data[var4 %in% c("s", "x")] #all rows has x, s in column var4
#     <4> Manipulating on cloumns
          data[,var3] # a vector
          data[,.(var1, var2)] # return var1, var2 as data.table -> add "."
          data[, sum(var1)] # summarize / a vector (new)
          data[,.(sum(var1), sd(var3))] # summarize multiple vars / in data.table (new)
          data[,.(xx1=sum(var1), xx2=sd(var3))] # same above / rename
          data[,.(var1, xx2=sd(var3))] # recyling xx2 to all elements on var1 / data.table
          data[,{print(var1) plot(var2) NULL}] #?? multiple expressions can be warpped in curly braces
#     <5> Group fun by columns
          data[,.(xx1=sum(var4)), by=var1]; data[,.(xx1=sum(var4)), by=.(var1, var2)] # every group of var1 and var2, do sum(var3)
          data[,.(xx1=sum(var4)), by=sign(var1 - 1)] # call fun in "by"
          data[,.(xx1=sum(var4)), by=.(group1=sign(var1 - 1))] # same above + rename "by"
          data[1:15,.(xx1=sum(var4)), by=var1] #only apply this by certain rows
          data[,.N, by=var1] # like "table" / get row numbers for each level in var1
#     <6> Adding / updating columns by fun
          data[, var1 := round(exp(var2),2)] # update var1 with fun
          data[, c("var4", "var1") := list(round(exp(var2),2), var1*20)] # update multiple vars
          ...[] # after formula, result print on screen
          data[, var1 := NULL]; data[, c("var1", "var2") := NULL] # remove a var / many
          Cols.chosen <- c("var1", "var3", "var4")
          data[, (Cols.chosen) := NULL] # another way to do above
#     <7> Indexing / Keys
          setkey(data, var1); setkey(data, var1, var3) # set keys on data (must)
          data["A"]; data[c("A", "B")] # Return all rows match key-var1 -> "A" and "B"
          data["A", nomatch = 0] # no NOT-match (key1, key2 -> key2 = NA) rows will be returned / default = NA, return "NA"
          data[c("A", "B"), sum(var4)] => data[c("A", "B"), sum(var4), by=.EACHI] # group fun by "A", "B"
          data[.(c("A", "B"), c(2, 4))] # if two keys -> key1, key2 / how to match key values

#
...
## [3]Combine / Create / Modify variables
##    <1> General
       view1$var.new = ifelse(substr(view1$var.cat, 1, 4) == "xxxx", "Yes", "No") ##Create new var based on substring of current categorical var
       view1$var1 = view1$var_new %in% c("xxx", "xx", "xxxx") ## If yes TRUE else FALSE (Logical var)
       view1$High = factor(view1$High, levels = c("Yes", "No"), labels = c("High", "Low")) ##Code Char factor with lebal
       view1 = gsub("regex...", replacement, var) ##Apply Regular Expressions on dataset
##    <2> Roll - metrics
       install.packages("zoo")
       library(zoo)
       rollmeanr(var, 10, align = "right") # -> specify range "10"
       rollmaxr(var, 10, align = "right")
       rollmedianr(var, 10, align = "right") # "10" must be obb
       rollsumr(var, 10, align = "right")
##    <3> Level Reduction Categorical (Entropy Gain)
##         - Entropy value (Categorical var)
               entropy = function(x) {
               labelcounts = table(x)
               prob = labelcounts/sum(labelcounts)
               shannonent = -sum(prob * log(prob,2))
               return(shannonent)
               }
               entropy(x) ##-> the close to 0.5 the better (Ave Information in each level) Need split?
##         - compare entropy gain by split (z - categorical)
               entropy_gain = function(data, value) {
               n = nrow(data)
               shan_tot = entropy(data$y)
               sub1 = subset(data, z==value) ##-> define how to split ca_var by looking up other ca_var
               sub2 = subset(data, z!=value)
               sub1n = nrow(sub1)
               sub2n = nrow(sub2)
               shan_sub = (sub1n*entropy(sub1$y)+sub2n*entropy(sub2$y))/n
               gain = shan_tot - shan_sub
               return(gain)
               }  
               result = sapply(unique(data$z), function(x) entropy_gain(data,x))
               data$z[which.max(result)] ## Split by which z value is the best?    
#
## [4]Transform data distribution / Form
##    <1> Numeric -> categorical
##       (1) Numeric -> Categorical
              HIGH = ifelse (view1$IR >= .7, "Yes","No") #Logic level[1]
##       (2) Numeric -> Categorical (bin)
		      bins = 10
              cutpoints = quantile(x,(0:bins)/bins)
              binned = cut(var,cutpoints,include.lowest=TRUE)
##       (3) Numeric -> Categorical (bin)2
              groupvec = c(0,20,40,60,80,100)
              labels = c('0-20','20-40','40-60','60-80','80-100')
              data$newvar = cut(data$numeric_var, breaks=groupvec,labels=labels, include.lowest=TRUE)
##       (4) Numeric -> Categorical (Entropy Gain)
##           - entropy value
               entropy = function(x) {
               labelcounts = table(x)
               prob = labelcounts/sum(labelcounts)
               shannonent = -sum(prob * log(prob,2))
               return(shannonent)
               }
##           - compare entropy gain by split (z - continous)
               entropy_gain = function(x,y,value) {
               n = length(y)
               shan_tot = entropy(y)
               subx1 = x[x<=value]
               subx2 = x[x>value]
               suby1 = y[x<=value]
               suby2 = y[x>value]
               n1 = length(suby1)
               n2 = length(suby2)
               shan_sub = (n1* entropy(suby1)+n2*entropy(suby2))/n
               gain = shan_tot - shan_sub
               return(gain)
               }
               result = sapply(unique(data$z), function(x) entropy_gain(data,x))
               data$z[which.max(result)] ## Split by which z value is the best?	
##       (5) Create Dummy Var
               install.packages("caret")
               library(caret)
               v = dummyVars(~ca_var, data) 
               v = data.frame(predict(v,data)) ##Dummy vars
			   
##    <2> Categorical -> Numeric
##       (1) Categorical <- Numeric
			  Transform.num = function (x, y, data1) {
                                        class_tot = data.frame(table(data1[,x]))
                                        encoded = numeric(length(data1[,x]))
                                        noise = runif(length(data1[,x]),0.9,1.1)
                                       for (i in 1:length(data1[,x])) {
                                        encoded[i] = round(sum(as.numeric(data1[,y][data1[,x] == data1[,x][i]][-i]))
                                         /(class_tot$Freq[class_tot$Var1 == data1[,x][i]] - 1), 
                                        digit = 3)
                                        }
                                        encoded.final = encoded * noise
                                        encoded.final = round(encoded.final, digit = 3)
                                        } 
##    <3> Tranform to different distribution
##       (1) Box-cox Transform
             install.packages("caret") ##--To linear
             library(caret)
             var = BoxCoxTrans(var)
##    <4> Convert Data Type (character, numeric, integer, complex, logic(T/F)) (vector, matrix =1; data.frame, list = n) #special -> Inf(1/0); NaN(0/0)
          class(a); attributes(a) # see object's attri
##       (1) Convert data type and turn set into data.frame (Coercion)
             data <- transform(data, var1=factor(var1), # treat specially for modeling categorical, better than numeric 1,2
			                         var2=as.numeric(var2),
									 var3=as.character(var3),
									 var4=as.logical(var4))
#           * Special for factor()
              x <- factor(c("yes", "yes", "no"),
			                levels = c("yes", "no")) # in level, the first level is the baseline in modeling 
## [5]Scale / Normalization
##    <1> Normalize Function
          normalize = function (x) {
          return ((x - min(x)) / (max(x) - min(x)))
          }
##    <2> recenter 
          data = scale(data) ## -> Center with 0 and deviate with few digits
##    <3> rescale to 0-1 
          install.packages("reshape")
          library(reshape)
          data = rescaler(data, "range") ## -> Scale data to min 0 and max 1
##    <4> rank 
          install.packages("reshape")
          library(reshape)
          data = reshape(data, "rank") ## -> Interested in relative position of the value rather than actual value
##    <5> log transform
          data = log(data) ## -> convert borad range of positive numbers to a narrow range positive, to extrame large var
          data[data == -Inf] = NA #(infinite value log(0) treated as missing)
##    <6> recenter using median/MAD
          install.packages("reshape")
          library(reshape)
          data = rescaler(data, "robust") ## -> Like recenter, yet more robust to outliers
#
## [6]Create emplty table / Convert table
##    <1> Create empty table
##        (1) Matrix
              number = c(#,#,#,#,#,#,#,#,#,#)
              table.matrix = matrix(table.matrix, byrow = T/F, ncol = 5 /nrow = 5)
              colnames(table.matrix) = paste0("xx", 1:100) ## xx1, xx2, ...
              rownames(table.matrix) = paste0("zz", 1:100) ## zz1, zz2, ...
#              Or:
              dimnames(table.matrix) = list(paste0("zz", 1:100), paste0("xx", 1:100))
##        (2) Vector
              x <- vector("numeric", length = 10) # define class, numbers
##        (3) List
              x <- list(); x <- data.frame()
##     - 
##    <2> Convert table
##     - 
#
## [7]Merge / Re-arrange Dataset
##    <1> Merge Data
          view1.merged = merge(dataset1, dataset2, by = "primary key") # --> simple primary key (the same)
          view1.merged = merge(dataset1, dataset2, by = c("primary key", "Secondary key")) # --> two match keys
          view1.merged = merge(dataset1, dataset2, by.x = "primary key1", by.y = "primary key2") # --> primary key with different names
          view1.merged = merge(dataset1, dataset2, by = "primary key", all=TRUE) # --> (Default-inner join) out join
          view1.merged = merge(dataset1, dataset2, by = "primary key", all.x=TRUE) # --> LEFT JOIN
          view1.merged = merge(dataset1, dataset2, by = "primary key", all.y=TRUE) # --> RIGHT JOIN
##    <2> Re-arrange Data
##        (1) Wide to long / Long to wide "Melt"
              install.packages("reshape2") ##"melt" (wide to long) / based and repeat on id var1 to stack all other columns into "value" column, you can rename "variable" column to identify their original column name
              library(reshape2)
              data.long = melt(data, id = "var1", variable.name = "xxxxx", value.name = "xxxxx")
              levels(data.long$variable) = c("var2.name", "var3.name", "var4.name", ...)
##           - "cast" (long to wide) / convert melt back
              install.packages("reshape2")
              library(reshape2)
              data.wide = dcast(data.long, var1~variable, value.var="value")
#
## [8]Imputation -> NA, NaN (undefined mathmatical operation)
##    (1) General - is.na() > is.nan() = logical vector (is.na covers all)
       view1 = (na.omit(view1)) ##remove na value
	   bad <- is.na(x); x[!bad] ##remove na value
	   view1$var[view1$var == ""] = NA ## -- Replace "missing" to NA
       view1$var = ifelse(view1$var >= 5 & view1$var < 30, view1$var, NA) ## -- Convert extrame case into "NA"
##        - Get row numbers of missing observs
       which(is.na(var))
       good <- complete.cases(data); data[good] ##Return "True/False" for all rows; get complete cases
##    (2) Imputation - use an assume AVG for NA and group by another variable
       AVG.FUN = ave(view1$var.na, view1$var.cate, FUN = function (x) mean(x, na.rm = TRUE))
       view1$var.na = ifelse(is.na(view1$var.na), AVG.FUN, view1$var.na)
##    (3) Zero & Constant/Missing
       data[is.na(data)] = 0 ## Known that missing is 0
       data[is.na(data)] = # ## Known that missing is #
##    (4) Mean(Normal distribute)
       data[is.na(data)] = mean(data, na.rm = TRUE)
##    (5) Median(Skewness data)
       data[is.na(data)] = median(data, na.rm = TRUE)
##    (6) Mode(Categorical var)
       data[is.na(data)] = mode(data, na.rm = TRUE)
##    (7) Impute with predict model
       install.packages("mice")
       library(mice)
       flux(data) ##Use the vars has a large "outflux" to build a model preidict missing (load all data with missing)
#        - imputation
       imp = mice(sleep, m=5, seed=1234, printFlag=FALSE) 
#        - build 5 linear models
       fit = with(imp, lm(Dream ~ Span + Gest))
#        - combine 5 lm results
       pooled = pool(fit)
       summary(pooled)
##    (8) Use "VIM" packages kNN to impute missing value
       install.packages("VIM")
       library(VIM)
       data.knn = kNN(data)	   
#
...
## ----------------------{Output}
## [1]Output to local
##    <1> csv
          write.table(view1, file = "view1.csv", sep=",", row.names=FALSE, col.names=TRUE) ## CSV
##    <2> PDF
          pdf("SampleGraph.pdf",width=7,height=5) ## PDF
              x=rnorm(100)
              y=rnorm(100,5,1)
              plot(x,lty=2,lwd=2,col="red")
              lines(y,lty=3,col="green")
          dev.off()
##    <3> Excel
...
##    <4> Textual Data Format (Save a textual format into file and copy the format to new data)
##       (1) copy formatting of data
          y <- data.frame(a = 1, b = "a")
		  dput(y, file = "y.R") # save to file
		  new.y <- dget("y.R") # copy to
		  # y = new.y in format
##		 (2) Deparsing Multiple files and read them back 
          x <- "foo"
		  y <- data.frame(a = 1, b = "a")
		  dump(x("x", "y"), file = "data.R")
		  rm(x, y)
		  source("data.R") # R object shows again
#
## [2]Output to Tableau
      Install.packages("Rserve") ##Link to Tableau
      library(Rserve)
      Rserve()
      SCRIPT_REAL -> sample scripts ## In Tableau Calculated feild ##
      ("
       table.data = data.frame(.arg1, .arg2, .arg3, .arg4); 
       table.lm = lm(.arg1~., data = table.data); 
       table.predict = predict(table.lm, table.data)
       "
       ,sum([Sales]),sum([Factor1]), sum([Factor2]), sum([Factor3])
       )
#
...













