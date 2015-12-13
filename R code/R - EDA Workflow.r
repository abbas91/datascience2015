## EDA


## >>>> Check Data Quality / Error
## >>>> Check Data Structure / Distribution
## >>>> Check Data Selection / Relationship / Reduction
## >>>> Visualization Techiques












## -----------------------{Check Data Quality / Error}
## [1] Missing Value
       install.packages("mice")
       library(mice)
       md.pattern(vars) ## n of obsevs of each missing pattern
##     - quick visualize missing status
       install.packages("VIM")
       library(VIM)
       aggr(data)
#
## [2] General Data loaded check
       head(view1,#)
       tail(view1,#)
       names(view1)
       dim(view1)
       str(view1)
       summary(view1)
       unique(var) ##Unique value
#
## [3] Outlier
       plot(predict(view1.lm), rstudent(view1.lm)) ##rstudent > 3 (outliers)
##     - Plot Leverage statistics
       view1.lm = lm(Y~., data = view1)
       hatvalues(view1.lm) ## leverage
       plot(hatvalues(view1.lm)) ## plot leverage
       which.max(hatvalues(view1.lm)) ## the biggest value
##     - Outlier / Leverage points (If greatly exccess the average) 
       plot(rstudent(view1.lm), hatvalues(view1.lm))
#
## [4] Collinearity (VIF) - the less the range to 1 the less collinearity
       library(car)
       vif(view1.lm) ##value >=5 or 10 means collinearity [min = 1]
	   
#
...

## ------------------------{Check Data Structure / Distribution}
## [1] Check dist normality
##    (1) Single var
          qqnorm(view2[,"IR"], main = "IR")
          qqline(view2[,"IR"])
##    (2) All Table
          layout(matrix(1:6, nc = 2))
          sapply(colnames(view2), function (x) {
          qqnorm(view2[[x]], main = x)
          qqline(view2[[x]])
          })
## [2] Data Structure
       round(prop.table(table(data$var)) * 100, digits = 1) ## table prop information of a category var
       quantile(var, probs = c(0.01, 0.99) ##show 1% 99% quantiles
	   round(prop.table(table(view1$var)) * 100, digits = 1) ##Give % of different factors
	   fivenum(var.numeric) #minimum, 25%, median, 75%, Maximum
	   identical(value1, value2) ##Test two objects are exactly the same
##     - Deatail Categorical summary
         install.packages("Hmisc")
         library(Hmisc)
         describe(data) ## Other information besides 'Summary"
##     - Deatil Numeric summary
         install.packages("fBasics")
         library(fBasics)
         basicStats(numeric vars) ## mean sd max min / .95 confidence Lcl mean and Ucl mean
#      - Pivot-plot cases match each group
         install.packages("reshape2")
         library(reshape2)
         dcast(data, ca_var1~ca_var2, value.var = 'var3', fun = length) ##-> count cases match each combination
##     - explore metrics of an numeric var by pivoting an categorical var
         install.packages("plyr")
         library(plyr)
         ddply(data, .(cate_var), function(x) each(mean, length) (x$numer_var)
##     - Advanced Scatter plot
         install.packages("psych")
         library(psych)
         pairs.panels(view1)
##       Others..
         install.packages("car")
         library(car)
         scatterplotMatrix(data, smoother = F)
#
#
#
#
...

## -------------------------{Check Data Selection / Relationship / Reduction}
## [1]Statistical elements
      range(var) ##min max
      diff(range(var)) ##Difference max min
      var(var) ##Variance
      sd(var) ##Standard Deviation
	  dist(scale(view2, center = FALSE)) ##Culmulated distance table
	  cov(view1[, c("var1", "var2", "var3")]) ##Covariance matrix
	  
##    - Find mode (function)
       Mode <- function(x) {
                       ux <- unique(x)
                       ux[which.max(tabulate(match(x, ux)))]
                       }
##    - Skewness
       install.packages("propagate")
       library(propagate)
       skewness(vars) ## Positive or negative skewed
       kurtosis(vars) ## Sharp peak (large vale) or flat peak (small value) or Gaussian peak (0)
##    - Test for normality
       shapiro.test(var) ##Whether normaily distributed   
	   	   
#
## [2]Statistical Test / Model
##    <1> Correlation
          install.packages("corrplot")
          library(corrplot)
          cortable = cor(view1[,c("var1", "var2", "var3")])
          corrplot(cortable)
##       - Correlation between missing value
          ind = as.data.frame(abs(is.na(view1)))
          new.ind = ind[, sapply(ind,sd)!=0]
          install.packages("corrplot")
          library(corrplot)
          cortable = cor(new.ind)
          corrplot(cortable)
##       - Hierarchical Correlation ## Early Join - High correlation (Signal or Group)
          cor.table = cor(data, use="pairwise", method="pearson")
          hc = hclust(dist(cor.table), method="average")
          dn = as.dendrogram(hc)
          plot(dn, horiz = TRUE)
##     <2> F / T Test
          t.test(var1, var2) ##-> whether those two vars are significantly different
          var.test(var1, var2) ##-> Test the equality of variance of two samples
##     <3> Chi-squared Test
          chisq.test(var1, var2) ##If there is correlation
##     <4> Anova / Ancova / Manova / Mancova
##        (1) Anova
##           - One Way Anova (Completely Randomized Design)
               fit = aov(y ~ A, data=view1) 
##           - Randomized Block Design (B is the blocking factor)
               fit = aov(y ~ A + B, data=view1) 
##           - Two Way Factorial Design
               fit = aov(y ~ A + B + A:B, data=view1)
               fit = aov(y ~ A*B, data=view1) # same thing 
##           - One Within Factor
               fit = aov(y~A+Error(Subject/A),data=view1)
##           - Two Within Factors W1 W2, Two Between Factors B1 B2
               fit = aov(y~(W1*W2*B1*B2)+Error(Subject/(W1*W2))+(B1*B2),data=view1) 
##           - Evaluate Result
               summary(fit) # Display type I ANOVA table
               drop1(fit,~.,test="F") # type III SS and F Tests (Cause A+B diff from B+A)
               anova(fit1, fit2) # # type III SS and F Tests (Cause A+B diff from B+A)
##           - Plot
               plot(fit)
##           - Tukey Honestly significant differences
               TukeyHSD(fit, conf.level = 0.95) # where fit comes from aov()
##         (2) Manova
              Y = cbind(y1, y2, y3)
              fit = manova(Y~ A*B)
              summary(fit, test = "pillai") ##Other test options - "Wilks", "Hotelling-Lawley", and "Roy"
              summary.aov(fit) ##Get univariate statisc each ys			   
##     <5> Exame relationship between two categoribles (Cross-tabulations)
           install.packages("gmodels")
           library(gmodels)
           CrossTable(x = var1, y = var2, chisq = TRUE/FALSE) ##values in cell (# of observ, Chi-square stats - each contribution to chi-test, row%, col%, table%) 
#
##     <6> Relationship between Xs and Y linear?
           view1.lm = lm(Y~., data = view1)
           plot(predict(view1.lm), residuals(view1.lm),) ##Linearity
           lines(lowess(predict(view1.lm),residuals(view1.lm)), col="blue")
##     <7> Relationship between X and Y linear? --> Non-linearity / porpotional noise / auto-correlation of noise
           plot(view1.lm$residuals~x) 
#
...


































## --------------------------{Visualization Techiques}
## [1] ggplot package
##   <1> Multiplot FUN 
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
...
#
## [2] base plot package
#
...
















