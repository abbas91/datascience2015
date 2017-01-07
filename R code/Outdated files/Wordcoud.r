## Remove Everything -------------- Dont RUN!!!
rm(list=ls(all=TRUE))

##Load in Dataset
setwd("C:/Users/mark.li/Desktop")
keyword.data = read.csv("Zdata.csv", stringsAsFactors = FALSE) 

##Duplicate cases FUN
duplicate = function (data) {
  loop = nrow(data)
  keyword.new = list()
  for (i in 1:loop) {
    keyword.new[[i]] = rep(data[i,1],data[i,2])
  }
  return(keyword.new)
}
## Test Run & Formating
keyword.new.1 = unlist(duplicate(keyword.data))
keyword.new.1 = as.matrix(keyword.new.1)
keyword.new.1 = paste(keyword.new.1, collapse = " ")

##Lower case
keyword.new.1.low = tolower(keyword.new.1)
##Take out numbers, marks, punctuations
keyword.new.1.clean = strsplit(keyword.new.1.low, "\\W")
##Convert "list" to var
keyword.new.1.clean = unlist(keyword.new.1.clean)
##Exclude "space" strings
not.blank = which(keyword.new.1.clean != " ")
keyword.new.1.clean = keyword.new.1.clean[not.blank]
##Exclude "Brand" strings
brand = grep("^z.*(n$|m$)",keyword.new.1.clean, ignore.case = FALSE)
unique(keyword.new.1.clean[brand]) ##Check
keyword.new.1.clean = keyword.new.1.clean[-brand]


## Word Frequency table sorted ##
keyword.new.1.clean.t = table(keyword.new.1.clean)
keyword.new.1.clean.t.sort = sort(keyword.new.1.clean.t, decreasing = TRUE)
## Get Frequency of any words ##
keyword.new.1.clean.t.sort["zicam"]
## Plot Top 10 frequency words ##
plot(keyword.new.1.clean.t.sort[1:10], type = "b", xlab = "Top Ten Words", ylab = "Percentage of full text", xaxt = "n")
axis(1,1:10, labels = names(keyword.new.1.clean.t.sort[1:10]))

## --------- Visualize Word could ------------ ##
install.packages("NLP")
library(NLP)
install.packages("tm")
library(tm)
install.packages("RColorBrewer")
library(RColorBrewer)
install.packages("wordcloud")
library(wordcloud)
## RUN
wordcloud(keyword.new.1.clean, min.freq = 40, random.order = FALSE, scale = c(5, 2))