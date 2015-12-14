# shiny server
         library(shiny)
         #install.packages("DBI")
         library(DBI)
         #install.packages("RMySQL")
         library(RMySQL)
         #install.packages("dplyr")
         library(dplyr)

percent <- function(x, digits = 2, format = "f", ...) {
                       paste0(formatC(100 * x, format = format, digits = digits, ...), "%")
}

pretty.number <- function(x, big.mark=",", scientific=FALSE, ...) {
                       prettyNum(x, big.mark = big.mark,scientific = scientific)
}

format.money  <- function(x, ...) {
  paste0("$", formatC(as.numeric(x), format="f", digits=2, big.mark=","))
}



shinyServer(function(input, output) {

# Connect MySQL
mydb <- dbConnect(MySQL(),
                  user='mark2015',
                  password='fghfgh67743185',
                  dbname='mysql_copilot_dash_1',
                  host='mysql-copilot-dash-instance1.cmzj9dljhdnd.us-east-1.rds.amazonaws.com')

# Create Data Set (table) - All
data_sql.all <- dbSendQuery(mydb, "select Seat_ID, insertion_order_name, line_item_name, line_item_id, 
                            IO_Type, lineitem_Type, date(hour) as Date1,
                            sum(imps) as IMPs,
                            sum(clicks) as CLICKs, IFNULL(sum(clicks) / sum(imps),0) as CTR,
                            sum(total_convs) as TOT_CONVs,
                            sum(cost) as COST,
                            sum(revenue) as REV,
                            sum(profit) as PRFT,
                            IFNULL(sum(cost) / sum(imps),0) * 1000 as CPM,
                            IFNULL(sum(cost) / sum(clicks),0) as CPC,
                            IFNULL(sum(cost) / sum(total_convs),0) as CPA
                            from copilot_dash_log
                            group by Seat_ID, insertion_order_name, lineitem_Type, date(hour), line_item_name, line_item_id
                            order by Seat_ID, lineitem_Type, Date1"
)
data_sql.all <- fetch(data_sql.all, n=-1)



# Quick format on two IDs
data_sql.all$Seat_ID <- as.character(data_sql.all$Seat_ID)
data_sql.all$line_item_id <- as.character(data_sql.all$line_item_id)



# Create Selection vars / formating in memory
IO_List <- unique(data_sql.all$insertion_order_name[data_sql.all$IO_Type =='Copilot_IO'])



# Reactive Selection list generation
output$IO_List <- renderUI({
    selectInput(inputId = 'IO', h4("Please Choose an IO to Display"), as.list(IO_List))
    })


# output$Seat_List <- renderUI({
#     selectInput(inputId = 'Seat', "Choose a Seat_ID to display", as.list(Seat_List))
#     })



benchmark <- function() {
# Building benchmark table
# Gloabl -
global <- data_sql.all %>%
          summarize(CTR=ifelse(is.nan(sum(CLICKs) / sum(IMPs)),0,sum(CLICKs) / sum(IMPs)),
                    MARGIN=sum(PRFT) / sum(REV), # new
                    CPM=ifelse(is.na(sum(COST) / sum(IMPs)*1000),0,sum(COST) / sum(IMPs)*1000),
                    CPC=ifelse(is.nan(sum(COST) / sum(CLICKs)),0,sum(COST) / sum(CLICKs)),
                    CPA=ifelse(is.na(sum(COST) / sum(TOT_CONVs)),0,sum(COST) / sum(TOT_CONVs))) %>%
          transform(CTR=percent(CTR),
                    MARGIN=percent(MARGIN), # new
                    CPM=format.money(CPM),
                    CPC=format.money(CPC),
                    CPA=format.money(CPA))
# Seat -
Seat <- data_sql.all %>%
        filter(Date1 > as.Date(input$dates[1])-1 &
                               Date1 < as.Date(input$dates[2])+1 &
                               Seat_ID == unique(data_sql.all$Seat_ID[data_sql.all$insertion_order_name == input$IO])) %>%
        group_by(Seat_ID) %>%
        summarize(CTR=ifelse(is.nan(sum(CLICKs) / sum(IMPs)),0,sum(CLICKs) / sum(IMPs)),
                  MARGIN=sum(PRFT) / sum(REV), # new
                  CPM=ifelse(is.na(sum(COST) / sum(IMPs)*1000),0,sum(COST) / sum(IMPs)*1000),
                  CPC=ifelse(is.nan(sum(COST) / sum(CLICKs)),0,sum(COST) / sum(CLICKs)),
                  CPA=ifelse(is.na(sum(COST) / sum(TOT_CONVs)),0,sum(COST) / sum(TOT_CONVs))) %>%
        summarize(CTR=mean(CTR[is.finite(CTR)]),
                  MARGIN=mean(MARGIN[is.finite(MARGIN)]), # new
                  CPM=mean(CPM[is.finite(CPM)]),
                  CPC=mean(CPC[is.finite(CPC)]),
                  CPA=mean(CPA[is.finite(CPA)])) %>%
        transform(CTR=percent(CTR),
                  MARGIN=percent(MARGIN), # new
                  CPM=format.money(CPM),
                  CPC=format.money(CPC),
                  CPA=format.money(CPA))
# Lineitem
Lineitem <- data_sql.all %>%
            filter(Date1 > as.Date(input$dates[1])-1 &
                                   Date1 < as.Date(input$dates[2])+1 &
                                   Seat_ID == unique(data_sql.all$Seat_ID[data_sql.all$insertion_order_name == input$IO])) %>%
            group_by(line_item_id) %>%
            summarize(CTR=ifelse(is.nan(sum(CLICKs) / sum(IMPs)),0,sum(CLICKs) / sum(IMPs)),
                      MARGIN=sum(PRFT) / sum(REV), # new
                      CPM=ifelse(is.na(sum(COST) / sum(IMPs)*1000),0,sum(COST) / sum(IMPs)*1000),
                      CPC=ifelse(is.nan(sum(COST) / sum(CLICKs)),0,sum(COST) / sum(CLICKs)),
                      CPA=ifelse(is.na(sum(COST) / sum(TOT_CONVs)),0,sum(COST) / sum(TOT_CONVs))) %>%
            summarize(CTR=mean(CTR[is.finite(CTR)]),
                      MARGIN=mean(MARGIN[is.finite(MARGIN)]), # new
                      CPM=mean(CPM[is.finite(CPM)]),
                      CPC=mean(CPC[is.finite(CPC)]),
                      CPA=mean(CPA[is.finite(CPA)])) %>%
            transform(CTR=percent(CTR),
                      MARGIN=percent(MARGIN), # new
                      CPM=format.money(CPM),
                      CPC=format.money(CPC),
                      CPA=format.money(CPA))
# combine
Benchmark_Scale <- c("Global_AVG", "Seat_AVG", "Lineitem_AVG")
benchmark <- data.frame(rbind(global, Seat, Lineitem))
benchmark <- data.frame(cbind(Benchmark_Scale, benchmark))

return (benchmark)
}


Market <- function() {

NAM <- data_sql.all %>%
       filter(Date1 > as.Date(input$dates[1])-1 &
                              Date1 < as.Date(input$dates[2])+1 &
                               Seat_ID == c("1661", "1662")) %>%
        summarize(CTR=ifelse(is.nan(sum(CLICKs) / sum(IMPs)),0,sum(CLICKs) / sum(IMPs)),
                  MARGIN=sum(PRFT) / sum(REV), # new
                  CPM=ifelse(is.na(sum(COST) / sum(IMPs)*1000),0,sum(COST) / sum(IMPs)*1000),
                  CPC=ifelse(is.nan(sum(COST) / sum(CLICKs)),0,sum(COST) / sum(CLICKs)),
                  CPA=ifelse(is.na(sum(COST) / sum(TOT_CONVs)),0,sum(COST) / sum(TOT_CONVs))) %>%
        summarize(CTR=mean(CTR[is.finite(CTR)]),
                  MARGIN=mean(MARGIN[is.finite(MARGIN)]), # new
                  CPM=mean(CPM[is.finite(CPM)]),
                  CPC=mean(CPC[is.finite(CPC)]),
                  CPA=mean(CPA[is.finite(CPA)])) %>%
        transform(CTR=percent(CTR),
                  MARGIN=percent(MARGIN), # new
                  CPM=format.money(CPM),
                  CPC=format.money(CPC),
                  CPA=format.money(CPA))

APAC <- data_sql.all %>%
       filter(Date1 > as.Date(input$dates[1])-1 &
                              Date1 < as.Date(input$dates[2])+1 &
                               Seat_ID == c("1832", "1837", "1841", "1843")) %>%
        summarize(CTR=ifelse(is.nan(sum(CLICKs) / sum(IMPs)),0,sum(CLICKs) / sum(IMPs)),
                  MARGIN=sum(PRFT) / sum(REV), # new
                  CPM=ifelse(is.na(sum(COST) / sum(IMPs)*1000),0,sum(COST) / sum(IMPs)*1000),
                  CPC=ifelse(is.nan(sum(COST) / sum(CLICKs)),0,sum(COST) / sum(CLICKs)),
                  CPA=ifelse(is.na(sum(COST) / sum(TOT_CONVs)),0,sum(COST) / sum(TOT_CONVs))) %>%
        summarize(CTR=mean(CTR[is.finite(CTR)]),
                  MARGIN=mean(MARGIN[is.finite(MARGIN)]), # new
                  CPM=mean(CPM[is.finite(CPM)]),
                  CPC=mean(CPC[is.finite(CPC)]),
                  CPA=mean(CPA[is.finite(CPA)])) %>%
        transform(CTR=percent(CTR),
                  MARGIN=percent(MARGIN), # new
                  CPM=format.money(CPM),
                  CPC=format.money(CPC),
                  CPA=format.money(CPA))

EMEA <- data_sql.all %>%
       filter(Date1 > as.Date(input$dates[1])-1 &
                              Date1 < as.Date(input$dates[2])+1 &
                               Seat_ID == c("1787", "1845", "2728", "1849", "1785")) %>%
        summarize(CTR=ifelse(is.nan(sum(CLICKs) / sum(IMPs)),0,sum(CLICKs) / sum(IMPs)),
                  MARGIN=sum(PRFT) / sum(REV), # new
                  CPM=ifelse(is.na(sum(COST) / sum(IMPs)*1000),0,sum(COST) / sum(IMPs)*1000),
                  CPC=ifelse(is.nan(sum(COST) / sum(CLICKs)),0,sum(COST) / sum(CLICKs)),
                  CPA=ifelse(is.na(sum(COST) / sum(TOT_CONVs)),0,sum(COST) / sum(TOT_CONVs))) %>%
        summarize(CTR=mean(CTR[is.finite(CTR)]),
                  MARGIN=mean(MARGIN[is.finite(MARGIN)]), # new
                  CPM=mean(CPM[is.finite(CPM)]),
                  CPC=mean(CPC[is.finite(CPC)]),
                  CPA=mean(CPA[is.finite(CPA)])) %>%
        transform(CTR=percent(CTR),
                  MARGIN=percent(MARGIN), # new
                  CPM=format.money(CPM),
                  CPC=format.money(CPC),
                  CPA=format.money(CPA))
name <- data.frame(c("NA", "APAC", "EMEA"))
colnames(name) <- "Market_Scale"
market <- data.frame(rbind(NAM, APAC, EMEA))
market <- data.frame(cbind(name, market))

return (market)

}






# Summary Data table1 generation
output$mytable1 <- renderDataTable({
    subset(data_sql <- data_sql.all %>%
                       filter(Date1 > as.Date(input$dates[1])-1 &
                              Date1 < as.Date(input$dates[2])+1 &
                              insertion_order_name == input$IO) %>%
                       group_by(Seat_ID, insertion_order_name,
                                lineitem_Type) %>%
                       summarize(IMPs=sum(IMPs),
                                 CLICKs=sum(CLICKs),
                                 CTR=ifelse(is.nan(sum(CLICKs) / sum(IMPs)),0,sum(CLICKs) / sum(IMPs)),
                                 TOT_CONVs=sum(TOT_CONVs),
                                 COST=sum(COST),
                                 REV=sum(REV),
                                 PRFT=sum(PRFT),
                                 MARGIN=sum(PRFT) / sum(REV), # new
                                 CPM=ifelse(is.na(sum(COST) / sum(IMPs)*1000),0,sum(COST) / sum(IMPs)*1000),
                                 CPC=ifelse(is.nan(sum(COST) / sum(CLICKs)),0,sum(COST) / sum(CLICKs)),
                                 CPA=ifelse(is.na(sum(COST) / sum(TOT_CONVs)),0,sum(COST) / sum(TOT_CONVs))) %>%
                       transform(IMPs=pretty.number(IMPs),
                                 CLICKs=pretty.number(CLICKs),
                                 CTR=percent(CTR),
                                 TOT_CONVs=pretty.number(TOT_CONVs),
                                 COST=format.money(COST),
                                 REV=format.money(REV),
                                 PRFT=format.money(PRFT),
                                 MARGIN=percent(MARGIN), # new
                                 CPM=format.money(CPM),
                                 CPC=format.money(CPC),
                                 CPA=format.money(CPA))
                       )

   })


# Summary Data table2 generation
output$mytable2 <- renderDataTable({
    subset(data_sql.hour <- data_sql.all %>%
                       filter(Date1 > as.Date(input$dates[1])-1 &
                              Date1 < as.Date(input$dates[2])+1 &
                              insertion_order_name == input$IO) %>%
                       group_by(Seat_ID, insertion_order_name,
                                lineitem_Type, Date1) %>%
                       summarize(IMPs=sum(IMPs),
                                 CLICKs=sum(CLICKs),
                                 CTR=ifelse(is.nan(sum(CLICKs) / sum(IMPs)),0,sum(CLICKs) / sum(IMPs)),
                                 TOT_CONVs=sum(TOT_CONVs),
                                 COST=sum(COST),
                                 REV=sum(REV),
                                 PRFT=sum(PRFT),
                                 MARGIN=sum(PRFT) / sum(REV), # new
                                 CPM=ifelse(is.na(sum(COST) / sum(IMPs)*1000),0,sum(COST) / sum(IMPs)*1000),
                                 CPC=ifelse(is.nan(sum(COST) / sum(CLICKs)),0,sum(COST) / sum(CLICKs)),
                                 CPA=ifelse(is.na(sum(COST) / sum(TOT_CONVs)),0,sum(COST) / sum(TOT_CONVs))) %>%
                       transform(IMPs=pretty.number(IMPs),
                                 CLICKs=pretty.number(CLICKs),
                                 CTR=percent(CTR),
                                 TOT_CONVs=pretty.number(TOT_CONVs),
                                 COST=format.money(COST),
                                 REV=format.money(REV),
                                 PRFT=format.money(PRFT),
                                 MARGIN=percent(MARGIN), # new
                                 CPM=format.money(CPM),
                                 CPC=format.money(CPC),
                                 CPA=format.money(CPA))
                       )


   })


# Summary Data table3 generation
output$mytable3 <- renderDataTable({
    subset(data_sql.line <- data_sql.all %>%
                       filter(Date1 > as.Date(input$dates[1])-1 &
                              Date1 < as.Date(input$dates[2])+1 &
                              insertion_order_name == input$IO) %>%
                       group_by(Seat_ID, insertion_order_name,
                                lineitem_Type, line_item_name,
                                line_item_id) %>%
                       summarize(IMPs=sum(IMPs),
                                 CLICKs=sum(CLICKs),
                                 CTR=ifelse(is.nan(sum(CLICKs) / sum(IMPs)),0,sum(CLICKs) / sum(IMPs)),
                                 TOT_CONVs=sum(TOT_CONVs),
                                 COST=sum(COST),
                                 REV=sum(REV),
                                 PRFT=sum(PRFT),
                                 MARGIN=sum(PRFT) / sum(REV), # new
                                 CPM=ifelse(is.na(sum(COST) / sum(IMPs)*1000),0,sum(COST) / sum(IMPs)*1000),
                                 CPC=ifelse(is.nan(sum(COST) / sum(CLICKs)),0,sum(COST) / sum(CLICKs)),
                                 CPA=ifelse(is.na(sum(COST) / sum(TOT_CONVs)),0,sum(COST) / sum(TOT_CONVs))) %>%
                       transform(IMPs=pretty.number(IMPs),
                                 CLICKs=pretty.number(CLICKs),
                                 CTR=percent(CTR),
                                 TOT_CONVs=pretty.number(TOT_CONVs),
                                 COST=format.money(COST),
                                 REV=format.money(REV),
                                 PRFT=format.money(PRFT),
                                 MARGIN=percent(MARGIN), # new
                                 CPM=format.money(CPM),
                                 CPC=format.money(CPC),
                                 CPA=format.money(CPA))
                       )


   })


# Summary benchmark by global, seat, and lineitem
output$mytable4 <- renderDataTable({
    subset(benchmark())

   })


output$mytable5 <- renderDataTable({
    subset(Market())

   })

})



