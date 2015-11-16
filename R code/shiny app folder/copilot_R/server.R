# shiny server

         #install.packages("DBI")
         library(DBI)
         #install.packages("RMySQL")
         library(RMySQL)
         # install.packages("dplyr")
         # library(dplyr)

percent <- function(x, digits = 2, format = "f", ...) {
                       paste0(formatC(100 * x, format = format, digits = digits, ...), "%")
}

pretty <- function(x, big.mark=",", scientific=FALSE, ...) {
                       prettyNum(x, big.mark = big.mark,scientific = scientific)
}


shinyServer(function(input, output) {

# Connect MySQL
mydb <- dbConnect(MySQL(), 
                 user='mark2015', 
                 password='fghfgh67743185', 
                 dbname='mysql_copilot_dash_1', 
                 host='mysql-copilot-dash-instance1.cmzj9dljhdnd.us-east-1.rds.amazonaws.com')


# Create Data Set (table1)
data_sql <- dbSendQuery(mydb, "select Seat_ID, insertion_order_name, lineitem_Type, 
                                      format(sum(imps),0) as IMPs, 
                                      format(sum(clicks),0) as CLICKs, sum(clicks) / sum(imps) as CTR,
                                      format(sum(total_convs),0) as TOT_CONVs, 
                                      format(sum(cost),2) as COST, 
                                      format(sum(revenue),2) as REV, 
                                      format(sum(profit),2) as PRFT,
                                      format(sum(cost) / sum(imps) * 1000,2) as CPM,
                                      format(sum(cost) / sum(clicks),2) as CPC,
                                      format(sum(cost) / sum(total_convs),2) as CPA
                               from copilot_dash_log
                               where IO_Type = 'Copilot_IO'
                               group by Seat_ID, insertion_order_name, lineitem_Type
                               order by Seat_ID"
                               )
                               # date(hour) as Date1,
data_sql <- fetch(data_sql, n=-1)

# Create Data Set (table2) - By hour
data_sql.hour <- dbSendQuery(mydb, "select date(hour) as Date1, Seat_ID, insertion_order_name, lineitem_Type, 
                                      format(sum(imps),0) as IMPs, 
                                      format(sum(clicks),0) as CLICKs, sum(clicks) / sum(imps) as CTR,
                                      format(sum(total_convs),0) as TOT_CONVs, 
                                      format(sum(cost),2) as COST, 
                                      format(sum(revenue),2) as REV, 
                                      format(sum(profit),2) as PRFT,
                                      format(sum(cost) / sum(imps) * 1000,2) as CPM,
                                      format(sum(cost) / sum(clicks),2) as CPC,
                                      format(sum(cost) / sum(total_convs),2) as CPA
                               from copilot_dash_log
                               where IO_Type = 'Copilot_IO'
                               group by Seat_ID, insertion_order_name, lineitem_Type, date(hour)
                               order by Seat_ID, lineitem_Type, Date1"
                               )
                               # date(hour) as Date1,
data_sql.hour <- fetch(data_sql.hour, n=-1)


# Table 1 summary *DONT USE*
# group <- group_by(data_sql.raw, Seat_ID, insertion_order_name, lineitem_Type, IMPs, CLICKs, CTR,
#                                 TOT_CONVs, COST, REV, PRFT, CPM, CPC, CPA)
# data_sql <- summarize(group, IMPs = sum(IMPs), CLICKs = sum(CLICKs), CTR = avg(CTR),
#                              TOT_CONVs = sum(TOT_CONVs), COST = sum(COST), )


# Create Selection vars in memory
IO_List <- unique(data_sql$insertion_order_name)
Seat_List <- unique(data_sql$Seat_ID)
data_sql$CTR <- percent(data_sql$CTR)
data_sql.hour$CTR <- percent(data_sql.hour$CTR)

# Reactive Selection list generation
output$IO_List <- renderUI({
    selectInput(inputId = 'IO', "Choose a IO to display", as.list(IO_List))
    })

# output$Seat_List <- renderUI({
#     selectInput(inputId = 'Seat', "Choose a Seat_ID to display", as.list(Seat_List))
#     })

# output$hour_list <- renderUI({

#     })



# Summary Data table1 generation
output$mytable1 <- renderDataTable({ 
    subset(data_sql, data_sql$insertion_order_name == input$IO)
                     # & data_sql$Seat_ID == input$Seat)
        
   })


# Summary Data table2 generation
output$mytable2 <- renderDataTable({ 
    subset(data_sql.hour, data_sql.hour$insertion_order_name == input$IO)
                     # & data_sql$Seat_ID == input$Seat)
        
   })
})