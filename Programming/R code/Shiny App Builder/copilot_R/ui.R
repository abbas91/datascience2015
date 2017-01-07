# shiny ui

shinyUI(fluidPage(
   titlePanel(h1("Copilot Dashboard")),

   sidebarLayout(
     sidebarPanel(
         helpText("Data source: Appnexus API datafeed"),
         br(),
         # uiOutput("Seat_List"),

         uiOutput("IO_List"),
         dateRangeInput("dates", label = h4("Please Choose the Date Range"),
                                 start = Sys.Date() - 8, end = Sys.Date()),
         helpText("NOTE: a valid date range should be within last 8 days (Apply to tab 1, 2 & 4)", style = "color:green"),

         p("Please find more project information at: ", a("Co-pilot Conference Page - JIRA", href = "https://confluence.xaxis.com/display/XENG/Co-Pilot"))
        ),
     mainPanel(
        h2("Lineitem Type Performance by IO", align = "top"),
         br(),
        tabsetPanel(
             tabPanel("Line-item View - Total",
                      dataTableOutput("mytable1")),
             tabPanel("Line-item View - By Lineitem",
                      dataTableOutput("mytable3")),
             tabPanel("Line-item View - By Date",
                      dataTableOutput("mytable2")),
             tabPanel("Benchmark View - All period",
                      dataTableOutput("mytable4"),
                      helpText("**NOTE: 'Seat_AVG' and 'Lineitem_AVG' are based on the Seat in which the Co-pilot IO is chosen", style = "color:green"),
                      dataTableOutput("mytable5"),
                      helpText("**NOTE: Markets grouped from current 11 seats", style = "color:green"))
        )
    )
 )
))
