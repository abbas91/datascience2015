# shiny ui

shinyUI(fluidPage(
   titlePanel(h1("Copilot Dashboard")),

   sidebarLayout(
     sidebarPanel(
         helpText("Data source: Appnexus API datafeed"),
         br(),
         # uiOutput("Seat_List"),

         uiOutput("IO_List"),

         img(src = "Co-pilot.jpg", heigh = 100, width = 150),
         p("Please find more project information at: ", a("Co-pilot Conference Page - JIRA", href = "https://confluence.xaxis.com/display/XENG/Co-Pilot"))
        ),
     mainPanel(
        h2("Lineitem Type Performance by IO", align = "top"),
         br(),
        tabsetPanel( 
             tabPanel("Line-item View - Total", 
                      dataTableOutput("mytable1")),
             tabPanel("Line-item View - By Last 7 Days ", 
                      dataTableOutput("mytable2"))   
        )
    )
 )
))