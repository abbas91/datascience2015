# shiny ui

shinyUI(fluidPage(
   titlePanel(h1("Copilot Dashboard")),

   sidebarLayout(
     sidebarPanel(
     	 helpText("Create demographic maps with information from the 2010 US Census."),
         h3("Appnexus Log Level Data"),
         br(),
         p("This dashbroad will be lively linked to AWS RDS and read data on a daily frequency. You will also be able to extract data from SQL:"),
         br(),
         br(),
         code("SELECT count(*) FROM TableA"),
         br(),
         br(),
         br(),
         img(src = "copilot.jpg", heigh = 200, width = 200), div("Co-pilot Project", style = "color:blue")
     	),
     mainPanel(
         h1("main1", align = "center"),
         h3("main2", align = "center"),
         h2("main3", align = "center"),
         h1("main4", align = "center"),
         selectInput("var", 
                     label = "Choose a variable to display",
                     choices = list("Percent White", "Percent Black", "Percent Hispanic", "Percent Asian"),
                     selected = "Percent White"),
         selectInput("var", 
                     label = "Choose a variable to display",
                     choices = list("Percent White", "Percent Black", "Percent Hispanic", "Percent Asian"),
                     selected = "Percent White"),
         p("Let's talk about something then since the app is running"),
         strong("Let's talk about something then since the app is running"),
         em("Let's talk about something then since the app is running"),
         br(),
         br(),
         p("A new paragraph.."),
         code("SELECT count(*) FROM TableA"),
         div("This is a paragraph with similar style of text", style = "color:blue"),
         br(),
         br(),
         p("The group of A and ", span("the group of B", style = "color:blue"), "are not the same.."),
         img(src = "copilot.jpg", heigh = 100, width = 100),
         p("Please visit: ", a("Google homepage", href = "http://www.google.com"))
     	)
   	)
))