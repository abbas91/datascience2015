## ------------------------- Shiny Application Development ---------------------------------- ##

# [1] --------------- Set up Shiny io server account ------------------- #
Go to "http://www.shinyapps.io/"
Sign up -> Log in

# Software Download
R; R studio; rtools # on dir - C:\

# Open R / R studio
find_rtools()
install.packages("shiny")
library(shiny)
install.packages('devtools')
library(devtools)
devtools::install_github('rstudio/rsconnect')
library(rsconnect)
devtools::install_github('rstudio/shinyapps')

# Login Shiny account -> login your server instance via your R session
# Show authentication >>> Show secret; copy whole code
shinyapps::setAccountInfo(name='copilot-report-dash',
                          token='E1561DF5AF6276DC62823DD00494BA7D',
                          secret='GdXHDSDUav1ma+/vwCqq4P+nTKLvg9OkCPAggJul')
Sys.setlocale(locale="English") # If need it in debugging
library(shinyapps) # Once authenticated can run / Might need more packages
# library(packrat)
# library(RJSONIO)
# library(yaml)

# Run app to test on local ************
setwd "\\...\\...\\myapp_R"
runApp("myapp_R", display.mode = "showcase")

# Deploy an app **********
library(rsconnect)
deployApp("myapp_R")

# Check the app on UI shiny io server
Go to "http://www.shinyapps.io/"
Sign up -> Log in
Applications -> Running -> URL # for share







# [2] ----------- Shiny Application file sturcture ------------------- #
shiny_app_folder {(www), (data), helper.R, server.R, ui.R}
(www) # Folder for pics
(data) # folder for data
helper.R # complex R function script can be "sourced" in server.R
server.R # scrpit handle the backend function of the application
ui.R # script functioned as html, css for the user interface function








# [3] --------------------------- ui.R ---------------------------------- #
>>>>Main frame:
# Template 1
shinyUI(fluidPage(
  titlePanel("title panel"),

  sidebarLayout(
    sidebarPanel( "sidebar panel"),
    mainPanel("main panel")
  )
))
# more -> http://shiny.rstudio.com/articles/layout-guide.html

>>>>Elements:
# shiny function HTML5 equivalent creates
p 	<p> 	A paragraph of text : p("xxxxxxxxxxxxxxxxx, xxxxxxxxxxxxxxxxxx.")
h1 	<h1> 	A first level header : >> h1("title", align = "center")
h2 	<h2> 	A second level header
h3 	<h3> 	A third level header
h4 	<h4> 	A fourth level header
h5 	<h5> 	A fifth level header
h6 	<h6> 	A sixth level header
a 	<a> 	A hyper link : >> a("Shiny homepage.", href = "http://www.rstudio.com/shiny"))
br 	<br> 	A line break (e.g. a blank line)
div 	<div> 	A division of text with a uniform style
span 	<span> 	An in-line division of text with a uniform style : >> span("RStudio", style = "color:blue")
pre 	<pre> 	Text ‘as is’ in a fixed width font
code 	<code> 	A formatted block of code: >> code('install.packages("shiny")')
img 	<img> 	An image : >> img(src = "my_image.png", height = 72, width = 72)
strong 	<strong> 	Bold text: >> strong("spreadsheets")
em 	<em> 	Italicized text
HTML 	  	Directly passes a character string as HTML code
helptext : >> helpText("Note: help")

# control wedgets
"function" 	                                "widget"
actionButton 	                         Action Button
checkboxGroupInput 	                     A group of check boxes
checkboxInput 	                         A single check box
dateInput 	                             A calendar to aid date selection
dateRangeInput 	                         A pair of calendars for selecting a date range
fileInput 	                             A file upload control wizard
helpText 	                             Help text that can be added to an input form
numericInput 	                         A field to enter numbers
radioButtons 	                         A set of radio buttons
selectInput 	                         A box with choices to select from
sliderInput 	                         A slider bar
submitButton 	                         A submit button
textInput 	                             A field to enter text
#two arguments - a name (will use internally) / a label (will show i interface)
# 
      actionButton("action", label = "Action")
# 
      submitButton("Submit")

# 
      checkboxInput("checkbox", label = "Choice A", value = TRUE)
# 
      checkboxGroupInput("checkGroup", 
        label = h3("Checkbox group"), 
        choices = list("Choice 1" = 1, 
           "Choice 2" = 2, "Choice 3" = 3),
        selected = 1)
# 
      dateInput("date", 
        label = h3("Date input"), 
        value = "2014-01-01")   
# 
      dateRangeInput("dates", label = h3("Date range"))
    
# 
      fileInput("file", label = h3("File input"))
# 
      numericInput("num", 
        label = h3("Numeric input"), 
        value = 1))   
# 
      radioButtons("radio", label = h3("Radio buttons"),
        choices = list("Choice 1" = 1, "Choice 2" = 2,
                       "Choice 3" = 3),selected = 1)),
# 
      selectInput("select", label = h3("Select box"), 
        choices = list("Choice 1" = 1, "Choice 2" = 2,
                       "Choice 3" = 3), selected = 1)),
# 
      sliderInput("slider1", label = h3("Sliders"),
        min = 0, max = 100, value = 50),
      sliderInput("slider2", "",
        min = 0, max = 100, value = c(25, 75))
# 
      textInput("text", label = h3("Text input"), 
        value = "Enter text...")   
# Display server.R result
"Output_function" 	             "creates"
htmlOutput 	                      raw HTML
imageOutput 	                    image
plotOutput 	                      plot
tableOutput 	                    table
textOutput 	                      text
uiOutput 	                        raw HTML
verbatimTextOutput 	              text

"--------------------------- example":
# shiny ui
shinyUI(fluidPage(
  titlePanel("censusVis"),
  
  sidebarLayout(
    sidebarPanel(
      helpText("Create demographic maps with 
        information from the 2010 US Census."),
      
      selectInput("var", 
        label = "Choose a variable to display",
        choices = c("Percent White", "Percent Black",
          "Percent Hispanic", "Percent Asian"),
        selected = "Percent White"),
      
      sliderInput("range", 
        label = "Range of interest:",
        min = 0, max = 100, value = c(0, 100))
    ),
    
    mainPanel(
      textOutput("text1"),
      textOutput("text2")
    )
  )
))




# [4] ----------------------------- server.R ------------------------------ #
>>>>Main frame:
# Template 1
shinyServer(function(input, output) {

     output$object <- render*({ 
          "process..."
     })

  }
)
# <input> holds user input value; <output> holders server processed objects := input$var, output$plot to use interchange objects
input$var; output$object

>>>>Elements:
# Each entry to output should contain the output of one of Shiny’s render* functions
"render* function"
renderImage 	  images (saved as a link to a source file)
renderPlot 	      plots
renderPrint 	  any printed output
renderTable 	  data frame, matrix, other table like structures
renderText 	      character strings
renderUI 	      a Shiny tag object or HTML

# use input values in server.R
output$text1 <- renderText({ 
      paste("You have selected", input$var)
    })

# execution path in server body
# server.R
[code1]...
- shinyServer(function(input, output) {
-
-	
-     [code2]...
-      -
-      - 
-      -
-      -> "run once each time a user visits the app"[2]
-
-      output$object <- render* ({
-
-          [code3]... 
-           -
-           -
-           -> "run each time a user changes a widget that output$object relies on"[3]
-	  })
-
-})
-
-
-
-> "run once when app is launched"[1]

# Reactive expressions
"You can limit what gets re-run during a reaction with reactive expressions."
# especially when you have a data source lively connected
# ex. - warp "input data" with "reactive" function to make it smater (re-run data only if user changed widgets)
dataInput <- reactive({
  getSymbols(input$symb, src = "yahoo", 
    from = input$dates[1],
    to = input$dates[2],
    auto.assign = FALSE)
})
#-- use this "dataInput()" instead of original input
output$plot <- renderPlot({    
  chartSeries(dataInput(), theme = chartTheme("white"), 
    type = "line", log.scale = input$log, TA = NULL)
})


"--------------------------- example":
# shiny server
library(maps)
library(mapproj)
counties <- readRDS("data/counties.rds")
source("helpers.R")

shinyServer(
  function(input, output) {

    objectA <- c(1,2,3,4,4,5)

    output$text1 <- renderText({ 
      paste("You have selected", input$var)
    })
    
    output$text2 <- renderText({ 
      paste("You have chosen a range that goes from",
        input$range[1], "to", input$range[2])
    })
    
  }
)


# [4] ----------------------------- helpers.R ------------------------------ #
# Template 1
# comments .................................
# ..........................................
# ..........................................

helper.function <- function (var, var2, ...) {
	code...
}

# use it in server.R
source("helpers.R")




