# packages Loading ---- ##
# install.packages("httr")
library(httr)
## ------------------------- ##

# -------------------------- #
#    Data Loading code       #
# -------------------------- #

# Create empty container
auth.file <- vector()
data.all <- data.frame(integer(),
                       factor(),
                       factor(),
                       character(),
                       integer(),
                       character(),
                       integer(),
                       character(),
                       integer(),
                       character(),
                       integer(),
                       integer(),
                       integer(),
                       numeric(),
                       numeric(),
                       numeric())

# Authentication

writeLines("\n\n\n------------------ Initiate --------------------\n\n\n")

for (i in 1:11) {
  setwd("~/auth.file") # --------------------- auth.js dir

  url.auth <- "https://api.appnexus.com/auth"
  url.call <- "https://api.appnexus.com/member"
  url.report <- "https://api.appnexus.com/report"

  auth.file[i] <- paste0(i,"auth.js")
  r <- POST(url.auth, body = upload_file(auth.file[i]))
  print.default("Authenticating the login credentials...")
  print.default("Test server connection...")
  stop_for_status(r)
  r.status <- unlist(http_status(r))
  print.default(paste0("Server Status: ",r.status))
  # Token success
  r.error <- as.vector(content(r)$response$error_code)
  r.success <- c(content(r)$response$token, content(r)$response$status)
  ifelse(is.null(r.error) == T, paste0("Token: ",content(r)$response$token, " ---- Status: ", content(r)$response$status), r.error)

  # Active token
  print.default("Activating Token by loging in...")
  r2 <- GET(url.call)
  stop_for_status(r2)
  print.default("Report Information loading...")
  seat_id <- content(r2)$response$member$id
  print.default(paste0("Report Status: ",content(r2)$response$status))
  print.default(paste0("Seat_ID: ",content(r2)$response$member$id))
  print.default(paste0("Seat_Name: ", content(r2)$response$member$name))

  # Get report id
  setwd("~/report_request.file") # --------------------------- report_request.js dir

  print.default("Retrieving data based on 'Report.js'...")
  r3 <- POST(url.report, body = upload_file("report_request_hour.js"))
  stop_for_status(r3)
  print.default(paste0("Report Building Process Status: ",content(r3)$response$status))
  report_id <- content(r3)$response$report_id


  # Check report status
  setwd("~/data/data.input") # --------------------------- data.input dir

  print.default("Checking downloading status...")
  url.check <- paste0('https://api.appnexus.com/report?id=',report_id)
  url.download <- paste0('https://api.appnexus.com/report-download?id=',report_id)
  repeat {
    r4 <- GET(url.check)
    stop_for_status(r4)
    status <- content(r4)$response$execution_status
    print.default(paste0("Report Status: ", status))
    if (status == 'ready') {
      break
    }
  }
  print.default("Done - Downloading...")
  print.default(paste0("Report Status: ", content(r4)$response$execution_status,
                       " ---- Row Count: ", content(r4)$response$report$row_count,
                       " ---- Report Size: ", content(r4)$response$report$row_count))
  r.download <- GET(url.download)
  print.default("Writing data to local...")
  writeBin(as.vector(r.download[[6]]), "data.input.csv", useBytes=TRUE)
  print.default("Reading data back to R...")
  data.raw.i <- read.csv("data.input.csv", header = T, sep = ",", stringsAsFactors = FALSE)
  print.default("Deleting local file...")
  if (file.exists("data.input.csv")) file.remove("data.input.csv")
  print.default("Deleting R objects...")
  rm(r, r2, r3, r4, r.download,
     r.error, r.status, r.success, report_id, status,
     url.auth, url.call, url.check,
     url.download, url.report)


  # -------------------------- #
  #        Data ETL code       #
  # -------------------------- #
  setwd("~/data/data.output") # --------------------------- data.output dir

  print.default("Creating Seat_ID, IO_Type, Lineitem_Type...")
  Seat_ID <- rep(seat_id,nrow(data.raw.i))
  IO_Type <- as.matrix(unique(ifelse(grepl('-cp$',data.raw.i$line_item_name, ignore.case = T),data.raw.i$insertion_order_name,
                                     ifelse(grepl('-manual$',data.raw.i$line_item_name, ignore.case = T), data.raw.i$insertion_order_name, "Manual_IO"))))
  filter <- IO_Type[IO_Type != "Manual_IO"]
  IO_Type <- ifelse(data.raw.i$insertion_order_name %in% filter, "Copilot_IO", "Manual_IO")
  lineitem_Type <-ifelse(!grepl('(-cp$)|(manual$)',data.raw.i$line_item_name, ignore.case = T),"Manual",
                           ifelse(grepl('manual$',data.raw.i$line_item_name, ignore.case = F),"Man-cp",
                             ifelse(grepl('DEV',data.raw.i$line_item_name, ignore.case = F),"Dev-cp", "Pro-cp")))
  print.default("Adding those features...")
  data.raw.i <- cbind(Seat_ID, IO_Type, lineitem_Type, data.raw.i)
  print.default("Done...")
  print.default(paste0("Generating Statistics for Seat Table: ", seat_id, "..."))
  print.default(paste0("Obervation: ",dim(data.raw.i)[1], " ---- Variables: ", dim(data.raw.i)[2]))
  # Fill in "Container"
  print.default(paste0("Adding Seat ", seat_id, "'s Data to final file..."))
  data.all <- rbind(data.raw.i, data.all)
  warning(paste0("Seat: ", seat_id, " exported completely."))
  print.default("Removing local R objects...")
  rm(data.raw.i, seat_id, Seat_ID, IO_Type, lineitem_Type, filter)
  writeLines("\n\n\n------------------ End --------------------\n\n\n")
}

# Saving file to local #
print.default("Saving All file back to local...")
write.csv(data.all, file = "report_log.csv", row.names=FALSE)