# Takes in JSON file with one auction per row and parses it into a data frame
# Data frame has one impression per row and 23 columns for various important variables

# Loading in useful packages
library(dplyr)
library(tidyr)
library(mosaic)
library(tidyjson)
library(readr)
library(lubridate)

text <- read_file("Data/alldata1.txt")
list <- unlist(strsplit(text, "\n"))   # One auction per row
samp <- sample(list, size=5000, replace=FALSE)

event <- samp %>% as.tbl_json %>%
  enter_object("em") %>%
  spread_values(timestamp = jnumber("t"),
                client = jstring("cl"),
                country = jstring("cc"),
                region = jstring("rg"),
                awsr = jstring("awsr"))
event <- event %>%
  mutate(timestamp = round(timestamp, 10)/1000) %>%   # Cut off milliseconds
  mutate(timestamp = as.POSIXct(timestamp, origin="1970-01-01", tz="GMT")) %>%
  mutate(dayofweek = wday(timestamp)) %>%
  mutate(hour = hour(timestamp)) %>%
  mutate(month = month(timestamp)) %>%
  select(-timestamp)

auction <- samp %>% as.tbl_json %>%
  enter_object("auction") %>%
  spread_values(margin = jnumber("margin"),
                rnk = jstring("rnk"),
                moid = jstring("moid"),
                tmax = jnumber("tmax")) %>%
  enter_object("user") %>%
  spread_values(bluekai = jstring("bkc"))
auction <- auction %>%
  mutate(bluekai = !is.na(bluekai))

dev <- samp %>% as.tbl_json %>%
  enter_object("auction") %>%
  enter_object("dev") %>%
  spread_values(dev = jstring("bti"))

site <- samp %>% as.tbl_json %>%
  enter_object("auction") %>%
  enter_object("site") %>%
  spread_values(zone = jstring("zoneDomain"),
                sitetype = jstring("typeid"),
                sitecat = jstring("cat")) %>%
  enter_object("publisher") %>%
  spread_values(pubID = jstring("id"))


bidreq <- samp %>% as.tbl_json %>%
  enter_object("auction") %>%
  enter_object("bidrequests") %>%
  gather_array() %>%
  spread_values(reqID = jstring("id"),
                vertical = jstring("verticalid"),
                dsp = jstring("bidderid")) %>%
  enter_object("impressions") %>%
  gather_array() %>%
  spread_values(bidfloor = jnumber("bidfloor"),
                adformat = jstring("format"),
                adproduct = jstring("product"))   # array.index is impression ID
                                                  # ignores bid reqs w/o impressions
banner <- samp %>% as.tbl_json %>%
  enter_object("auction") %>%
  enter_object("bidrequests") %>%
  gather_array() %>%
  spread_values(reqID = jstring("id")) %>%
  enter_object("impressions") %>%
  gather_array() %>%
  enter_object("banner") %>%
  spread_values(width = jnumber("w"),
                height = jnumber("h"))   # array.index is impression ID

bid <- samp %>% as.tbl_json %>%
  enter_object("auction") %>%
  enter_object("bids") %>%
  gather_array %>%
  spread_values(reqID = jstring("requestid"),
                array.index = jstring("impid"),
                winner = jstring("winner"))
bid <- bid %>%
  mutate(winner = !is.na(winner))
banner <- banner %>%
  select(-document.id)
bid <- bid %>%
  select(-document.id)
foo <- bidreq %>%
  left_join(banner, by=c("reqID", "array.index")) %>%
  left_join(bid, by=c("reqID", "array.index"))
full <- foo %>%
  left_join(event, by="document.id") %>%
  left_join(auction, by="document.id") %>%
  left_join(dev, by="document.id") %>%
  left_join(site, by="document.id") %>%
  rename(Auction=document.id, BidReq=array.index)
full <- full %>%
  select(-reqID)

full <- full[c("Auction","client", "country","region", "awsr",
               "dayofweek","hour","month","margin","rnk", "moid", "tmax",
               "bluekai","dev","zone", "sitetype","sitecat","pubID",
               "BidReq","dsp","vertical",
               "bidfloor","adformat","adproduct","width","height","winner")]

full <- full %>%
  filter(dsp != "35" & dsp != "37") %>%   # Filter out DSPs we should ignore
  filter(adformat != "1" &   # Filter out Ad Formats we should ignore
           adformat != "2" &
           adformat != "3" &
           adformat != "4" &
           adformat != "6" &
           adformat != "7" &
           adformat != "20" &
           adformat != "21" &
           adformat != "22" & 
           adformat != "25" & 
           adformat != "26")

library(dummies)
spread <- full %>%
  dummy.data.frame()
