library(qmap)
library(dplyr)
library(zoo)
library(lubridate)
library(ggplot2)
library(tidyr)

home <- ("/home/ana/")

data_CMIP<- read.csv(paste0(home, "Seafile/Tianshan_data/CMIP/CMIP5/EC-EARTH_r6i1p1_r7i1p1_r8i1p1/temp_prec_rcp26_rcp45_rcp85_2006-2100.csv"))
data_CMIP$time <- as.Date(data_CMIP$time)

data_CMIP$month <- month(data_CMIP$time)
data_CMIP$year <- year(data_CMIP$time)

#data_CMIP <- data_CMIP[!(data_CMIP$year=="2100"),]

data_CMIP_monthly <- data_CMIP %>%
  group_by(month, year) %>%
  summarise(temp_26=mean(temp_26), temp_45=mean(temp_45), temp_85=mean(temp_85), prec_26=sum(prec_26), prec_45=sum(prec_45), prec_85=sum(prec_85))


data_CMIP_monthly$period[data_CMIP_monthly$year >= 2006 & data_CMIP_monthly$year <= 2020] <- "2006_2020" 
data_CMIP_monthly$period[data_CMIP_monthly$year >= 2021 & data_CMIP_monthly$year <= 2040] <- "2021_2040" 
data_CMIP_monthly$period[data_CMIP_monthly$year >= 2041 & data_CMIP_monthly$year <= 2060] <- "2041_2060" 
data_CMIP_monthly$period[data_CMIP_monthly$year >= 2061 & data_CMIP_monthly$year <= 2080] <- "2061_2080" 
data_CMIP_monthly$period[data_CMIP_monthly$year >= 2081 & data_CMIP_monthly$year <= 2100] <- "2081_2100" 

trend_CMIP <- data_CMIP_monthly  %>%
  group_by(month, period)  %>%
  summarise(temp_26=mean(temp_26), temp_45=mean(temp_45), temp_85=mean(temp_85), prec_26=mean(prec_26), prec_45=mean(prec_45), prec_85=mean(prec_85))

monthly_trend_cmip <-gather(trend_CMIP, scenario, temp, "temp_26", "temp_45", "temp_85", "prec_26", "prec_45", "prec_85")
monthly_trend_cmip <- spread(monthly_trend_cmip, period, temp)

monthly_trend_cmip$diff_hist_2040 <- monthly_trend_cmip$`2021_2040` - monthly_trend_cmip$`2006_2020`
monthly_trend_cmip$diff_hist_2060 <- monthly_trend_cmip$`2041_2060` - monthly_trend_cmip$`2006_2020`
monthly_trend_cmip$diff_hist_2080 <- monthly_trend_cmip$`2061_2080` - monthly_trend_cmip$`2006_2020`
monthly_trend_cmip$diff_hist_2100 <- monthly_trend_cmip$`2081_2100` - monthly_trend_cmip$`2006_2020`
monthly_trend_cmip$prec_fact_2040 <- monthly_trend_cmip$`2021_2040` / monthly_trend_cmip$`2006_2020`
monthly_trend_cmip$prec_fact_2060 <- monthly_trend_cmip$`2041_2060` / monthly_trend_cmip$`2006_2020`
monthly_trend_cmip$prec_fact_2080 <- monthly_trend_cmip$`2061_2080` / monthly_trend_cmip$`2006_2020`
monthly_trend_cmip$prec_fact_2100 <- monthly_trend_cmip$`2081_2100` / monthly_trend_cmip$`2006_2020`

cmip_26 <- monthly_trend_cmip[monthly_trend_cmip$scenario == "temp_26" | monthly_trend_cmip$scenario == "prec_26", ]


#write.csv(monthly_trend_cmip, "/home/ana/Seafile/Tianshan_data/CMIP/CMIP5/EC-EARTH_r6i1p1_r7i1p1_r8i1p1/CMIP5_monthly_trend2.csv")
