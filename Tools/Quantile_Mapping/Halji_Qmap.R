#install.packages('raster')
library(qmap)
library(raster)
setwd("/home/anz/Seafile/work/io/Halji/QMAP/")
#T2 <- read.csv('20200302_temperatures_for_qmap.csv')
#era_T2 <- read.csv('20200302_Halji_ERA5_land_T2_2000_2019_closest_gp.csv')
#output_file <- ''

T2 <- read.csv('20200701_temperatures_for_qmap.csv')
era_T2 <- read.csv('20200701_Halji_ERA5_land_T2_2000_2019_closest_gp.csv',header=FALSE)
output_file <- '20200701_mapped_T2.csv'


qmap_T2 <- qmap::fitQmap(T2$measured_temperatures, T2$simulated_temperatures)
test <- qmap::doQmap(era_T2$V2, qmap_T2)
write.csv(qmap::doQmap(era_T2$V2, qmap_T2), output_file, row.names = FALSE)

# G <- read.csv('G_for_qmap.csv')
# PRES <- read.csv('PRES_for_qmap.csv')
# RH2 <- read.csv('RH2_for_qmap.csv')
# TP <- read.csv('TP_for_qmap.csv')
# U2 <- read.csv('U2_for_qmap.csv')
# 
# era_G <- read.csv('20200210_Halji_ERA5_land_G_2000_2019_closest_gp.csv')
# era_PRES <- read.csv('20200210_Halji_ERA5_land_PRES_2000_2019_closest_gp.csv')
# era_RH2 <- read.csv('20200210_Halji_ERA5_land_RH2_2000_2019_closest_gp.csv')
# era_TP <- read.csv('20200210_Halji_ERA5_land_RRR_2000_2019_closest_gp.csv')
# era_U2 <- read.csv('20200210_Halji_ERA5_land_U2_2000_2019_closest_gp.csv')
# 
# qmap_G <- qmap::fitQmap(G$measured_G, G$simulated_G)
# qmap_PRES <- qmap::fitQmap(PRES$measured_pressure, PRES$simulated_pressure)
# qmap_RH2 <- qmap::fitQmap(RH2$measured_rh2, RH2$simulated_rh2)
# qmap_TP <- qmap::fitQmap(TP$measured_TP, TP$simulated_TP)
# qmap_U2 <- qmap::fitQmap(U2$measured_U2, U2$simulated_U2)
# 
# write.csv(qmap::doQmap(era_G$X0, qmap_G), "mapped_G.csv", row.names = FALSE)
# write.csv(qmap::doQmap(era_PRES$X0, qmap_PRES), "mapped_PRES.csv", row.names = FALSE)
# write.csv(qmap::doQmap(era_RH2$X0, qmap_RH2), "mapped_RH2.csv", row.names = FALSE)
# write.csv(qmap::doQmap(era_TP$X0, qmap_TP), "mapped_TP.csv", row.names = FALSE)
# write.csv(qmap::doQmap(era_U2$X0, qmap_U2), "mapped_U2.csv", row.names = FALSE)
