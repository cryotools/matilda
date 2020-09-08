hbv_lite_ng <- read.table("/home/phillip/Seafile/Ana-Lena_Phillip/data/HBV-Light/HBV-light_data/Glacier_No.1/Python/Noglacier_Run/Results/Results.txt", header = 1)
hbv_lite_g <- read.table("/home/phillip/Seafile/Ana-Lena_Phillip/data/HBV-Light/HBV-light_data/Glacier_No.1/Python/Glacier_Run/Results/Results.txt", header = 1)
final_model <- read.csv("/home/phillip/Seafile/Ana-Lena_Phillip/data/scripts/Final_Model/Output/model_output_2011-01-01-2011-12-31.csv")

summary(hbv_lite_ng)
hbv_lite_ng[hbv_lite_ng == -9999] <- 0
colSums(hbv_lite_ng)

summary(hbv_lite_g)
hbv_lite_g[hbv_lite_g == -9999] <- 0
colSums(hbv_lite_g)

summary(final_model)
final_model$X <- as.POSIXct(final_model$X)
colSums(final_model[,2:8]) * 8
