library(ggplot2)
library(wmtsa)
library(NlinTS)
library(tseries)
library(forecast)
library(zoo)
library(rugarch)
library(lmtest)
library(FinTS)
library(timeSeries)
library(robustHD)
library(readxl)
library(multDM)

Final_Data_2 <- read_excel("Downloads/covid/MetalVol/Final_Data_2.xlsx")
View(Final_Data_2)

metalret <- as.matrix(Final_Data_2[2:3575,14])
head(metalret)
##Autocorrelation is present in squared returns, ARCH effect is present
metalret_arma <- arma(metalret, order = c(1,0))
metalret_res <- metalret_arma$residuals
m <- lm(metalret_res ~ 1)
bgtest(m)
dwtest(m)
##Cannot reject null of bg test, hence no serial autocorrelation, Model
ndiffs(metalret_res)
ArchTest(as.matrix(metalret_res)^2)
Box.test(metalret_res, type = "Box-Pierce", lag = 2, fitdf = 0)
##Reject Null


##Running standard GARCH, EGARCH, TGarch
for (i in (1:2)){
  for (j in (1:2)){
    if ( i == 2 && j == 1 )
    { next}
    metalret_g11 <- ugarchspec(variance.model = list(model = "eGARCH",garchOrder = c(i,j)), 
                               mean.model = list(armaOrder = c(1,1)))
    metalret_g11_fit <- ugarchfit(spec = metalret_g11, data = metalret)
    write.table("Alluminium",file =( "/home/kshitish/Downloads/covid/MetalVol/metTGARCH.txt"), append = TRUE, row.names = FALSE,col.names = FALSE)
    write.table(capture.output(metalret_g11_fit), file =( "/home/kshitish/Downloads/covid/MetalVol/metEGARCH.txt"), append = TRUE, row.names = FALSE,col.names = FALSE)
  }
}

##Rolling window fitting of GARCH models
metalret <- as.data.frame(Final_Data_2[2:3575,21])
head(metalret)
length(metalret)
metalret <- 1000*metalret
metalret[3574,]

l <- c()
metalret_g11 <- ugarchspec(variance.model = list(model = "sGARCH",garchOrder = c(1,1)),
                           mean.model = list(armaOrder = c(1,0)))
for (i in (1:3322)){
  dat <- metalret[i:(i+251),]
  metalret_g11_fit <- ugarchfit(spec = metalret_g11, data = dat, solver = "hybrid")  
  metalret_g11_for <- ugarchforecast(metalret_g11_fit, n.ahead = 1, data = dat)
  l[[i]] <- (metalret_g11_for@forecast$sigmaFor)^2
}

head(l)
length(l)
write.csv(l, file = "/home/kshitish/Downloads/covid/MetalVol/sGarch1110Crude_252roll.csv")

k <- c()
metalret_g11 <- ugarchspec(variance.model = list(model = "eGARCH",garchOrder = c(1,1)),
                           mean.model = list(armaOrder = c(1,0)))
for (i in (1:3322)){
  dat <- metalret[i:(i+251),]
  metalret_g11_fit <- ugarchfit(spec = metalret_g11, data = dat, solver = "hybrid")  
  metalret_g11_for <- ugarchforecast(metalret_g11_fit, n.ahead = 1, data = dat)
  k[[i]] <- (metalret_g11_for@forecast$sigmaFor)^2
}
k = k/10000
write.csv(k, file = "/home/kshitish/Downloads/covid/MetalVol/eGarch1110Gold_252roll.csv")

m <- c()
metalret_g11 <- ugarchspec(variance.model = list(model = "fGARCH",garchOrder = c(1,1), submodel = 'TGARCH'),
                           mean.model = list(armaOrder = c(1,0)))
for (i in (1:3322)){
  dat <- metalret[i:(i+251),]
  metalret_g11_fit <- ugarchfit(spec = metalret_g11, data = dat, solver = "hybrid")  
  metalret_g11_for <- ugarchforecast(metalret_g11_fit, n.ahead = 1, data = dat)
  m[[i]] <- (metalret_g11_for@forecast$sigmaFor)^2
}

length(m)
m = m/1000000
head(m)
