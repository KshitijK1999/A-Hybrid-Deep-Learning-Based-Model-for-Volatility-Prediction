library(ggplot2)
library(wmtsa)
library(NlinTS)
library(tseries)
library(forecast)
library(zoo)
library(rugarch)
library(lmtest)
library(FinTS)
install.packages("fNonlinear")
library(timeSeries)
install.packages("robustHD")
library(robustHD)

metalret <- as.matrix(Final_Data_2[2:3575,14])
head(metalret)
##Autocorrelation is present in squared returns, ARCH effet is present
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

##Reject Null

##Running standard GARCH, EGARCH, TGarch
metalret_g11 <- ugarchspec(variance.model = list(model = "eGARCH",garchOrder = c(2,2)), mean.model = list(armaOrder = c(1,1)))
metalret_g11_fit <- ugarchfit(spec = metalret_g11, data = metalret)
show(metalret_g11_fit)

metalret_g11_fit_res <- residuals(metalret_g11_fit)
acf()
metalret_g11_fit_cvar <- ts(metalret_g11_fit@fit$sigma^2)
plot.ts(metalret_g11_fit_res)
plot.ts(standardize(metalret_g11_fit_cvar))
metalret_g11_fit_res <- unname(as.matrix((metalret_g11_fit_res)))
0
#Standardizing residuals

metalret_g11_fit_res <- standardize(metalret_g11_fit_res)
Box.test(metalret_g11_fit_res^2, type = "Ljung-Box", lag = 2, fitdf = 0)
ArchTest(metalret_g11_fit_res, lag = 1)
jarque.bera.test(metalret_g11_fit_res)
shapiro.test(metalret_g11_fit_res)

fore <- ugarchforecast(metalret_g11_fit, n.ahead = 365)
fore
linearMod <- lm(metalret[3:3575,1] ~ metalret[2:3574, 1]  ) 
linearMod

metalret_g11_roll <- ugarchroll(metalret_g11, data = metalret,n.ahead = 1, n.start = 1000, refit.every = 30, refit.window = "moving", solver = "hybrid", keep.coef = TRUE)
show(metalret_g11_roll)
0
report(metalret_g11_roll, type = "VaR", VaR.alpha = 0.01, conf.level =0.99)
m <- metalret_g11_roll@forecast$
head(m)
plot(as.ts(m), type = 'b')
lines(metalret_g11_roll@forecast$VaR[,2] , col = "blue")
metalret_g11_forecast <- ugarchforecast(metalret_g11_fit, n.ahead = 365)
plot((metalret_g11_forecast))
1
