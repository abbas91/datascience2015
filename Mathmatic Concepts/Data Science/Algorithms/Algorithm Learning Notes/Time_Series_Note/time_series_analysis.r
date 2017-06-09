>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>                                                          >
>>>>>>>>>>>>>>>>>>> Time Series Analysis >>>>>>>>>>>>>>>>>>>
>                                                          >
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

" A time series is a series of data points indexed (or listed or graphed) in time order. Most commonly, "
" a time series is a sequence taken at successive equally spaced points in time. Thus it is a sequence of "
" discrete-time data. Examples of time series are heights of ocean tides, counts of sunspots, and the daily "
" closing value of the Dow Jones Industrial Average. "


[1] >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " Characteristics of Time Series "

 
[1] >>>>>>>>>>>>>>> "Sample Time series Data / Plotting"
#1 
install.packages("astsa")
library(astsa)
plot(jj, type="o", ylab="Quarterly Earning per Share")

#2
plot(globtemp, type="o", ylab="Global Temperature Deviations")

#3 - Combine two plot into 1 plot
par(mfrow = c(2,1))
plot(jj, type="o", ylab="Quarterly Earning per Share")
plot(globtemp, type="o", ylab="Global Temperature Deviations")

# Plot multiple series in different colors
ts.plot(fmri1[,2:5], col=1:4, ylab="BOLD", main="Cortex")





[2] >>>>>>>>>>>>>>> "Time Series Statistical Models " > 'total random trend'

>1 " White Noise "
" w(t) - Uncorrelated random variable, with mean 0, variance a^2 "
" w(t) ~ iid N(0, a^2) : Gausian White Noise "
w = rnorm(500,0,1) # mean=0, sd=1




>2 " Moving Averages and Filtering " > 'smooth trend'
" v(t) - moving average of a range of value to soomth the series "
" v(t) = i/3 * (w(t-1) + w(t) + w(t+1)) "
w = rnorm(500,0,1) # mean=0, sd=1
v = filter(w, sides=2, filter=rep(1/3,3)) # moving average
par(mfrow=c(2,1))
plot.ts(w, main="white noise")
plot.ts(v, ylim=c(-3,3), main="moving average")





>3 " Autoregressions " > 'smooth trend'
" x(t) - a regression prediction of current value as a function of the past n values "
" x(t) = x(t-1) - .9x(t-2) + w(t) "
w = rnorm(500,0,1) # mean=0, sd=1
x = filter(w, filter=c(1,-.9), method="recursive")[-(1:50)] # remove first 50
plot.ts(x, main="Autoregressions")





>4 " Random Walk with Drift " > 'Global trend'
" X(t) - A random walk model with a drift, if drift = 0, it is simply a random walk "
" X(t) = d + X(t-1) + w(t) " # d = drift 
set.seed(154)
w = rnorm(200); x = cumsum(w)
wd = w + .2; xd = cumsum(wd)
plot.ts(xd, ylim=c(-5,55), main="random walk", ylab='')
lines(x, col=4); abline(h=0, col=4, lty=2); abline(a=0, b=.2, lty=2)





>5 " Signal in Noise " > 'Periodic trend'
" real signal + noise => x(t) = s(t) + v(t) | s:signal, v:noise correlated with t"
" p(t) - using cosin wave model to micmic real signal"
" p(t) = 2cos(2pie * (t+15) / 50) + w(t) " # Add noise
cs = 2*cos(2*pi*1:500/50 + .6*pi); w = rnorm(500,0,1)
par(mfrow=c(3,1), mar=c(3,2,2,1), cex.main=1.5)
plot.ts(cs, main=expression(2*cos(2*pi*1:500/50 + .6*pi)))
plot.ts(cs + w, main=expression(2*cos(2*pi*1:500/50 + .6*pi) + N(0,1)))
plot.ts(cs + 5*w, main=expression(2*cos(2*pi*1:500/50 + .6*pi) + N(0,25)))







[3] >>>>>>>>>>>>>>> " Measure of Dependence "

>1 " Mean Function "
" u - the usual expected value "
-- "Moving Average Series" : E(v(t)) = 0 # White noise average = 0
-- "Random Walk with Drift" : E(x(t)) = d # drift will be expected
-- "Signal Plus Nosie" : E(p(t)) = 2cos(2pie*(t+15)/50) # Just the cosin wave





>2 " Autocovariance Function "
-- "White Noise" : cov(w(s), w(t)) = {s=t then SD, s<>t then 0}
-- "Moving Average Series (t-1 + t + t+1)" : cov(v(s),v(t)) = {s=t then 3/9 * sd, 
	                                                          |s-t|=1 then 2/9 * sd, 
	                                                          |s-t|=2 then 1/9 * sd, 
	                                                          |s-t| > 2 then 0}
-- "Random Walk" : cov(x(s), x(t)) = min{s,t} sd





>3 " Autocorrelation Function (ACF)"
-- "s,t correlated" : P(s,t) 




>4 " Cross-Covariance Function " >> "Between two series x(t), y(t) "
-- " [TBD] "




>5 " Cross-Correlation Function " >> "Between two series x(t), y(t) "
-- " [TBD] "







[3] >>>>>>>>>>>>>>> " Stationary Time Seroes "

" A regularity through out of time series called 'Stationarity' "


>1 " Strictly Stationary "
" Probablistic behaviors of {x(t1), x(t2), ... , x(tk)} is the same as "
"                           {x(t1+h), x(t2+h), ... , x(tk+h)} for all h = 0, +-1, +-2, ..."




>2 " Weakly Stationary " >> "Used Mostly"
" u mean function is a constant, not dependes on t "
" autocovariance function cov(x(t), x(s)) only dependes on |s-t| "




** " autocovariance function of a stationary time series "
" P-21 "

** " Autocorrelation function of a stationary time series "
" P-21 "

-- "White noise" : "Stationary, strictly stationary if var ~ N"
-- "Moving Average" : "Stationary"
-- "Random Walk" : " Not a Stationary"


>3 " Trend Stationarity "
" p-22 "


>4 " Joint Stationary "
" P-23 "
" Consider two series, x(t) and y(t), formed from the sum and difference of two successive values of a white noise process: "
" x(t) = w(t) + w(t-1) and y(t) = w(t) - w(t-1) "
" Clearly autocovariance and cross-covariance only depends on the lag separation -- joint stationary "


>5 " Cross Correlation "
" P-24 "
x = rnorm(100)
y = lag(x, -5) + rnorm(100)
ccf(y, x, ylab='CCovF', type='covariance')


>6 " Linear Process "
" x(t) = u + w(t) "




[4] >>>>>>>>>>>>>>> " Estimation of correlation "

>1 "Sample Autocovariance function"
" p-27 "


>2 "Sample Autocorrelation function (Sample ACF)"
" p-28 "
# SOI series - compare sample ACF for different sample size
(r = round(acf(soi, 6, plot=FALSE)$acf[-1],3)) # First 6 sample acf values
" [1] 0.604 0.374 0.214 0.050 -0.107 -0.187 "
par(mfrow=c(1,2))
plot(lag(soi,-1),soi); legend('topleft', legend=r[1])
plot(lag(soi,-6),soi); legend('topleft', legend=r[6])

# A simulated time series
set.seed(123123)
x1 = 2*rbinom(11,1,.5) - 1 # Simulate sequence of coin tosses n = 10
x2 = 2*rbinom(101,1,.5) - 1 # Simulate sequence of coin tosses n = 100

y1 = 5 + filter(x1, sides=1, filter=c(1,-.7))[-1] # y(t) = 5 + x(t) - .7x(t-1)
y2 = 5 + filter(x2, sides=1, filter=c(1,-.7))[-1] # y(t) = 5 + x(t) - .7x(t-1)

plot.ts(y1, type='s'); plot.ts(y2, type='s') # plot both series
c(mean(y1),mean(y2)) # sample mean

acf(y1, lag.max=4, plot=FALSE) 
" 0   1      2      3       4     "
" 1   -0.6   0.42   -0.30   -0.007"
acf(y1, lag.max=4, plot=FALSE) 
" 0   1      2       3        4     "
" 1   -0.48  -0.002  -0.004   0.000 "




>3 "Sample Cross-Covariance Function"
" p-30 "



>4 "Sample Cross-Correlation Function (CCF) "
" p-30 "

# Compare two series ACF - period trend for each, CCF, which lag associated with each other
par(mfrow=c(3,1))
acf(soi, 48, main="SOI")
acf(rec, 48, main="rec")
ccf(soi, rec, 48, main="soi vs rec", ylab="CCF")







[5] >>>>>>>>>>>>>>> " Vector-Valued and Multidimensional Series "

" A vector of time series -- contains p univariate time series "
" x(t) = (x(t1), x(t2), ... , x(tp)) "

>1 "Plotting - A vector of time series"
persp(1:64, 1:36, soiltemp, phi=25, theta=25, scale=FALSE, expand=4, ticktype="detailed", xlab="rows", ylab="cols", zlab="temperature")
# Plot means of each row
plot.ts(rowMeans(soiltemp), xlab="row", ylab="AVG_temp")


>2 "Sample ACF - A vector of time series"
" p-36 "
fs = Mod(fft(soiltemp-mean(soiltemp)))^2/(64*36)
cs = Re(fft(fs, inverse=TRUE)/sqrt(64*36)) # ACovF
rs = cs/cs[1,1]
rs2 = cbind(rs[1:41,21:2], rs[1:41,1:21])
rs3 = rbind(rs2[41:2,],rs2)
par(mar=c(1,2.5,0,0)+.1)
persp(-40:40, -20:20, rs3, phi=30, theta=30, expand=30, scale="FALSE", ticktype="detailed", xlab="row lags", ylab="column lags", zlab="ACF")

















































[2] >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " Time Series Regression and Exploratory Data Analysis "


-1- " Classical Regression in the Time Series Context "

x(t) = beta0 + beta1*z1(t) + beta2*z2(t) + .. .. .. + betaq*zq(t) + w(t)
" OLS - Regression Attributes ..."





-2- " Metrics to choose Models "
" MSE - mean squared error "
" F = MSR/MSE "
" R^2 "
" AIC - p49 small better (tends to be superior in smaller samples where relative number of parameter is large) "
" AICc - p49 "
" BIC - p50 small better (Tends to be superior in large samples, choose smaller model - higher penality "

# ----------- Examples 01 - fitting model and select models
"M1" M(t) = beta0 + beta1*t + w(t)
"M2" M(t) = beta0 + beta1*t + beta2*(T(t)-T(mean)) + w(t)
"M3" M(t) = beta0 + beta1*t + beta2*(T(t)-T(mean)) + beta3*(T(t)-T(mean))^2 + w(t)
"M4" M(t) = beta0 + beta1*t + beta2*(T(t)-T(mean)) + beta3*(T(t)-T(mean))^2 + beta4*P(t) + w(t)

# Plot data
par(mfrow=c(3,1))
plot(cmort, main="Cardiovascular Morality", xlab="", ylab="")
plot(tempr, main="Temperature", xlab="", ylab="")
plot(part, main="Particulates", xlab="", ylab="")
dev.new()
ts.plot(cmort, tempr, part, col=1:3) # all in same plot

dev.new()
pairs(cbind(Mortality=cmort, Temperature=tempr, Particulates=part))
temp = tempr-mean(tempr) # center temperature
temp2 = temp^2
trend = time(cmort) # time

fit = lm(cmort~ trend + temp + temp2 + part, na.action=NULL) # M4

summary(fit) # Regression result
summary(aov(fit)) # ANOVA table 

summary(aov(lm(cmort~cbind(trend, temp, temp2, part))))

num = length(cmort) # sample size
AIC(fit)/num - log(2*pi) # AIC
BIC(fit)/num - log(2*pi) # BIC
(AICc = log(sum(resid(fit)^2)/num) + (num + 5)/(num-5-2)) # AICc

# -------------------------------------------------------------------


# ------------ Example 02 - Regression with Lagged Variabe
" Usually find out from ACF - ex. S is -6 lag associated with R "
R(t) = beta0 + beta1*S(t-6) + w(t)

# Example V1
fish = ts.intersect(rec, soiL6=lag(soi, -6), dframe=TRUE)
summary(fit1 <- lm(rec~soiL6, data=fish, na.action=NULL))

# Example V2
library(dynlm)
summary(fit2 <- dynlm(rec~ L(soi,6)))









-2- " Exploratory Data Analysis "

" It is necessary for time-series data to be stationary >>> "

>1 " Assume -- Trend Stationary (Detrending) "
x(t) = u(t) + y(t) # y(t) is a stationary process

fit = lm(chicken~time(checken), na.action=NULL) # Regress checken on time
plot(resid(fit), type="o", main="detrend")
acf(resid(fit), 48, main="detrend")


>2 " Assume -- Random Walk With Drift (Differencing) "
u(t) = d + u(t-1) + w(t) # First differencing

plot(diff(checken), type="o", main="First Differencing")
acf(diff(checken), 48, main="Frist Differencing")
mean(diff(checken)) # drift estimate

" BackShift Operator " k>0
B^k * x(t) = x(t-k)
" ForwardShift Operator " k>0
B^-k * x(t) = x(t-k)



>3 " ... -- ... (Fractional Differencing)"
" A less sever differencing operation "

" Nonstationary + nonlinear >>> "

>4 " (transformation) -- Log transformation "
" Equalize variability over the length of single series "

y(t) = log(x(t))

log(x)


>5 " (transformation) -- Box-cox transformation "

boxCox(x, lambda = seq(-2, 2, 1/10), plotit = TRUE)





" Scatter plot - find nonlinear relationship x(t) - x(t-6) or x(t) - y(t-3) "

>6 " Scatter lag plot "
library(astsa)

lag1.plot(soi, 12) "x(t) - x(t-12)"
lag2.plot(soi, rec, 8) "x(t) - y(t-12)"

# If find a lag var relationship is non-linear
rec(t) = beta0 + beta1*soi(t-6) + w(t)  ----------- rec(t) = 
# Original
fish = ts.intersect(rec, soiL6=lag(soi, -6), dframe=TRUE)
summary(fit1 <- lm(rec~soiL6, data=fish, na.action=NULL))

# With dummy var to address nonlinear
dummy = ifelse(soi<0, 0, 1)
fish = ts.intersect(rec, soiL6=lag(soi, -6), dL6=lag(dummy, -6), dframe=TRUE)
summary(fit1 <- lm(rec~soiL6*dL6, data=fish, na.action=NULL))




>7 " Regression to discover a Signal in Noise "
" Generate 500 observation from a cosin wave model "
set.seed(90210)
x = 2*cos(2*pi*1:500/50 + .6*pi) + rnorm(500,0.5)
z1 = cos(2*pi*1:500/50)
z2 = sin(2*pi*1:500/50)
summary(fit <- lm(x~0+z1+z2)) # zero to exclude intercept

par(mfrow=c(2,1))
plot.ts(x)
plot.ts(x, col=8, ylab=expression(hat(x)))
lines(fitted(fit), col=2) # fit regression trend to see
" p - 64 "





" Smoothing in Time Series "

>8 " Moving Average Soomther - useful in discovering certain traits in a time series, long term trebd, seseanal components "
" Moving with weights to the positions "
wgts = c(.5, rep(1,11), .5) / 12
soif = filter(soi, sides=2, filter=wgts)
plot(soi)
lines(soif, lwd=2, col=4)
# insert weight
par(fig=c(.65, 1, .65, 1), new = TRUE)
nwgts = c(rep(0,20), wgts, rep(0,20))
plot(nwgts, type="1", ylim=c(-.02,.1), xaxt='n', yaxt='n', ann=FALSE)



>9 " Kernel Smoothing - use a weighted function (normal kernel) "
" The wider the bandwith - b --> the smoother the result > ksmooth "
plot(soi)
lines(ksmooth(time(soi), soi, "normal", bandwidth=1), lwd=2, col=4)
# insert weight function
par(fig=c(.65, 1, .65, 1), new = TRUE)
gauss = function(x) { 1/sqrt(2*pi) * exp(-(x^2)/2) }
x = seq(from = -3, to = 3, by = 0.001)
plot(x, gauss(x), type="1", ylim=c(-.02,.45), xaxt='n', yaxt='n', ann=FALSE)



>10 " Lowess - Nearest Neighbor Regression "
" The bigger % data used to train, the smoother "
plot(soi)
lines(lowess(soi, f=.05), lwd=2, col=4) # 0.05 El Nino Cycle
lines(lowess(soi, f=2/3), lty=2, lwd=2, col=2) # 2/3 trend


>11 " Smoothing Splines - fit a polynomial regression in terms of time "
m(t) = beta0 + beta1*t + beta2*t^2 + beat3*t^3
" lambda, the larger the smoother "
plot(soi)
lines(smooth.spline(time(soi), soi, spar=.5), lwd=2, col=4)
lines(smooth.spline(time(soi), soi, spar=1), lty=2, lwd=2, col=2)


** " Smoothing can also being used to determine nonlinear relationship between series "
plot(tempr, cmort, xlab="Temp", ylab="Moral")
lines(lowess(temr, cmort))














































[2] >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " ARIMA Models -- Time domain "

" Classical regression is insufficient for explaining all of the interesting dynamics of a time series. For example, additive assumption - residual, x only influenced by current state of value "

" Auto-regressive (AR) " --> " Auto-Regressive moving Average (ARMA) " --> " Auto-Regressive integrated moving average (ARIMA) "

** " General Linear Process -- Basic concept of parameteric time series models "
Y(t) = w(t) + beta1*w(t-1) + beta2*w(t-2) + .. .. 
" w(t) -- white noise "
" beta -- weights "
" Invertibility : time series process can be re-expressed as a general linear process above "




>>>> " Stationary Process --> a constant mean over time "

-1- " Auto-Regression Process (AR) "

" Linear Process "
Y(t) = gamma1*Y(t-1) + gamma2*Y(t-2) + .. .. + gammap*Y(t-p) + w(t) " AR(p), p-order "

** "|gamma| < 1, auto-correlation decreased exponentially with lag "
** " 0< gamma <1, Y(t) positive correlated to Y(t-p) | -1< gamma <0, Y(t) negative correlated to Y(t-p)"
** "Plot Y(t) by t, most above 0 == Y(t) postive correlated to Y(t-p) |  most around 0 == Y(t) negative correlated to Y(t-p)"

# Simulation
par(mfrow=c(2,1))
plot(arima.sim(list(order=c(1,0,0), ar=.9), n=100), ylab="x", main="gamma = 0.9") # 1 order
plot(arima.sim(list(order=c(1,0,0), ar=-.9), n=100), ylab="x", main="gamma = -0.9") # 1 order
# Model
set.seed(987686)
x = rnorm(150, mean=5) 
arima(x, order=c(1,0,0)) # 1 order

# Model - forecast
reger = ar.ols(rec, order=2, demean=FALSE, intercept=TRUE)
fore = predict(reger, n.ahead=24)

ts.plot(rec, fore$pred, col=1:2, xlim=c(1980,1990), ylab="Recruitment")
U = fore$pred+fore$se; L = fore$pred-fore$se
xx = c(time(U), rev(time(U))); yy = c(L, rev(U))
polygon(xx, yy, border=8, col=gray(.6, alpha=.2))
lines(fore$pred, type="p", col=2)






-2- " Moving Average Process (MA) "

" Non-linear process "
Y(t) = w(t) - theta1*w(t-1) - theta2*w(t-2) - .. .. - thetaq*w(t-q) " MA(q), q-order"

** "Plot Y(t) by t, most above 0 == Y(t) postive correlated to Y(t-q) |  most around 0 == Y(t) negative correlated to Y(t-q)"

# Simulation
par(mfrow=c(2,1))
plot(arima.sim(list(order=c(0,0,1), ma=.9), n=100), ylab="x", main="theta = 0.9") # 1 order
plot(arima.sim(list(order=c(0,0,1), ma=-.9), n=100), ylab="x", main="theta = -0.9") # 1 order
# Model
set.seed(987686)
x = rnorm(150, mean=5) 
arima(x, order=c(0,0,1)) # 1 order






-3- " Auto-Regression Moving Average Model (ARMA) "

" Non-linear Process "
Y(t) = gamma1*Y(t-1) + gamma2*Y(t-2) + .. .. + gammap*Y(t-p) + w(t) - theta1*w(t-1) - theta2*w(t-2) - .. .. - thetaq*w(t-q) " ARMA(p,q), p,q-order "

# Model
set.seed(987686)
x = rnorm(150, mean=5) 
arima(x, order=c(1,0,1)) # 1 order

# Model - Backcasting
set.seed(90210)
x = arima.sim(list(order = c(1,0,1), ar = .9, ma = .5), n = 100)
xr = rev(x) # Reversed data
pxr = predict(arima(xr, order=c(1,0,1)), 10) # predict the reversed data
pxrp = rev(pxr$pred) # Reorder the predictors
pxrse = rev(pxr$se) # Reorder the SEs
nx = ts(c(pxrp, x), start = -9) # Attach the back cast data

plot(nx, ylab=..., main="Backcasting")
U = nx[1:10] + pxrse; L = nx[1:10] - pxrse
xx = c(-9:0, 0:-9); yy = c(L, rev(U))
ploygon(xx, yy, border = 8, col = gray(0.6, alpha = 0.2))
lines(-9:0, nx[1:10], col = 2, type="o")



** " Causal "

** " Invertable "








** "ACF ---- PACF "
" Help to identify the order numbers -- don't directly fit a large orders which tends to 'Over Fitting' the data "

" ACF - effect between x(t) and x(t+h) " > " for MA "
" PACF - effect between x(t) and x(t+h) removing effect from x(t+1), ... , x(t+h-1) " > " for AR "
ACF = ARMAacf(ar=c(1.5, -.75), ma=0, 24)[-1]
PACF = ARMAacf(ar=c(1.5, -.75), ma=0, 24, pacf=TRUE)
par(mfrow=c(1,2))
plot(ACF, type="h", xlab="lag", ylim=c(-.8,1)); abline(h=0)
plot(PACF, type="h", xlab="lag", ylim=c(-.8,1)); abline(h=0)
# Example 2
library(astsa)
acf2(rec, 48) # acf, pacf







** " Estimation "
" All methods leads to optimal for AR, MA, ARMA for large sample "

" OLS " 
> # for linear process
ar.ols
reger = ar.ols(rec, order=2, demean=FALSE, intercept=TRUE)
fore = predict(reger, n.ahead=24)

" Yule-Walker "
> # for linear process
ar.yw
reger = ar.yw(rec, order=2)
fore = predict(reger, n.ahead=24)

" Durbin-Levinson "

" MLE "
ar.mle
rec.mle = ar.mle(rec, order=2)

" Gauss-Newton "
> # for non-linear process







** " Bootstrapping - Small n -> poor approximation "
# Data Series
set.seed(101010)
e = rexp(150, rate=.5); u = runif(150, -1, 1); de = e*sign(u)     # | True form 
dex = 50 + arima.sim(n=100, list(ar=.95), innov=de, n.start=50)   # | True form
# Plot data series
plot.ts(dex, type='o', ylab="Data")

# Simple AR fit and estimates
fit = ar.yw(dex, order=1)
round(cbind(fit$x.mean, fit$ar, fit$var.pred), 2) # Can be off


# True estimate distribution
" Know all true details of the model (impossible) "
set.seed(111)
phi.yw = rep(NA, 1000)
for (i in 1:1000) {
	e = rexp(150, rate=.5); u = runif(150, -1, 1); de = e*sign(u) # | True form 
    x = 50 + arima.sim(n=100, list(ar=.95), innov=de, n.start=50) # | True form 
    phi.yw[i] = ar.yw(x, order=1)$ar
}
# -- Plotting
plot(density(phi.yw, bw=.02), lwd=2)


# Bootstrap 500  estimate distrubtion
" Only know data 'dex' (possible) "
set.seed(666)
fit = ar.yw(dex, order=1)
m = fit$x.mean
phi = fit$ar
nboot = 500
resids = fit$resid[-1]
x.star = dex
phi.star.yw = rep(NA, nboot)
# Bootstrapping
for (i in 1:nboot) {
	resid.star = sample(resids, replace=TRUE)
	for (t in 1:99) {
		x.star[t+1] = m + phi*(x.star[t]-m) + resid.star[t]
	}
	phi.star.yw[i] = ar.yw(x.star, order=1)$ar
}
# -- plotting
plot(density(phi.star.yw, bw=.02), lwd=2)

" Bootstrap estimate distribution ~ True estimate distribution "





>>>> " Non-Stationary Process --> any time series without a constant mean over time "

-4- " Integrated Auto-Regression Moving Average Model (ARIMA) - including differencing "
** " Steps to fitting a ARIMA model "
" [1] - Plotting the data "
" [2] - Possibly transforming the data"
" [3] - identify the dependence orders of the model "
" [4] - parameter estimation "
" [5] - diagnostics, and "
" [6] - Model choice "

Y(t) = (1+gamma1)*Y(t-1) + (gamma2-gamma1)*Y(t-2) + (gamma3-gamma2)*Y(t-3) + .. .. + (gammap-gammap-1)*Y(t-p) + w(t) - theta1*w(t-1) - theta2*w(t-2) - .. .. - thetaq*w(t-q) " ARIMA(p,1,q), p,1,q-order "

# Model
set.seed(987686)
x = rnorm(150, mean=5) 
arima(x, order=c(1,1,1)) # 1 order 

" +++++ Example Modeling ARIMA Model "
# ----------------------------------------------------------------------------------------------- #
" 1. Plotting time series / acf - pacf / "
plot(gnp)
acf2(gnp, 50)

" 2. Possibly transforming the data "
gnpgr = diff(log(gnp)) # Growth rate

" 3. identify the dependence orders of the model "
plot(gnpgr)
acf2(gnpgr, 24) # ---- implies AR(1), MA(2)

" 4. parameter estimation "
sarima(gnpgr, 1, 0, 0) # AR(1)
sarima(gnpgr, 0, 0, 2) # MA(2)

" 5. diagnostics - if residual show like whitenoise - well fit"
sarima # By product of run

" 6. Model choice - AIC, AICc, BIC "
fit1 = sarima(gnpgr, 1, 0, 0) # AR(1)
fit2 = sarima(gnpgr, 0, 0, 2) # MA(2)

fit1$AIC; fit1$AICc; fit1$BIC # Lower the better
fit2$AIC; fit2$AICc; fit2$BIC


" ---------------- Special cases "
**# Evaluated in step 5, not good -> no constant in model
sarima(log(varve), 0, 1, 1, no.constant=TRUE)

" ---------------- Special cases (Regression with auto-correlated error) "
**
" First, run OLS regression on data to get residual "
" Then, identify ARMA model for residual "
" Run weighted OLS on regression with new error term "
" Inspect residual again, see if is whitenoise now"
# Example 1
" M(t) = beta1 + beta2*t + beta3*T(t) + beta4*T(t)^2 + beta5*P(t) + x(t) "
" x(t) not whitenoise "
trend = time(cmort); temp = tempr - mean(tempr); temp2 = temp^2
summary(fit <- lm(cmort ~ trend + temp + temp2 + part, na.action=NULL))
acf2(resid(fit), 52) # ----- implies AR(2) model
sarima(cmort, 2, 0, 0, xreg=cbind(trend,temp,temp2,part)) # correlated error model, error term fits a AR(2) model

# Example 2
" R(t) = beta0 + beta1*S(t-6) + beta2*D(t-6) + beta3*(D(t-6)*S(t-6)) + w(t) "
" w(t) not whitenoise "
dummy = ifelse(soi<0, 0, 1)
fish = ts.intersect(rec, soiL6=lag(soi, -6), dL6=lag(dummy, -6), dframe=TRUE)
summary(fit <- lm(rec ~ soiL6*dL6, data=fish, na.action=NULL))
attach(fish)
plot(resid(fit))
acf2(resid(fit)) # ----- implies AR(2) model
intract = soiL6*dL6 # interaction term
sarima(rec, 2, 0, 0, xreg=cbind(soiL6, dL6, intract)) # correlated error model, error term fits a AR(2) model



" ---------------- Special cases (Multiplicative Seasonal ARIMA Model) "
**
" Like s = 12, s = 6, s = 24, etc "
" Seasonal and nonstationary behavior "
# We simulate a 3 years data seasonal at 12 AR(1)
set.seed(666)
phi = c(rep(0, 11), .9) # AR(!) with coefficient = .9
sAR = arima.sim(list(order=c(12,0,0), ar=phi), n=37) # s = 12
sAr = ts(sAR, freq=12)
# ACF - PACF plots on the theoriatical data
ACF = ARMAacf(ar=phi, ma=0, 100)
PACF = ARMAacf(ar=phi, ma=0, 100, pacf=TRUE)
plot(ACF, type="h", xlab="LAG", ylim=c(-.1,1)); abline(h=0)
plot(PACF, type="h", xlab="LAG", ylim=c(-.1,1)); abline(h=0)

" ARIMA(0,1,1) X (0,1,1) 12 " # 
# Example 'Air passengers data'
" ---- stablized the time series "
x = AirPassengers
lx = log(x); # -- log
dlx = diff(lx); # -- log, then diff
ddlx = diff(dlx, 12) # -- log, then diff, then 12 order diff
# plot
par(mfrow=c(2,1))
monthplot(dlx); monthplot(ddlx)
acf2(ddlx, 50)
" ----- Seasonal component "
# ACF cutting off a lag s=12, PACF tailing off at lags 1,2,3,4... -> SMA(1)
" ----- Non-seasonal component "
# Both ACF and PACF tailing off - ARMA(1,1)

sarima(lx, 1,1,1, 0,1,1, 12)
" Lowesr AIC, BIC, AICc used to evaluate "

# make prediction for 12 months
sarima.for(lx, 12, 0,1,1, 0,1,1, 12)


# ----------------------------------------------------------------------------------------------- #


-5- " Integrated Moving Average Model (IMA) "
" Exponentially Weighted Moving Average (EWMA) "
# Model
set.seed(987686)
x = rnorm(150, mean=5) 
arima(x, order=c(0,1,1)) # 1 order 
# Simulation'
set.seed(123)
x = arima.sim(list(order = c(0,1,1), ma=-0.8), n = 100) # ma = lambda = -beta
" 1 - lambda = smoothing para -> larger = smoother forecast "
# Estimate from data
x.ima = HoltWinters(x, beta=FALSE, gamma=FALSE)
" alpha: 0.1663072 "
  

-6- " Integrated Auto-Regression Model (ARI) "
# Model
set.seed(987686)
x = rnorm(150, mean=5) 
arima(x, order=c(1,1,0)) # 1 order 


























[3] >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " Spectral Analysis and Filtering -- Frequency domain "

" [Spectral Analysis] -- in which we focus on frequency domain approach. We argue about the concept of regularity of a series can best be "
"                      expressed in terms of periodic variations of the underlaying phenomenon that produced the series. "

" [Investigation and exploitation of the properties of the time-invariant] -- linear transformation (similiar to the linear regression used in "
"                                                                             traditional statistics. "

" [Coherency] -- a tool for relating the common periodic behaviors of two series. It is a frequency based measure of the correlation between two "
"                series at a given frequency."


-1- " Cyclical Behavior and Periodicity "
# one Peroidic process
x(t) = A cos(2*pi*w*t + o)
A: "heigh/amplitude of the function"
w: "Frequency index"
o: "Start point of cosine function"

# >>> Easier form to analyze
x(t) = U1 cos(2*pi*w*t) + U2 sin(2*pi*w*t)

x(t): "Trend at t"
U1: A cos(o) "Normally distributed variables "
U2: -A sin(o) "Normally distributed variables"
sqrt(U1^2 + U2^2): "heigh/amplitude of the function"
w: "frequency index"
tan^-1(-U2/U1): "Start point of cosine function"

# A mixtures of periodic series with multiple frequenceies and amplitudes
x(t) = SUM(k)[ Uk1 cos(2*pi*w*t) + Uk2 sin(2*pi*w*t) ]

** " R Example -------------------------------------------------------- "
x1 = 2*cos(2*pi*1:100*6/100) + 3*sin(2*pi*1:100*6/100) # perodic frequency1
x2 = 4*cos(2*pi*1:100*10/100) + 5*sin(2*pi*1:100*10/100) # perodic frequenct2
x3 = 6*cos(2*pi*1:100*40/100) + 5*sin(2*pi*1:100*40/100) # perodic frequency3

x = x1 + x2 + x3 # Mixture of multiple periodic trends into one

plot.ts(x1, ylim=c(-10,10), main="x1") 
plot.ts(x2, ylim=c(-10,10), main="x2") 
plot.ts(x3, ylim=c(-10,10), main="x3") 
plot.ts(x, ylim=c(-10,10), main="x") 
" --------------------------------------------------------------------- "


** " [Periodogram] - it indicates which frequency components in time series are large in magnitude and which are small "
# With previous x var
P = Mod(2*fft(x)/100)^2; Fr = 0:99/100
plot(Fr, P, type="o", xlab="frquency", ylab="scaled periodogram") # fft - fast fourier transform

# example 2
t = 1:200
plot.ts( x <- 2*cos(2*pi*.2*t)*cos(2*pi*.01*t))
lines(cos(2*pi*.19*t) + cos(2*pi*.21*t), col=2)
Px = Mod(fft(x))^2; plot(0:199/200, Px, type='o') # the periodogram

# example 3 - with 'star' data
n = length(star)
par(mfrow=c(2,1), mar=c(3,3,1,1), map=c(1.6,.6,0))
plot(star, ylab="star magnitude", xlab="day")
Per = Mod(fft(star-mean(star)))^2/n
Freq = (1:n -1)/n

plot(Freq[1:50], Per[1:50], type="h", lwd=3, ylab="Periodogram", xlab="Frequency")
u = which.max[Per[1:50]]   "should 22  frequency=22/600=.035 cycles/day"
uu = which.max(Per[1:50][-u])    "should be 25  frequency=25/600=.041 cycles/day"
text(.05, 7000, "24 day cycle"); test(.027, 9000, "29 day cycles") # Add text
y = cbind(1:50, Freq[1:50], Per[1:50]); y[order(y[,3]),] # Sort by high magnitude frequency components





-2- " The Spectral Density - Spectrum Plot (population based) "

" It is the fundamental frequency domain tool -- density function of autocovariance function "

# Simulated Data
** " R Example --------------------------------------------------------- "
par(mfrow=c(3,1))
arma.spec(log="no", main="Whitenoise") # All frequency equals - straight line
arma.spec(ma=.5, log="no", main="Moving Average") # High in low frequencies and decaying to high
arma.spec(ar=c(1,-.9), log="no", main="Autoregression") # First order, so there is a peak at around 1.6 freq
arma.spec(ar=c(1,-.9), xlim=c(.15,.151), n.freq=100000) 
" ---------------------------------------------------------------------- "




-3- " Periodogram and Discrete Fourier Transform (DFT) - (sample based)"

" Sample based version of Spectral Density "
** " Discrete Fourier Transformation - DFT "
" {1,2,3,4} dataset"
(dft = fft(1:4)/sqrt(4))
(idft = fft(dft, inverse=TRUE)/sqrt(4))
(Re(idft)) # Keep it real


** " Generate Periodogram "
library(astsa)
par(mfrow=c(2,1))
soi.per = mvspec(soi, log="no")
abline(v=1/4, lty=2)
rec.per = mvspec(rec, log="no")
abline(v=1/4, lty=2)

# Calculate conf interval for series
soi.per$spec[40]  # 0.97223; soi pgrm at freq 1/12 = 40/480
soi.per$spec[10]  # 0.05372; soi pgrm at freq 1/48 = 10/480

U = qchisq(.025,2) # 0.05063
L = qchisq(.975,2) # 7.37775

2*soi.per$spec[10]/L
2*soi.per$spec[10]/U

2*soi.per$spec[40]/L
2*soi.per$spec[40]/U





-4- " Nonparametric Spectral Estimation "

" No assumption made about the spectral density "

** " Averaged Periodogram "
" Easier to spot the true peak of frequency "
soi.ave = mvspec(soi, kernel=kernel('daniell',4), log="no")

** " Smoothing Periodogram "
soi.smo = mvspec(soi, kernel=kernel("modified.daniell",c(3,3)), taper=.1, log="no") # taper - smooth little peak - stop leakage

# ------ Effect of tapering
s0 = mvspec(soi, spans=c(7,7), plot=FALSE) # without taper
s1 = mvspec(soi, spans=c(7,7), taper=.5, plot=FALSE) # full taper
plot(s1$freq, s1$spec, log="y", type="l", ylab="Spectrum", xlab="frequency")
lines(s0$freq, s0$spec, lty=2) # dash line




-5- " Parameteric Spectral Estimation "

" Assuming a function for spectral density -- ARMA, AR, ... "

** " Use AR to approximate spectral density function for Periodogram "
spaic = spec.ar(soi, log="no")    # fit best model via AIC and plot the result spectrum
abline(v=frequency(soi)*1/52, lty=3) # El Nino peak

(soi.ar = ar(soi, order.max=30)) # estimate and AICs
dev.new()
plot(1:30, soi.ar$aic[-1], type="o") # Plot AICs

# ----------- Plot AIC and BIC for a range of orders
n = length(soi)
AIC = rep(0, 30) -> AICc -> BIC
for (k in 1:30) {
	sigma2 = ar(soi, order=k, aic=FALSE)$var.pred
	BIC[k] = log(sigma2) + (k*log(n)/n)
	AICc[k] = log(sigma2) + ((n+k)/(n-k-2))
	AIC[k] = log(sigma2) + ((n+2*k)/n)
}
IC = cbind(AIC, BIC+1)
plot.ts(IC, type="o", xlab="p", ylab="AIC / BIC")




-6- " Multiple Series and Cross-Spectra "

" There are a few jointly stationary series, for example, x(t) and y(t) "
" Correlation indexed by frequency, called the 'cohenrence' --> 'cross-spectrum' "

** " R- Example ------------------------------------------------------------ "
sr = mvspec(cbind(soi,rec), kernel=("daniell",9), plot=FALSE) # use squared coherence 
sr$df # df = 35.8652
f = qf(.999, 2, sr$df-2) # = 8.529792 - F-statistics at alpha level = .001
C = f/(18+f) # = 0.321517 (Any value exceed this value means significant freqs)

plot(sr, plot.type = "coh", ci.lty = 2)
abline(h = C)
" -------------------------------------------------------------------------- "




-7- " Linear Filters "

" How linear filters can be used to extract signals from a time series "
" Identify slower frequency - vs - identify faster frequency "

** " Differencing Filter - (identify faster frequency) "
fd.soi = diff(soi)
plot(fd.sol) # more faster frequency enhanced
spectrum(fd.soi, spans=9, log="no") # spectral analysis

** " Moving Average Filter - (identify slower frequency) "
k = kernel("modified.daniell", 6) # filter weights
ma.soi = kernapply(soi, k) # 12 month filter
plot(ma.soi) # more slower frequency enhanced
spectrum(ma.soi, spans=9, log="no") # spectral analysis




-8- " Lagged Regression Models "
 
y(t) = SUM(r) beta(r)*x(t-r) + v(t)

r: "+infinite ~ -infinite"
v(t) ~ "stationary noise process"

** " R- Example -------------------------------------------------------- "
LagReg(soi, rec, L=15, M=32, threshold=6)

"    lag s      beta(s)  "
"        5      -18.47   "
"        6      -12.26   "
"        7      -8.53    "
"        8      -6.98    "
" rec(t) = ....., where alpha = 65.97 "
" y(t) = 66 - 18.5*x(t-5) - 12.3*x(t-6) - 8.5*x(t-7) - 7*x(t-8) + w(t) "

# Inverse relationship of M1
LagReg(rec, soi, L=15, M=32, inverse=TRUE, threshold=.01)

"    lag s      beta(s)  "
"        4      0.01593  "
"        5      -0.0212  "
" soi(t) = ....., where alpha = 0.41 "
" x(t) = .41 + .016*y(t+4) - .02*y(t+5) + v(t) "

" P - 220 ??? "
" ---------------------------------------------------------------------- "




-9- " Signal Extraction and Optimum Filtering "

" Deep dive in Chapter [7] ... "




-10- " Spectral Analysis of Multidimensional Series "

" Multidimensional series of the form x(s) -> s = {s1, s2, s3, ..., sr) * ia an r-dimensional vector of spatial coordinates "
" or a combination of space and time coordinates. "

# ---- Plotting 2 dimensional periodogram of soiltemp data
** " R - Example -------------------------------------------------------------------------------------------------------------------------------------- "
soiltemp " 64 X 36 dimension dataset "
per = Mod(fft(soiltemp-mean(soiltemp))/sqrt(64*36))^2 # Calculate the multidimensional spectrum -- 
# Plotting process
per2 = cbind(per[1:32,18:2], per[1:32,1:18])
per3 = rbind(per2[32:2,], per2)

par(mar=c(1,2.5,0,0)+.1)
persp(-31:31/64, -17:17/36, per3, phi=30, theta=30, expand=.6, ticktype="detailed", xlab="cycles/row", ylab="cycles/column", zlab="Periodogram Ordinate")
" ------------------------------------------------------------------------------------------------------------------------------------------------------ "









































[4] >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " Additional Time Domain Topics "

" Some of the topics be considered special and advanced topics in the time domain "


-1- " Long Memory ARMA and Fractional Differencing "
" Usually ARMA process been referred as 'short memory process' since coeiffiencts are exponentially decaying -- ACF effect exponentially decaying "
" Long memeory time series is considered 'intermidiate compromises between the short memory ARMA and fully integrated non-stationary process. "
" The easiest way to generate a long memory series is to use the differencing operator '(1-B)^d' for fractional 'd', say 0 < d < 0.5 "
" Long memeory time series data tend to exhibit sample autocorrelations that are not necessarily large but persist for a long time."


(1-B)^d * x(t) = w(t) # Fractional noise 

w(t): "Still whitenosie"
d: "fraction"

# Show long memeory data ACF
acf(log(varve), 100) # log varve dataset
acf(cumsum(rnorm(1000)), 100) # random walk


** " Fractional ARIMA -- ARFIMA(p,d,q) " # 'd' Order of difference in ARIMA, fractional..
library(fracdiff)
lvarve = log(varve) - mean(log(varve)) # detrend series
varve.fd = fracdiff(lvarve, nar=0, nma=0, M=30) 
varve.fd$d # = 0.3845
varve.fd$stderror.dpq # = 4.589514e-06
res.fd = diffseries(log(varve),varve.fd$d) # Residual


** " Long memeory (Franctional) Spectra "
series = log(varve)
d0 = .1 # initiate value of d
n.per = nextn(length(series)) 
m = (n.per)/2
per = Mod(fft(series-mean(series))[-1])^2 # remove 0 freq 
per = per/n.per # and scale the peridogram
g = 4*(sin(pi*((1:m)/n.per))^2) # sin equation of spectra 
# Function to calculate -log likelihood
whit.like = function(d){
	g.d = g^d
	sig2 = (sum(g.d*per[1:m])/m)
	log.like = m*log(sig2) - d*sum(log(g)) + m
	return(log.like)
}
# Estimation
(est = optim(d0, whit.like, gr=NULL, method="L-BFGS-B", hessian=TRUE, lower=-.5, upper=.5, control=list(trace=1,REPORT=1)))
est$par # d.hat
1/sqrt(est$hessian) # se(hat)
g.dhat = g^est$par; sig2 = sum(g.dhat*per[1:m])/m # sig2hat

"???"
library(fracdiff)
fdGPH(log(varve), bandw=.9) # m = n^bandw
" dhat = 0.383     se(dhat) = 0.041 "

" P - 249 ** Combined long & short "




-2- " Unit Root Testing "

" A unit root test provides a way to test whether a series is a 'random walk' (Null) or it is a causal process (Alternative) "

" -- Dickey-Fuller (DF) Test " > " P - 252 "
" -- Augmented Dickey-Fuller (ADF) Test " > " P - 252 " 
" -- Phillips-Perron (PP) Test " > " P - 252 "

library(tseries)
adf.test(log(varve), k=0) # DF Test
" ..... Alternative ....... "
adf.test(log(varve)) # ADF Test
".......Alternative........ "
pp.test(log(varve)) # PP test
".......Alternative........ "



-3- " GARCH Models -- GARCH(p,q) "

" Generalized autoregressive conditionally heteroscedastic Models "
" While ARMA models are used to model the conditional mean of a process when the conditional variance was constant, "
" GARCH models are used to model the changes in variance (volatility) "

r(t) = { x(t) - x(t-1) } / x(t-1) # Growth rate
r(t) = a(t)e(t)
a(t)^2 = a(0) + a(1)*r(t-1)^2 # ARCH(1)

a(t)^2 = a(0) + a(1)*r(t-1)^2 + b(1)*a(t-1)^2 # GARCH(1,1)

" ARMA process (conditional mean) + GARCH process (Conditional Variance) "

# Simulation Data Example
u = sarima(diff(log(gnp)), 1, 0, 0)
acf2(resid(u$fit)^2, 20)

library(fGarch)
summary(garchFit(~arma(1,0)+garch(1,0), diff(log(gnp)))) # AR(1) + ARCH(1)

# Analyze Time series
library(xts) # load djia dataset
djiar = diff(log(djia$Close))[-1]
acf2(djiar) # exhibits some autocorrelation 
acf2(djiar^2) # oozes autocorrelation 

library(fGarch)
summary(djia.g <- garchFit(~arma(1,0)+garch(1,1), data=djiar, cond.dist='std'))
plot(djia.g)





-4- " Threshold Models "

" It allows for changes in ARMA models' coefficients over time (based on previous values)  -- series seasonal but not perfect seasonal "
" Threshold - AR ==> TAR "
" Threshold - MA ==> TMA "
" Threshold - ARMA ==> TARMA "

x(t) = ...r(1)...r(2)... , ...r(k)... # which ... x(t) in? r - threshold

(k)... each AR, MA, ARMA processes

** " Identify thresshold "
diff.x(t) = x(t) - x(t-1)
diff.x(t-1) = x(t-1) - x(t-2)
plot(diff.x(t), diff.x(t-1)) # identify threshold - nonlinear, where on diff.x(t-1) 'trun'?

library(tsDyn)
vignette("tsDyn") # for package details
(u = setar(dflu, m=4, thDelay=0, th=.05)) # fit the model and view result
(u = setar(dflu, m=4, thDelay=0)) # let program fit threshold (-.036)
BIC(u); AIC(u) # If you want to try other models; m=3 works well too
plot(u) # Graph - ?plot.setar for information

" P- 265 "




-5- " Lagged Regression and Transfer Function Modeling "

y(t) = y(t-k) + x(t-j)

" Predict Series x(t) based on its own lag x(t-k) and other series log y(t-k) "

" R Example -------------------------------------------------------- "
# Causal relationship
soi.d = resid(lm(soi~time(soi), na.action=NULL)) # detrend soi series
acf2(soi.d) # suggest AR(!) model
fit = arima(soi.d, order=c(1,0,0)) # AR(1)
ar1 = as.numeric(coef(fit)[1]) # AR coefficient = 0.5875 used below
soi.pw = resid(fit) # ------------------------------------------ pre-whitened detrend soi series with coe = 0.5875
rec.fil = filter(rec, filter=c(1, -ar1), sides=1) # ------------ Filtered (pre-whitened detrend) rec series with coe = 0.5875

ccf(soi.pw, rec.fil, ylab="CCF", na.action=na.omit, panel.first=grid()) # Plot CCF between whitened detrend soi series and filtered rec series
" Negative lags indicates soi leads rec, so -> rec ~ soi ... "

# Modeling 
soi.d = resid(lm(soi~time(soi), na.action=NULL)) # detrend soi series
fish = ts.intersect(rec, RL1=lag(rec,-1), SL5=lag(soi.d,-5)) # Create lag model --> " rec(t) ~ rec(t-1) + soi(t-5) "
(u = lm(fish[,1]~fish[,2:3], na.action=NULL)) # Run regression 
acf2(resid(u)) # identify residual follows a AR(1) series
(arx = sarima(fish[,1], 1,0,0, xreg=fish[,2:3])) # rec = lag-regression(mean) + AR(1)(residual) ==> Final model

pred = rec + resid(arx$fit) # 1-step-ahead prediction
ts.plot(pred, rec, col=c('gray90',1), lwd=c(7,1)) # visual prediction with real data (u-l predictions)

" ------------------------------------------------------------------ "



-6- " Multivariate ARMAX Models "

" Many datasets involve more than one time series, and we are often interested in the possible dynamics relating all series. "
" In the situation, we are interested in modeling and forecasting k X 1 vector valued time series - x(t) = {x1(t), x2(t), ... , xk(t)}, t = 0, -+1, -+2, ... "

y(i)(t) = b1(i)*z1(t) + b2(i)*z2(t) + ... + br(i)*zr(t) + w(i)(t), for i in k

y(t) = {y1(t), y2(t), ... , yk(t)} " K X 1 y vector "




** " First Order Vector Autoregressive Model -- VAR(1) "
x(t) = a + M*x(t-1) + w(t) 

M: " k X k transition matrix - dependence of x(t) on x(t-1) "

" R - Example --------------------------------------------------------- "
# Quick example
library(vars)
x = cbind(cmort, tempr, part) # Three series
summary(VAR(x, p=1, type='both')) # 'both' fits constant and trend

# Example 2
library(vars)
x = cbind(cmort, tempr, part) # Three series
VARselect(x, lag.max=10, type="both") # Identify which order the best by AIC, AICc, BIC
" if use BIC recommended -- p = 2 "
# Model
summary(fit <- VAR(x, p=2, type="both"))
acf(resid(fit), 52) # ACF plot pair-wised
serial.test(fit, lags.pt=12, type="PT.adjusted") # Q-test - normal?
# Predict
(fit.pr = predict(fit, n.ahead = 24, ci = 0.95)) # 4 weeks ahead
fanchart(fit.pr) # plot prediction + error
" --------------------------------------------------------------------- " 



** " Generalized Vector ARMA Model -- ARMAX "

" Extending univariate ARMA models to multivariates -- P - 280 "
" autoregressive operator - p280" | " moving average operator - p280"

" R - Example VARMA(2,1) ---------------------------------------------- "
library(marima)
model = define.model(kvar = 3, ar = c(1,2), ma = c(1)) # Define VARMA(2,1) model
arp = model$ar.pattern; map = model$ma.pattern # extract AR, MA patterns
cmort.d = resid(detr <- lm(cmort ~ time(cmort), na.action=NULL)) # Detrend cmort series
xdata = matrix(cbind(cmort.d, tempr, part), ncol=2) # combine 3 series together

fit = marima(xdata, ar.pattern=arp, ma.pattern=map, means=c(0,1,1), penalty=1) # fit models
# Residual analysis
innove = t(resid(fit)); plot.ts(innove); acf(innove, na.action=NULL)
# Fitted value to prediction
pred = ts(t(fitted(fit))[,1], start=start(cmort), freq=frequency(cmort)) + detr$$coef[2]*time(cmort) 
# plot prediction with data
plot(pred, ylab="xxxxx", lwd=2, col=4); points(cmort) 

" --------------------------------------------------------------------- "



































[5] >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " State Space Models -- Dynamic Linear Model (DLM) "

" A very general model that subsumes a whole class of special cases of interest in much the same way that linear regression does "
" is the sate-space model or the dynamic linear model. "

" A state-space Model: (1) There is a hidden or laten process x(t) called the state process. "
"                      (2) The state process is assumed to be a Markov Process - x(t-1) conditioned independent to x(t+1) by x(t) "



**[1] " Linear Gaussian State-Space Model -- Dynamic Linear Model (DLM) "

# State Equation
x(t) = M * x(t-1) + w(t)
M: "p-dimensional vector"
w: "P X 1 iid N() "

# Observation Equation
y(t) = A * x(t) + v(t)
y(t): "q-dimensional vector"
A: "q X p matrix, q can be smaller or larger than p"
v: "iid N()"

# Example -- 3 variables - estimate the missing values

y(t) = (y1(t), y2(t), y3(t))

" x(t) = M * x(t-1) + w(t) "

(x1(t))   (O11 O12 O13)   (x1(t-1))   (w1(t))
(x2(t)) = (O21 O21 O23) X (x2(t-1)) + (w2(t))
(x3(t))   (O31 O32 O33)   (x3(t-1))   (w3(t))

" Model estimate unkonwn parameters "







**[2] " Filtering, Smoothing, Forecasting "
" Filtering -- train-time-s = t"
" Prediction -- train-time-s < t"
" Smoothing -- train-time-s > t"









**[3] " Maximum Likelihood Estimation "






**[4] " Missiong Data Modifications "






**[5] " Structural Models "





**[6] " State-Space Model with Correlated Errors "






**[7] " Bootstrapping State Space Models "






**[8] " Smoothing Splines & Kalman Smoother "







**[9] " Hidden Markov Model (HMM) & Switching Autoregressions "







**[10] " Dynamic Linear Model with Switching "






**[11] " Stochastic Volatility "






**[12] " Bayesian Analysis of State-Space Models "










































































