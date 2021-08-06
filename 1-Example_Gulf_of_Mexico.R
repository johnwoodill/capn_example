
######################################################################################
# {capn} example: Guld of Mexico (GOM)
# Updated: 3/30/2017
# Original script written by Eli Fenichel, 2013 (in MATHEMATICA)
# Updated in R script by Seong Do Yun and Eli Fenichel
# Reference: Fenichel & Abbott (2014)
# File location: system.file("demo", "GOM.R", package = "capn")
######################################################################################

library(capn)
rm(list=ls())

## parameters from Fenichel & Abbott (2014)
r <- 0.3847                              # intrinsick growth rate
param <- as.data.frame(r)

param$k <- 359016000                      # carry capacity
param$q <- 0.00031729344157311126         # catchability coefficient
param$price <- 2.70                       # price
param$cost <- 153.0                       # cost
param$alpha <- 0.5436459179063678         # tech parameter
param$gamma <- 0.7882                     # pre-ITQ management parameter
param$y <- 0.15745573410462155            # system equivalence parameter
param$delta <- 0.02                       # discount rate

param$order <- 50                         # Cheby polynomial order
param$upperK <- param$k                   # upper K
param$lowerK <- 5*10^6                    # lower K
param$nodes <- 500                        # number of Cheby poly nodes

## functions from Fenichel & Abbott (2014)

# Effort function x(s) = ys^gamma
effort <- function(s, Z){
  Z$y * s ^ Z$gamma
}

# Catch function (harvest) h(s, x) = q(y^alpha)(s^gamma * alpha)
catch <- function(s, Z){
  Z$q * effort(s, Z) ^ Z$alpha * s
}

# Profit function w(s, x) price * h(s, x) - cost * x(s)
# w(s, x) price * q(y^alpha)(s^gamma * alpha) - cost * ys^gamma
profit <- function(s, Z){
  Z$price * catch(s, Z) - Z$cost * effort(s, Z)
}

# Evaluated dst/dt (biological growth function)
sdot <- function(s, Z){
  Z$r * s * (1-s / Z$k) - catch(s, Z)
}

# Evaluated dw/ds (derivate of profit function)
dwds <- function(s, Z){
  (Z$gamma * Z$alpha + 1) * Z$price * Z$q * (Z$y ^ Z$alpha) * (s ^ (Z$gamma * Z$alpha)) -
    Z$gamma * Z$cost * Z$y *(s ^ (Z$gamma - 1))
}

# Evaluated (d/ds) * (dw/ds)
dwdss <- function(s, Z){
  (Z$gamma * Z$alpha + 1) * Z$gamma * Z$alpha * Z$price * Z$q * (Z$y ^ Z$alpha) * (s ^ (Z$gamma * Z$alpha - 1)) -
    Z$gamma * (Z$gamma - 1) * Z$cost * Z$y * (s ^ (Z$gamma - 2))
}

# Evaluated dsdot/ds
dsdotds <- function(s, Z){
  Z$r - 2 * Z$r * s / Z$k - (Z$gamma * Z$alpha + 1) * Z$q * (Z$y ^ Z$alpha) * (s ^ (Z$gamma * Z$alpha))
}

# Evaluated (d/ds) * (dsdot/ds)
dsdotdss <- function(s, Z){
  -2 * Z$r / Z$k - 
    (Z$gamma * Z$alpha + 1) * Z$gamma * Z$alpha * Z$q * (Z$y ^ Z$alpha) * (s ^ ((Z$gamma * Z$alpha - 1)))
}


## shadow prices
# prepare capN
Aspace <- aproxdef(param$order, 
                   param$lowerK,
                   param$upperK,
                   param$delta) #defines the approximation space

nodes <- chebnodegen(param$nodes,
                     param$lowerK,
                     param$upperK) #define the nodes

# prepare for simulation
simuDataV <- cbind(nodes, 
                   sdot(nodes, param), 
                   profit(nodes, param))

simuDataP <- cbind(nodes, 
                   sdot(nodes, param),
                   dsdotds(nodes, param), 
                   dwds(nodes, param))

simuDataPdot <- cbind(nodes, 
                      sdot(nodes, param),
                      dsdotds(nodes, param),
                      dsdotdss(nodes, param),
                      dwds(nodes, param),
                      dwdss(nodes ,param))

# recover approximating coefficents
vC <- vaprox(Aspace, simuDataV)  #the approximated coefficent vector for prices

pC <- paprox(Aspace,
             simuDataP[,1],
             simuDataP[,2],
             simuDataP[,3],
             simuDataP[,4])  #the approximated coefficent vector for prices

pdotC <- pdotaprox(Aspace,
                   simuDataPdot[,1],
                   simuDataPdot[,2],
                   simuDataPdot[,3],
                   simuDataPdot[,4],
                   simuDataPdot[,5],
                   simuDataPdot[,6])

sum(pdotC$coefficient)

# project shadow prices and wealth
GOMSimV <- vsim(vC, 
                as.matrix(simuDataV[,1], ncol=1), 
                profit(nodes,param))

sum(GOMSimV$shadowp)

GOMSimP <- psim(pC,
                simuDataP[,1],
                profit(nodes,param),
                simuDataP[,2]) 

GOMSimPdot <- pdotsim(pdotC,
                      simuDataPdot[,1],
                      simuDataPdot[,2],
                      simuDataPdot[,3],
                      profit(nodes,param),
                      simuDataPdot[,5])


sum(GOMSimPdot$shadowp)


# Three price curves
plot(nodes,GOMSimV$shadowp, type='l', lwd=2, col="blue",
     ylim = c(0,15),
     xlab="Stock size, s",
     ylab="Shdow price")
lines(nodes, GOMSimP$shadowp, lwd=2, col="red")
lines(nodes, GOMSimPdot$shadowp, lwd=2, col="green")
