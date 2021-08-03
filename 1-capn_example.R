### Load libraries
library(tidyverse)
library(capn)
library(repmis)

### Set git branch
gitbranch <- "master"

### Load external functions (system_fns.R)
source(paste0("https://raw.githubusercontent.com/efenichel/capn_stuff/",gitbranch,"/system_fns.R")) #will need to fix in the end

### Load script to process raw data (data_setup.R)
source(paste0("https://github.com/efenichel/capn_stuff/raw/",gitbranch,"/data_setup.R"))

### Get data
source_data(paste0("https://github.com/efenichel/capn_stuff/raw/",gitbranch,"/KSwater_data.RData")) #Rdata file upload
ksdata <- KSwater_data #save KSwater_data as ksdata 

## STRUCTURE of ksdata 
#The object ksdata is a list of 7 lists. Each of the 7 lists corresponds to a groundwater management district (1-5), 
# the outgroup(6), or the entire state of Kansas(7).
# Each list will have 11 named elements, the number reference the elements in the list:
# [1] $gmdnum: int   1:7
# [2] $mlogitcoeff: df with dim = (#cropcodes x 23) containing the coefficients and intercept terms from the multinomial logit model
# [3] $mlogitmeans: df containing the mean for each of the variables run in the mlogit model
# [4] $cropamts: df containing the mean acres planted to each of the 5 crop cover types for each cropcode
# [5] $watercoeff: df containing the water withdrawal regression coefficients and intercept
# [6] $wwdmeans: df containing the means for each of the variables in the water withdrawal regression
# [7] $costcropacre: df dim=(1x5) containing the cost per acre of planting each of the 5 crops
# [8] $cropprices: df dim=(5x1) containing the per unit prices of each of the crops
# [9] $meanwater: num mean AF water in the region of interest
# [10]$recharge: num recharge rate for the region of interest
# [11]$watermax: num upper bound of domain for node space, max water observed in region.

### Set region
my.region <- 7

### Create data structure
if (!exists("region")){ region <- my.region }
region_data <- ksdata[[region]] 

### Process data
gw.data <- datasetup(region) 

# the gw.data data repackages parameters and means, see the datasetup code. 
# the return is of the form 
# list(crop.coeff, crop.amts, alpha, beta, gamma, gamma1, gamma2, crop.prices, cost.crop.acre)

### Economic Parameters
dr <- 0.03    # discount rate

### System parameters
recharge <- region_data[['recharge']]    # units are inches per year constant rate

### capN parameters
order <- 10        # approximaton order
NumNodes <- 100    # number of nodes
wmax <- region_data[['watermax']]        # This sets the the upper bound on water amount to consider

### Prepare capN
Aspace <- aproxdef(order, 0, wmax, dr)      # defines the approximation space
nodes <- chebnodegen(NumNodes, 0, wmax)     # define the nodes

### Prepare for simulation
simuData <- matrix(0, nrow = NumNodes, ncol = 5)


### Simulate at nodes
for(j in 1:NumNodes){
  simuData[j, 1] <- nodes[j]                            # water depth nodes
  simuData[j, 2] <- sdot(nodes[j], recharge, gw.data)   # change in stock over change in time
  simuData[j, 3] <- 0 - WwdDs1(nodes[j], gw.data)       # d(sdot)/ds, of the rate of change in the change of stock
  simuData[j, 4] <- ProfDs1(nodes[j], gw.data)          # Change in profit with the change in stock
  simuData[j, 5] <- profit(nodes[j], gw.data)           # profit
}


### Recover approximating coefficents
pC <- paprox(Aspace, simuData[, 1], simuData[, 2], simuData[, 3], simuData[, 4])  #the approximated coefficent vector for prices
pC

### Project shadow prices, value function, and inclusive wealth
waterSim <- psim(pcoeff = pC,
                 stock = simuData[ ,1],
                 wval = simuData[ ,5],
                 sdot = simuData[ ,2])

### Convert to data.frame 
waterSim <- as.data.frame(waterSim)

cat("if everything runs well the next line should say 17.44581", "\n")
cat("At 21.5 acre feet of water, the shadow price is" , psim(pC, 21.5)$shadowp, "\n")



### Plot water shadow price function
ggplot() + 
  geom_line(data = waterSim, aes(x = stock, y = shadowp), color = 'blue') +
  labs(x= "Stored groundwater", y = "Shadow price")  +
  theme(axis.line = element_line(color = "black"), 
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA))

### Plot value function
lrange <- 6 # the closest nodes to zero have some issues. 

ggplot() + 
  geom_line(data = waterSim[lrange :100,], aes(x = stock[lrange :100], y = vfun[lrange :100]), color = 'blue') +
  xlim(0, 120) +
  ylim(0, 2600) +
  labs(x= "Stored groundwater", y = "Intertemporial Welfare")  +
  theme(axis.line = element_line(color = "black"), 
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA))




### Check on results
testme <- psim(pcoeff = pC, 
               stock = c(18.5, 21.5), 
               wval = c(profit(18.5, gw.data), profit(21.5, gw.data)),
               sdot = c(sdot(18.5, recharge, gw.data), sdot(21.5, recharge, gw.data)))
print(testme)
testme









