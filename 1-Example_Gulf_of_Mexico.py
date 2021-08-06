
######################################################################################
# {capn} example: Guld of Mexico (GOM)
# Updated: 3/30/2017
# Original script written by Eli Fenichel, 2013 (in MATHEMATICA)
# Updated in R script by Seong Do Yun and Eli Fenichel
# Reference: Fenichel & Abbott (2014)
# File location: system.file("demo", "GOM.R", package = "capn")
######################################################################################

import pandas as pd 

import numpy as np 
import matplotlib.pyplot as plt
from libs.capn import *

plt.style.use('seaborn-whitegrid')


## parameters from Fenichel & Abbott (2014)
r = 0.3847                              # intrinsick growth rate
param = pd.DataFrame({'r': [r]})

param['k'] = 359016000                      # carry capacity
param['q'] = 0.00031729344157311126         # catchability coefficient
param['price'] = 2.70                       # price
param['cost'] = 153.0                       # cost
param['alpha'] = 0.5436459179063678         # tech parameter
param['gamma'] = 0.7882                     # pre-ITQ management parameter
param['y'] = 0.15745573410462155            # system equivalence parameter
param['delta'] = 0.02                       # discount rate

param['order'] = 50                         # Cheby polynomial order
param['upperK'] = param['k']                # upper K
param['lowerK'] = 5*10**6                    # lower K
param['nodes'] = 500                        # number of Cheby poly nodes






## functions from Fenichel & Abbott (2014)

# Effort function x(s) = ys^gamma
def effort(s, Z):
  return Z['y'][0] * s ** Z['gamma'][0]

# Catch function (harvest) h(s, x) = q(y^alpha)(s^gamma * alpha)
def catch(s, Z):
  return Z['q'][0] * effort(s, Z) ** Z['alpha'][0] * s

# Profit function w(s, x) price * h(s, x) - cost * x(s)
# w(s, x) price * q(y^alpha)(s^gamma * alpha) - cost * ys^gamma
def profit(s, Z):
  return Z['price'][0] * catch(s, Z) - Z['cost'][0] * effort(s, Z)

# Evaluated dst/dt (biological growth function)
def sdot(s, Z):
  return Z['r'][0] * s * (1 - s / Z['k'][0]) - catch(s, Z)

# Evaluated dw/ds (derivate of profit function)
def dwds(s, Z):
  return (Z['gamma'][0] * Z['alpha'][0] + 1) * Z['price'][0] * Z['q'][0] * (Z['y'][0] ** Z['alpha'][0]) * (s ** (Z['gamma'][0] * Z['alpha'][0])) - Z['gamma'][0] * Z['cost'][0] * Z['y'][0] * (s ** (Z['gamma'][0] - 1))  

# Evaluated (d/ds) * (dw/ds)
def dwdss(s, Z):
  return (Z['gamma'][0] * Z['alpha'][0] + 1) * Z['gamma'][0] * Z['alpha'][0] * Z['price'][0] * Z['q'][0] * (Z['y'][0] ** Z['alpha'][0]) * (s ** (Z['gamma'][0] * Z['alpha'][0] - 1)) - Z['gamma'][0] * (Z['gamma'][0] - 1) * Z['cost'][0] * Z['y'][0] * (s ** (Z['gamma'][0] - 2)) 

# Evaluated dsdot/ds
def dsdotds(s, Z):
  return Z['r'][0] - 2 * Z['r'][0] * s / Z['k'][0] - (Z['gamma'][0] * Z['alpha'][0] + 1) * Z['q'][0] * (Z['y'][0] ** Z['alpha'][0]) * (s ** (Z['gamma'][0] * Z['alpha'][0]))

# Evaluated (d/ds) * (dsdot/ds)
def dsdotdss(s, Z):
  return -2 * Z['r'][0] / Z['k'][0] - (Z['gamma'][0] * Z['alpha'][0] + 1) * Z['gamma'][0] * Z['alpha'][0] * Z['q'][0] * (Z['y'][0] ** Z['alpha'][0]) * (s ** ((Z['gamma'][0] * Z['alpha'][0] -1)))



## shadow prices
# prepare capN
Aspace = approxdef(param['order'], 
                   param['lowerK'],
                   param['upperK'],
                   param['delta']) #defines the approximation space


nodes = chebnodegen(param['nodes'],
                     param['lowerK'],
                     param['upperK']) #define the nodes


# prepare for simulation
simuDataV = pd.DataFrame({
  'nodes': nodes, 
  'sdot': sdot(nodes, param), 
  'profit': profit(nodes, param)})


simuDataP = pd.DataFrame({
  'nodes': nodes, 
  'sdot': sdot(nodes, param), 
  'dsdotds': dsdotds(nodes, param),
  'dwds': dwds(nodes, param)})


simuDataPdot = pd.DataFrame({
  'nodes': nodes, 
  'sdot': sdot(nodes, param), 
  'dsdotds': dsdotds(nodes, param),
  'dsdotdss': dsdotdss(nodes, param),
  'dwds': dwds(nodes, param),
  'dwdss': dwdss(nodes, param)})



vC = vapprox(Aspace, simuDataV)  #the approximated coefficent vector for prices

pC = papprox(Aspace,
             simuDataP.iloc[:, 0],
             simuDataP.iloc[:, 1],
             simuDataP.iloc[:, 2],
             simuDataP.iloc[:, 3])  #the approximated coefficent vector for prices


pdotC = pdotapprox(Aspace,
             simuDataPdot.iloc[:, 0],
             simuDataPdot.iloc[:, 1],
             simuDataPdot.iloc[:, 2],
             simuDataPdot.iloc[:, 3],
             simuDataPdot.iloc[:, 4],
             simuDataPdot.iloc[:, 5])  #the approximated coefficent vector for prices


GOMSimV = vsim(vC, 
              simuDataV.iloc[:, 0], 
              profit(nodes, param))



GOMSimP = psim(pC,
                simuDataP.iloc[:, 0],
                profit(nodes,param),
                simuDataP.iloc[: ,1]) 


GOMSimPdot = pdotsim(pdotC,
                      simuDataPdot.iloc[:, 0],
                      simuDataPdot.iloc[:, 1],
                      simuDataPdot.iloc[:, 2],
                      profit(nodes,param),
                      simuDataPdot.iloc[:, 4])


# Plot shadow prices
#%%
fig, axs = plt.subplots(3)
fig.subplots_adjust(hspace=.5)
axs[0].plot(nodes, GOMSimV['shadowp'], color='blue');
axs[1].plot(nodes, GOMSimP['shadowp'], color='red');
axs[2].plot(nodes, GOMSimPdot['shadowp'], color='green');
axs[0].title.set_text('V Apprximation of Price')
axs[1].title.set_text('P Approximation of Price')
axs[2].title.set_text('P-dot Approximation of Price')
fig.supxlabel('Stock Size')
fig.supylabel('Shadow Price ($)')
plt.show()
