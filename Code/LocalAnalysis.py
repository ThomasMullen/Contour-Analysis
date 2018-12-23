# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 10:53:36 2018

@author: Alexander
"""

# =============================================================================
# https://stackoverflow.com/questions/48981986/plotting-in-spherical-coordinates-given-the-radial-distance
# =============================================================================

#Created: 27/11/2018

#import librarys
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


#Directory Main
tomLocalAnalysis = "/Users/Tom/Documents/University/ProstateCode/LocalAnalysis/"
tomRadialMap = "/Users/Tom/Documents/University/ProstateCode/Data/60x60 Data/RadialDifference/"
tomProstateLink = "Tom/Documents/University/ProstateCode/Data/60x60 Data/GlobalDifference/"


# Create containers to store patients prostate maps
patientMapRecurrenceContainer = []
patientMapNonRecurrenceContainer = []

# Get total number of patients

patientMap = pd.read_csv(r"/Users/Tom/Documents/University/ProstateCode/Data/OARlocations/Prostate.csv",header=None)

  
        
# print(patientMap.shape)


# =============================================================================
# Make arrays for theta and phi axes labels
# =============================================================================
# Create Arrays
phi = []; theta =[]
for i in range(0,120):
    phi.append('')
for i in range(0,60):
    theta.append('')
# Define ticks
phi[0] = 0; phi[30] = 90; phi[60] = 180; phi[90] = 270; phi[119] = 360;
theta[0] = 90; theta[30] = 0; theta[59] = -90


# =============================================================================
# # Display 2D Heat maps
# =============================================================================

# f, (recurrenceMean, recurrenceVar) = plt.subplots(1, 2)
bladder = sns.heatmap(patientMap, center=0,xticklabels=phi,yticklabels=theta)
bladder.set(ylabel='Theta, $\dot{\Theta}$', xlabel='Azimutal, $\phi$')
fig = bladder.get_figure()
# fig.savefig(tomLocalAnalysis + "16Frac" + "60x60meanRecurrenceMap.png")
plt.show()
fig.clear()
