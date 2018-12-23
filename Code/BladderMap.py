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
tomLocalAnalysis = "/Users/Tom/Documents/University/ProstateCode/LocalAnalysis/Final/OARLocationsinPython/"
tomRadialMap = "/Users/Tom/Documents/University/ProstateCode/Data/60x60 Data/RadialDifference/"
tomProstateLink = "Tom/Documents/University/ProstateCode/Data/60x60 Data/GlobalDifference/"


# Create containers to store patients prostate maps
patientMapRecurrenceContainer = []
patientMapNonRecurrenceContainer = []

# Get total number of patients

patientBladderMap = pd.read_csv(r"/Users/Tom/Documents/University/ProstateCode/Data/OARlocations/bladder.csv",header=None).as_matrix()
patientSVMap = pd.read_csv(r"/Users/Tom/Documents/University/ProstateCode/Data/OARlocations/SemiV_ADMIRE.csv",header=None).as_matrix()
patientRectumMap = pd.read_csv(r"/Users/Tom/Documents/University/ProstateCode/Data/OARlocations/rectum.csv",header=None).as_matrix()
patientProstateMap = pd.read_csv(r"/Users/Tom/Documents/University/ProstateCode/Data/OARlocations/prostate.csv",header=None).as_matrix()
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
        
OARcontainer = np.zeros((60,120,4)) # Recurrence containter

# Make OAR map block colour
for x in range(0,60):
	for y in range(0,120):
		# Set Bladder to 0
		if patientBladderMap[x,y] < 40:
			patientBladderMap[x,y] = 100
		else:
			patientBladderMap[x,y] = -100

for x in range(0,60):
	for y in range(0,120):
		# Set Rectum to 0
		if patientRectumMap[x,y] < 40:
			patientRectumMap[x,y] = 75
		else:
			patientRectumMap[x,y] = -100

for x in range(0,60):
	for y in range(0,120):
		# Set Rectum to 0
		if patientSVMap[x,y] < 40:
			patientSVMap[x,y] = 50
		else:
			patientSVMap[x,y] = -100

for x in range(0,60):
	for y in range(0,120):
		# Set Rectum to 0
		patientProstateMap[x,y] = 20*patientProstateMap[x,y]

OARcontainer[:,:,0] = patientBladderMap[:,:]
OARcontainer[:,:,1] = patientRectumMap[:,:]
OARcontainer[:,:,2] = patientSVMap[:,:]
OARcontainer[:,:,3] = patientProstateMap[:,:]


# OARsNames = ["Bladder","Rectum","SV"]

emptymap = np.zeros((60,120))

for x in range(0,60):
	for y in range(0,120):
		for i in range(0,4):		
			emptymap[x][y] = emptymap[x][y] + OARcontainer[x][y][i]

# =============================================================================
# # Display 2D Heat maps
# =============================================================================

# f, (recurrenceMean, recurrenceVar) = plt.subplots(1, 2)
bladder = sns.heatmap(emptymap, center=0,xticklabels=phi,cbar=1,yticklabels=theta)
bladder.set(ylabel='Theta, $\dot{\Theta}$', xlabel='Azimutal, $\phi$')
fig = bladder.get_figure()
# fig.savefig(tomLocalAnalysis + "wAllOARs.png")
plt.show()
fig.clear()
